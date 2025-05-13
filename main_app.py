# planner.py
#
# To Run:
# 1. Make sure Weaviate instance is running and .env has correct keys.
# 2. pip install comet_ml weaviate-client "llama-index-core>=0.10.0" "llama-index-vector-stores-weaviate" "llama-index-embeddings-openai" "llama-index-llms-openai" beautifulsoup4 requests python-dotenv openai "crewai>=0.119.0" "crewai-tools>=0.44.0"
# 3. Fill in .env with:
#    OPENAI_API_KEY="sk-..."
#    WEAVIATE_URL="YOUR_WEAVIATE_CLUSTER_URL"
#    WEAVIATE_API_KEY="YOUR_WEAVIATE_API_KEY"
#    COMET_API_KEY="YOUR_COMET_API_KEY" # Optional, script handles absence
# 4. python planner.py

import os
import re
import json # Used by LlamaIndex/Weaviate, but not directly here
import requests
import weaviate
import urllib.parse as up
from bs4 import BeautifulSoup
from weaviate.auth import AuthApiKey
from llama_index.core import VectorStoreIndex, Document # Need Document for manual loading if needed
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import weaviate.classes.config as wvc # For schema definition if needed
import comet_ml # Import Comet
from opik import track
import time # Import time for potential delays
from typing import Any

# --- Environment Loading & Constants ---
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("OPIK_API_KEY") # Load Comet API Key
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")
COMET_PROJECT_NAME = "sf_trip_planner" # Comet Project Name

COLLECTION_NAME = "Location"
WIKI_CATEGORY_URL = "https://en.wikipedia.org/wiki/Category:Tourist_attractions_in_San_Francisco"
MIN_LOCATION_THRESHOLD = 40 # Trigger scrape if less than this many locations exist
SCRAPE_LIMIT = 80 # Limit number of pages to scrape for speed

# --- Check Environment Variables ---
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Ensure WEAVIATE_URL, WEAVIATE_API_KEY, and OPENAI_API_KEY are set in .env")

# Set OpenAI key for CrewAI/LlamaIndex
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Helper Functions ---

def get_weaviate_client():
    """Initializes and returns a Weaviate v4 client."""
    print("Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        client.is_ready()
        print("Successfully connected to Weaviate.")
        return client
    except Exception as e:
        print(f"FATAL: Failed to connect to Weaviate: {e}")
        exit()

def get_location_count(client):
    """Gets the total number of objects in the Location collection."""
    try:
        if client.collections.exists(COLLECTION_NAME):
            location_collection = client.collections.get(COLLECTION_NAME)
            response = location_collection.aggregate.over_all(total_count=True)
            print(f"Found {response.total_count} existing locations in Weaviate.")
            return response.total_count
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist yet.")
            return 0
    except Exception as e:
        print(f"Warning: Error checking Weaviate object count: {e}")
        return 0 # Assume 0 if check fails

def ensure_location_schema(client):
    """Creates the Location collection if it doesn't exist."""
    if not client.collections.exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
            try:
                client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
                        # Use a known available model name
                        model="text-embedding-3-small", # CHANGED FROM ada-002
                    ),
                    properties=[
                        wvc.Property(name="name", data_type=wvc.DataType.TEXT),
                        wvc.Property(name="description", data_type=wvc.DataType.TEXT),
                        wvc.Property(name="source", data_type=wvc.DataType.TEXT), # Store URL
                    ]
                )
                print(f"Collection '{COLLECTION_NAME}' created.")
            except Exception as e:
                 print(f"Warning: Failed to create collection '{COLLECTION_NAME}': {e}. Proceeding might fail.")

def scrape_and_ingest_wikipedia(client):
    """Scrapes Wikipedia category AND its subcategories (1 level deep), extracts data, and ingests into Weaviate."""
    print(f"Attempting to scrape Wikipedia category and subcategories from: {WIKI_CATEGORY_URL}")
    attraction_urls = set() # Use a set to automatically handle duplicates
    categories_to_visit = [WIKI_CATEGORY_URL]
    visited_categories = set()

    # --- Step 1: Collect URLs from Category and Subcategories --- 
    print("Collecting attraction URLs...")
    while categories_to_visit:
        current_category_url = categories_to_visit.pop(0)
        if current_category_url in visited_categories:
            continue
        visited_categories.add(current_category_url)
        print(f"  Visiting category page: {current_category_url}")

        try:
            response = requests.get(current_category_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find direct attraction links on this page
            pages_div = soup.find('div', {'id': 'mw-pages'})
            if pages_div:
                links = pages_div.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if href.startswith('/wiki/') and ':' not in href:
                        attraction_urls.add(f"https://en.wikipedia.org{href}")

            # Find subcategory links (only if we are on the *main* category page)
            if current_category_url == WIKI_CATEGORY_URL:
                subcat_div = soup.find('div', {'id': 'mw-subcategories'})
                if subcat_div:
                    links = subcat_div.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        # Add valid subcategory URLs to visit
                        if href.startswith('/wiki/Category:'):
                             subcat_url = f"https://en.wikipedia.org{href}"
                             if subcat_url not in visited_categories:
                                 print(f"    Adding subcategory to visit: {subcat_url}")
                                 categories_to_visit.append(subcat_url)
            
            # Optional: Brief delay to avoid overwhelming Wikipedia
            time.sleep(0.1) 

        except requests.exceptions.RequestException as e_cat:
            print(f"Warning: Failed to fetch category/subcategory page {current_category_url}: {e_cat}")
        except Exception as e_outer:
            print(f"Warning: Unexpected error processing category page {current_category_url}: {e_outer}")

    print(f"Collected {len(attraction_urls)} unique attraction URLs.")

    # --- Step 2: Scrape Individual Attraction Pages --- 
    locations_to_ingest = []
    scraped_count = 0
    ingested_count = 0
    urls_to_scrape = list(attraction_urls) # Convert set to list for iteration

    for page_url in urls_to_scrape:
        if scraped_count >= SCRAPE_LIMIT:
            print(f"Reached scrape limit of {SCRAPE_LIMIT}.")
            break
        
        # Extract title from URL (simple version)
        title = page_url.split('/wiki/')[-1].replace('_', ' ') 
        # Attempt to decode URL encoding if present (e.g., %27)
        try: 
            title = up.unquote(title)
        except Exception:
            pass # Ignore decoding errors

        print(f"  Scraping: {title} ({page_url})")
        scraped_count += 1
        try:
            page_response = requests.get(page_url, timeout=15)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.content, 'html.parser')

            # --- Description Extraction (Using previous robust method) ---
            description = ""
            content_parser_output = page_soup.select_one('#mw-content-text .mw-parser-output')
            if content_parser_output:
                paragraphs = content_parser_output.find_all('p')
                for p in paragraphs:
                    p_text = p.get_text(strip=True)
                    if len(p_text) > 50 and not p_text.startswith(('Coordinates:', '(')):
                        description = p_text
                        if len(description) > 300:
                            description = description[:300] + "..."
                        break
            # --- End Description Extraction ---

            if description:
                 locations_to_ingest.append({
                    "name": title,
                    "description": description,
                    "source": page_url
                 })
                 print(f"    Extracted: {description[:70]}...")
            else:
                print(f"    Warning: Could not extract a suitable descriptive paragraph for {title}")
            
            # Optional delay
            time.sleep(0.1)

        except requests.exceptions.RequestException as e_page:
            print(f"    Warning: Failed to fetch/parse page {page_url}: {e_page}")
        except Exception as e_inner:
             print(f"    Warning: Unexpected error processing {page_url}: {e_inner}")

    # --- Step 3: Ingest Data --- (Logic remains the same)
    if locations_to_ingest:
        print(f"\n--- Starting Weaviate Batch Import for {len(locations_to_ingest)} scraped locations ---")
        ensure_location_schema(client) # Recreates schema if it was deleted
        try:
            location_collection = client.collections.get(COLLECTION_NAME)
            with location_collection.batch.dynamic() as batch:
                for i, item in enumerate(locations_to_ingest):
                    properties = {
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "source": item.get("source", "")
                    }
                    batch.add_object(properties=properties)
                    if (i + 1) % 10 == 0:
                         print(f"  Added {i + 1} objects to Weaviate batch...")

            print("Batch import process finished.")
            # Attempt error check if method becomes available in future client versions
            # if hasattr(location_collection.batch, 'num_errors') and callable(location_collection.batch.num_errors):
            #     if location_collection.batch.num_errors() > 0:
            #         print(f"Warning: Errors occurred during Weaviate batch import: {location_collection.batch.errors}")
            # else:
            #     print("(Could not check for batch errors automatically)")
            ingested_count = len(locations_to_ingest)

        except Exception as e_batch:
            print(f"Warning: Error during Weaviate batch import: {e_batch}")
            ingested_count = 0
    else:
        print("No valid locations extracted from Wikipedia to ingest.")

    return ingested_count

# --- CrewAI Tools ---
class LocationSearchTool(BaseTool):
    name: str = "San Francisco Location Search"
    description: str = ("Searches a database of San Francisco tourist attractions based on user interests. "
                        "Input should be a comma-separated string of interests (e.g., 'historic landmarks, good views, art'). "
                        "Returns a multi-line string, where each item found includes its Name, Description, Source, and Relevance Score. "
                        "If no relevant items are found, it returns 'NO_RESULTS_FOUND' or the LLM's direct assessment.")
    query_engine: Any = None

    def _run(self, interests: str) -> str:
        print(f"\n>> LocationSearchTool searching for: {interests}")
        if not self.query_engine:
             return "Error: Query engine not initialized."
        try:
            # The query_engine.query() call itself performs retrieval and then synthesis.
            # We are interested in the source_nodes it retrieved.
            response_obj = self.query_engine.query(interests)

            retrieved_data_parts = []
            if hasattr(response_obj, 'source_nodes') and response_obj.source_nodes:
                print(f"  >> DIAGNOSTIC: Raw nodes retrieved by LlamaIndex (query_engine uses these for its synthesis):")
                for i, node_with_score in enumerate(response_obj.source_nodes):
                    try:
                        node_text = node_with_score.node.get_content() # Should be 'description' due to text_key
                        node_metadata = node_with_score.node.metadata or {}
                        # Access name and source within the nested 'properties' dict
                        node_properties = node_metadata.get('properties', {}) # Get the inner dict or empty if missing
                        node_name = node_properties.get('name', 'Unknown Location')
                        node_source = node_properties.get('source', 'N/A')
                        node_score = node_with_score.score if node_with_score.score is not None else 0.0

                        # Prepare a structured string for each item
                        item_detail = (
                            f"Item {i+1}:\n"
                            f"  Name: {node_name}\n"
                            f"  Description: {node_text}\n"
                            f"  Source: {node_source}\n"
                            f"  Relevance Score: {node_score:.4f}"
                        )
                        retrieved_data_parts.append(item_detail)
                        cleaned_text = node_text[:60].replace('\n', ' ') 
                        # Print full metadata for debugging
                        print(f"Node {i+1}: Name='{node_name}', Score={node_score:.4f}, Text='{cleaned_text}...', METADATA: {node_metadata}")
                    except Exception as e:
                        print(f"Node {i+1}: Error extracting details: {e}")
                        # Fallback if node structure is unexpected
                        try:
                            raw_content = node_with_score.node.get_content()
                            if raw_content:
                                retrieved_data_parts.append(f"Item {i+1} (fallback):\n  Content: {raw_content}")
                        except Exception as e_fallback:
                             print(f"    Node {i+1}: Error getting fallback content: {e_fallback}")


            if not retrieved_data_parts:
                # If no source nodes, or failed to extract from them, use the query engine's original synthesized response
                # or indicate no results if that synthesis was also empty.
                synthesized_response = str(response_obj).strip()
                print(f"  >> LocationSearchTool: No usable raw nodes extracted. Using synthesized response from query engine: '{synthesized_response[:100]}...'")
                if not synthesized_response or "empty response" in synthesized_response.lower() or \
                   "no relevant" in synthesized_response.lower() or \
                   "does not directly relate" in synthesized_response.lower() or \
                   "no information provided" in synthesized_response.lower(): # Added common LLM phrase
                    return "NO_RESULTS_FOUND" # General "not found" marker
                return synthesized_response # Return the original synthesis

            # Concatenate all item details
            final_output = "\n\n---\n\n".join(retrieved_data_parts)
            print(f">> LocationSearchTool returning {len(retrieved_data_parts)} concatenated raw items. Total length: {len(final_output)}")
            return final_output

        except Exception as e:
            print(f"Error during LocationSearchTool processing: {e}")
            # Log the full error for debugging
            import traceback
            traceback.print_exc()
            return f"Error searching for locations: Critical tool failure - {e}"

# --- Main Execution Logic ---
if __name__ == "__main__":
    client = get_weaviate_client()
    total_locations_in_db = get_location_count(client)
    results_returned = 0 # Initialize metric
    final_itinerary = "Could not generate itinerary." # Default message

    # a) Check Weaviate count and run scraper if needed
    if total_locations_in_db < MIN_LOCATION_THRESHOLD:
        print(f"\nLocation count ({total_locations_in_db}) is below threshold ({MIN_LOCATION_THRESHOLD}). Running Wikipedia ingest...")
        ingested_count = scrape_and_ingest_wikipedia(client)
        total_locations_in_db = get_location_count(client) # Re-check count after ingest
        print(f"Ingest complete. Total locations now: {total_locations_in_db}")
    else:
         print(f"Sufficient locations ({total_locations_in_db}) found. Skipping Wikipedia ingest.")

    if total_locations_in_db == 0:
         print("FATAL: No locations available in Weaviate database. Exiting.")
         client.close()
         exit()

    # --- Set up LlamaIndex for QUERYING ---
    print(f"\nInitializing LlamaIndex VectorStore for collection: {COLLECTION_NAME}")
    try:
        vector_store = WeaviateVectorStore(
            weaviate_client=client, 
            index_name=COLLECTION_NAME, 
            text_key="description"
        )
        print("Creating LlamaIndex VectorStoreIndex from existing Weaviate data...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Creating LlamaIndex Query Engine...")
        # Increase similarity_top_k to give Researcher more options
        query_engine = index.as_query_engine(similarity_top_k=5)
        print("Query Engine created.")
    except Exception as e:
        print(f"FATAL: Failed to initialize LlamaIndex query engine: {e}")
        client.close()
        exit()

    # --- Instantiate Tool with Query Engine ---
    location_tool = LocationSearchTool(query_engine=query_engine)

    # --- Define CrewAI Agents ---
    print("\nDefining CrewAI Agents...")
    researcher = Agent(
      role='San Francisco Location Researcher',
      goal="Retrieve 3-5 distinct SF locations matching user interests via the tool. Provide name/summary for each. State clearly if none found.",
      backstory="Expert SF researcher using tools for factual results. If tool returns 'NO_RESULTS_FOUND', output must reflect that.",
      verbose=False, # SET TO FALSE FOR CLEANER DEMO OUTPUT
      allow_delegation=False,
      tools=[location_tool]
    )
    planner = Agent(
      role='Creative Itinerary Planner',
      goal="Based *only* on researcher's findings, create a mini-itinerary. If findings are 'NO_RESULTS_FOUND', apologize. Otherwise, list 2-3 locations (new line, start with 🗺️ Name), briefly explaining connection to user interests ({interests}).",
      backstory="Witty planner using *only* provided context. Apologizes gracefully if context shows no results.",
      verbose=False, # SET TO FALSE FOR CLEANER DEMO OUTPUT
      allow_delegation=False
    )
    print("Agents defined.")

    # --- Define CrewAI Tasks ---
    print("\nDefining CrewAI Tasks...")
    research_task = Task(
      description="Find San Francisco locations based on these interests: {interests}. Use the Location Search Tool.",
      expected_output="A list of 3-5 distinct locations (name/summary) OR a clear statement like 'NO_RESULTS_FOUND' based on tool output.",
      agent=researcher
    )
    plan_task = Task(
      description="Review researcher findings. If 'NO_RESULTS_FOUND', create apology. Otherwise, create itinerary list (🗺️ Name per line) explaining relevance to interests: {interests}.",
      expected_output="Engaging list/paragraph (2-3 locations, 🗺️ Name per line, explaining relevance to interests: {interests}) OR an apology message if researcher found no results.",
      agent=planner,
      context=[research_task]
    )
    print("Tasks defined.")

    # --- Assemble Crew ---
    trip_crew = Crew(
        agents=[researcher, planner],
        tasks=[research_task, plan_task],
        process=Process.sequential,
        verbose=False # SET TO FALSE FOR CLEANER DEMO OUTPUT
    )

    # b) Prompt user and run CrewAI flow
    print("\n--- AI Trip Idea Generator ---")
    interests = input("Enter your interests for San Francisco (e.g., 'history, good coffee, parks'): ")
    crew_inputs = {'interests': interests}
    print("\n--- Running the Crew... ---")
    try:
        crew_output_obj = trip_crew.kickoff(inputs=crew_inputs) # Get the CrewOutput object
        # Attempt to get the raw string output from the CrewOutput object
        # Common attributes are .raw or accessing the result of the last task
        # Let's assume .raw is available or it's the direct string representation
        if hasattr(crew_output_obj, 'raw') and isinstance(crew_output_obj.raw, str):
            final_itinerary = crew_output_obj.raw
        elif isinstance(crew_output_obj, str): # If kickoff directly returns string in some versions
            final_itinerary = crew_output_obj
        else: # Fallback or needs more specific access if .raw isn't it
            # This might require inspecting crew_output_obj structure more closely
            # For example, accessing the output of the last task if available
            # For now, convert to string as a general fallback, might not be ideal format
            print("Warning: Could not find a direct .raw string attribute on CrewOutput. Converting object to string.")
            final_itinerary = str(crew_output_obj) 

        print("\n\n--- Crew Run Complete! --- ")
        # print(result) # Don't print raw result object, final_itinerary is the string
    except Exception as e:
        print(f"\nError running CrewAI: {e}")
        final_itinerary = "Error: Could not generate itinerary due to CrewAI execution failure." # Default message

    # --- Process Result for Links ---
    print("\n--- Final Itinerary ---")
    if not isinstance(final_itinerary, str):
        print(f"Warning: final_itinerary is not a string, it is: {type(final_itinerary)}. Using its string representation.")
        final_itinerary_str = str(final_itinerary)
    else:
        final_itinerary_str = final_itinerary

    # Update regex to capture only the name after the emoji, stopping at a colon or newline
    # [^:\n]+ captures one or more characters that are NOT a colon or a newline.
    locations_found = re.findall(r'🗺️\s*([^:\n]+)', final_itinerary_str)
    results_returned = len(locations_found)
    print(final_itinerary_str) # Print the planner's output string

    if locations_found:
        print("\n--- Map Links ---")
        for name in locations_found:
            name = name.strip() # Clean up whitespace
            if name: # Avoid empty strings
                map_url = f"https://maps.google.com/?q={up.quote(name + ', San Francisco, CA')}"
                print(f"📍 {name} → {map_url}")
    elif "could not find" in final_itinerary_str.lower() or "no specific recommendations" in final_itinerary_str.lower():
         print("\n(No specific locations found to generate map links)") # Handle apology case
    else:
        print("\n(Could not extract specific locations to generate map links)")


    # c) Log metrics to Comet
    print("\n--- Logging Metrics ---")
    if COMET_API_KEY:
        try:
            # Check if experiment already exists - reuse if possible (simplistic check)
            # More robust check would query Comet API
            experiment = comet_ml.Experiment( # Use Experiment, not ExistingExperiment unless ID known
                api_key=COMET_API_KEY,
                project_name=COMET_PROJECT_NAME,
                # workspace="YOUR_WORKSPACE" # Optional: specify workspace if needed
            )
            print("Connected to Comet ML.")
            experiment.log_metrics({
                "locations_in_db": total_locations_in_db,
                "results_returned": results_returned
            })
            print("Metrics logged to Comet.")
            print(f"View Experiment: {experiment.url}")
            experiment.end() # Explicitly end experiment
        except ImportError:
             print("Warning: comet_ml library not found. Cannot log metrics. Run: pip install comet_ml")
        except Exception as e:
            print(f"Warning: Failed to log metrics to Comet ML: {e}")
    else:
        print("COMET_API_KEY not found in .env. Skipping Comet ML logging.")


    client.close()
    print("\nWeaviate client connection closed.")
