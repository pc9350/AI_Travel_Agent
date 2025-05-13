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
from crewai.tools import BaseTool # Correct import needed
import weaviate.classes.config as wvc # For schema definition if needed
import comet_ml # Import Comet
from opik import track

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
                        vectorize_collection_name=True
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
    """Scrapes Wikipedia category, extracts data, and ingests into Weaviate."""
    print(f"Attempting to scrape Wikipedia category: {WIKI_CATEGORY_URL}")
    locations_to_ingest = []
    scraped_count = 0
    ingested_count = 0

    try:
        response = requests.get(WIKI_CATEGORY_URL, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        category_links_div = soup.find('div', {'id': 'mw-pages'})
        if not category_links_div:
            print("Warning: Could not find the category links section ('mw-pages') on the Wikipedia page.")
            return 0

        links = category_links_div.find_all('a')
        print(f"Found {len(links)} potential links in category.")

        for link in links:
            if scraped_count >= SCRAPE_LIMIT:
                print(f"Reached scrape limit of {SCRAPE_LIMIT}.")
                break

            href = link.get('href', '')
            title = link.get('title', '')

            if href.startswith('/wiki/') and ':' not in href and title:
                page_url = f"https://en.wikipedia.org{href}"
                print(f"  Scraping: {title} ({page_url})")
                scraped_count += 1
                try:
                    page_response = requests.get(page_url, timeout=15)
                    page_response.raise_for_status()
                    page_soup = BeautifulSoup(page_response.content, 'html.parser')

                    # --- More Robust Description Extraction ---
                    description = ""
                    content_parser_output = page_soup.select_one('#mw-content-text .mw-parser-output')
                    if content_parser_output:
                        # Find ALL paragraphs within the content area
                        paragraphs = content_parser_output.find_all('p') # Removed recursive=False
                        for p in paragraphs:
                            p_text = p.get_text(strip=True)
                            # Find the first paragraph that looks like real content
                            if len(p_text) > 50 and not p_text.startswith(('Coordinates:', '(')):
                                description = p_text
                                # Limit length
                                if len(description) > 300:
                                    description = description[:300] + "..."
                                break # Stop after finding the first good paragraph
                    # --- End More Robust Extraction ---

                    if description:
                         locations_to_ingest.append({
                            "name": title,
                            "description": description,
                            "source": page_url
                         })
                         print(f"    Extracted: {description[:70]}...")
                    else:
                        # Use a more specific warning
                        print(f"    Warning: Could not extract a suitable descriptive paragraph for {title}")

                except requests.exceptions.RequestException as e_page:
                    print(f"    Warning: Failed to fetch/parse page {page_url}: {e_page}")
                except Exception as e_inner:
                     print(f"    Warning: Unexpected error processing {page_url}: {e_inner}")


    except requests.exceptions.RequestException as e_cat:
        print(f"Warning: Failed to fetch Wikipedia category page: {e_cat}")
        return 0
    except Exception as e_outer:
        print(f"Warning: Unexpected error during scraping setup: {e_outer}")
        return 0

    # --- Ingest Data using Native Weaviate Batch Import ---
    if locations_to_ingest:
        print(f"\n--- Starting Weaviate Batch Import for {len(locations_to_ingest)} scraped locations ---")
        ensure_location_schema(client) # Make sure schema exists
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
            if location_collection.batch.num_errors() > 0:
                 print(f"Warning: Errors occurred during Weaviate batch import: {location_collection.batch.errors}")
            ingested_count = len(locations_to_ingest) # Consider successful additions

        except Exception as e_batch:
            print(f"Warning: Error during Weaviate batch import: {e_batch}")
            ingested_count = 0 # Reset count on failure

    else:
        print("No valid locations extracted from Wikipedia to ingest.")

    return ingested_count

# --- CrewAI Tool Definition ---
class LocationSearchTool(BaseTool):
    name: str = "Location Search Tool"
    description: str = ("Searches a database of San Francisco tourist attractions based on user interests. "
                        "Input should be a comma-separated string of interests (e.g., 'historic landmarks, good views, art'). "
                        "Returns a raw list of matching locations and their descriptions, or a specific message if none found.")
    query_engine: object # Store query engine instance

    def _run(self, interests: str) -> str:
        """Use the LlamaIndex query engine to find relevant locations."""
        print(f"\n>> LocationSearchTool searching for: {interests}")
        if not self.query_engine:
             return "Error: Query engine not initialized."
        try:
            response = self.query_engine.query(interests)
            response_text = str(response).strip()
            print(f">> LocationSearchTool raw response: {response_text[:100]}...") # Log snippet

            # Check if the response indicates no results were found
            # LlamaIndex might return an empty string or a default message.
            # Adapt this check based on observed behavior for empty results.
            if not response_text or "empty response" in response_text.lower() or "no relevant" in response_text.lower():
                 print(">> LocationSearchTool concluded: No relevant results.")
                 return "NO_RESULTS_FOUND" # Return a specific marker

            print(f">> LocationSearchTool found: {response_text[:100]}...")
            return response_text # Return the raw response text
        except Exception as e:
            print(f"Error during LocationSearchTool query: {e}")
            return f"Error searching for locations: {e}"

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
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=COLLECTION_NAME)
    print("Creating LlamaIndex VectorStoreIndex from existing Weaviate data...")
    try:
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

    # Update regex to capture name on the line after the emoji
    # The [^\n]+ part captures one or more characters that are NOT a newline.
    locations_found = re.findall(r'🗺️\s*([^\n]+)', final_itinerary_str)
    results_returned = len(locations_found)
    print(final_itinerary_str) # Print the planner's output string

    if locations_found:
        print("\n--- Map Links ---")
        for name in locations_found:
            name = name.strip().rstrip('.') # Clean up name
            # Further clean up potential trailing explanations from the planner
            name = name.split(' - ')[0].split(', which')[0].split(' because')[0].strip()
            if name: # Avoid empty strings
                map_url = f"https://maps.google.com/?q={up.quote(name + ', San Francisco, CA')}"
                print(f"📍 {name} → {map_url}")
    elif "could not find" in final_itinerary_str.lower() or "no specific recommendations" in final_itinerary_str.lower():
         print("(No specific locations found to generate map links)") # Handle apology case
    else:
        print("(Could not extract specific locations to generate map links)")


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
