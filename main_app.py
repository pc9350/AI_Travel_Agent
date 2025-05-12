import os
import weaviate
from weaviate.auth import AuthApiKey
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional: Set model for CrewAI/LlamaIndex if not default GPT-4
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o" 

# --- Check Environment Variables ---
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Ensure WEAVIATE_URL, WEAVIATE_API_KEY, and OPENAI_API_KEY are set in .env")

# Set OpenAI key for CrewAI/LlamaIndex
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Connect to Weaviate (v4 client) ---
print("Connecting to Weaviate...")
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY} # Needed for vector search maybe?
    )
    client.is_ready()
    print("Successfully connected to Weaviate.")
except Exception as e:
    print(f"Failed to connect to Weaviate: {e}")
    exit()

# --- Set up LlamaIndex WeaviateVectorStore for QUERYING ---
COLLECTION_NAME = "Location" # Must match the collection name used in data_loader.py
print(f"Initializing LlamaIndex VectorStore for collection: {COLLECTION_NAME}")
vector_store = WeaviateVectorStore(weaviate_client=client, index_name=COLLECTION_NAME)

print("Creating LlamaIndex VectorStoreIndex from existing Weaviate data...")
# Load index from existing vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print("Creating LlamaIndex Query Engine...")
# Create a query engine
# similarity_top_k=3 means it will retrieve the top 3 most relevant locations
query_engine = index.as_query_engine(similarity_top_k=3)
print("Query Engine created.")

# --- Define CrewAI Tool for Location Searching ---
class LocationSearchTool(BaseTool):
    name: str = "Location Search Tool"
    description: str = "Searches for San Francisco locations based on user interests. Input should be a comma-separated string of interests."

    def _run(self, interests: str) -> str:
        """Use the LlamaIndex query engine to find relevant locations."""
        print(f"\n>> LocationSearchTool searching for: {interests}")
        try:
            response = query_engine.query(interests)
            print(f"\n>> LocationSearchTool found: {response}")
            # Return the response as a string
            # Ensure response has a string representation or access .response attribute
            return str(response) 
        except Exception as e:
            print(f"Error during LocationSearchTool query: {e}")
            return f"Error searching for locations: {e}"

# Instantiate the tool
location_tool = LocationSearchTool()

# --- Define CrewAI Agents ---
print("Defining CrewAI Agents...")

# Research Agent
researcher = Agent(
  role='San Francisco Location Researcher',
  goal='Find relevant San Francisco locations based on user interests using the Location Search Tool',
  backstory=("You are an expert researcher specializing in finding points of interest in San Francisco. "
             "You use your tools to query a database of locations and return factual results."),
  verbose=True, # See agent's thought process
  allow_delegation=False,
  tools=[location_tool] # Assign the tool to this agent
)

# Planning Agent
planner = Agent(
  role='Creative Itinerary Planner',
  goal='Create a short, engaging list of 2-3 suggested places to visit in San Francisco based on the research findings',
  backstory=("You are a witty and creative travel planner. You take research findings about locations "
             "and turn them into a fun, easy-to-read suggestion list. Focus on the highlights."),
  verbose=True,
  allow_delegation=False
  # No specific tools needed, relies on LLM capabilities
)

print("Agents defined.")

# --- Define CrewAI Tasks ---
print("Defining CrewAI Tasks...")

# Task for Researcher Agent
# The description will be formatted dynamically when the crew runs
research_task = Task(
  description="Find San Francisco locations based on these interests: {interests}",
  expected_output='A list of 2-3 relevant locations with brief descriptions, pulled from the Location Search Tool results.',
  agent=researcher,
  tools=[location_tool] # Explicitly mention tool use here too
)

# Task for Planner Agent
plan_task = Task(
  description='Based on the research findings, create a short, fun suggestion list (2-3 items) for visiting San Francisco.',
  expected_output='A formatted paragraph or list highlighting 2-3 cool spots, written in an engaging tone.',
  agent=planner,
  context=[research_task] # Ensure output of research_task is input here
)

print("Tasks defined.")

# --- Assemble and Run the Crew --- 
# Moved crew assembly outside the __main__ block so it's defined globally
trip_crew = Crew(
    agents=[researcher, planner],
    tasks=[research_task, plan_task],
    process=Process.sequential,  # Run tasks sequentially
    verbose=True # Changed from 2 to boolean True
)

# --- Main Execution Logic ---
if __name__ == "__main__":
    print("\n--- AI Trip Idea Generator ---")
    interests = input("Enter your interests for San Francisco (e.g., 'history, good coffee, parks'): ")

    # Prepare inputs for the kickoff
    crew_inputs = {
        'interests': interests
    }

    # Kick off the crew with user interests
    print("\n--- Running the Crew... ---")
    result = trip_crew.kickoff(inputs=crew_inputs)

    print("\n\n--- Crew Run Complete! --- Final Result: ---")
    print(result)

    # Close Weaviate client connection
    client.close()
    print("Weaviate client connection closed.") 