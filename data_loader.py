import weaviate
import os
import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from dotenv import load_dotenv
import weaviate.classes.config as wvc # Correct import for Configure and Property
from weaviate.auth import AuthApiKey

# Load environment variables (API Keys, URLs)
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # LlamaIndex uses this for embeddings and Weaviate might too

# Check if environment variables are loaded
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please ensure WEAVIATE_URL, WEAVIATE_API_KEY, and OPENAI_API_KEY are set in your .env file")

# --- Weaviate Client Setup (v4 Syntax - Updated) ---
print("Connecting to Weaviate...")
try:
    # Use connect_to_weaviate_cloud instead of deprecated connect_to_wcs
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        headers={ # Pass OpenAI key for potential module use (e.g., text2vec-openai)
            "X-OpenAI-Api-Key": OPENAI_API_KEY
        }
    )
    client.is_ready() # Check connection
    print("Successfully connected to Weaviate.")
except Exception as e:
    print(f"Failed to connect to Weaviate: {e}")
    exit()


# --- Define Weaviate Schema (Collection - v4 Syntax - Updated) ---
COLLECTION_NAME = "Location"

if client.collections.exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists in Weaviate.")
    # Optionally delete if you want a clean slate (USE WITH CAUTION)
    print(f"Deleting existing collection '{COLLECTION_NAME}'...")
    client.collections.delete(COLLECTION_NAME)
    print("Collection deleted.")
else:
    print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
    client.collections.create(
        name=COLLECTION_NAME,
        # Use wvc.Configure for vectorizer settings
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
            model="ada", # Specify the embedding model
            vectorize_collection_name=True # Default is True
        ),
        # Define properties using wvc.Property
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            # Using TEXT for tags joined by comma
            wvc.Property(name="tags", data_type=wvc.DataType.TEXT),
        ]
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

# --- Direct Weaviate Client Test Insertion ---
print("\n--- Attempting Direct Weaviate Insertion Test ---")
try:
    location_collection = client.collections.get(COLLECTION_NAME)
    test_object = {
        "name": "Direct Test Location",
        "description": "This is a test object inserted directly via Weaviate client.",
        "tags": "test,direct"
    }
    # The .insert() method is part of .data, not .batch directly for single objects
    uuid_returned = location_collection.data.insert(properties=test_object)
    print(f"Directly inserted object with UUID: {uuid_returned}")

    # Verify this single insertion immediately
    inserted_obj = location_collection.query.fetch_object_by_id(uuid_returned)
    if inserted_obj:
        print(f"Successfully fetched directly inserted object: {inserted_obj.properties.get('name')}")
    else:
        print("Failed to fetch directly inserted object.")

    # Check count after direct insert
    response = location_collection.aggregate.over_all(total_count=True)
    print(f"Total objects after direct insert test: {response.total_count}")

except Exception as e:
    print(f"Error during direct Weaviate insertion test: {e}")

# --- Load Data from JSON ---
try:
    with open('locations.json', 'r') as f:
        locations_data = json.load(f)
    print(f"Loaded {len(locations_data)} locations from locations.json")
except FileNotFoundError:
    print("Error: locations.json not found. Please create it with sample data.")
    exit()
except json.JSONDecodeError:
    print("Error: locations.json contains invalid JSON.")
    exit()

# --- Ingest Data using Native Weaviate Batch Import ---
print("\n--- Starting Native Weaviate Batch Import ---")
try:
    location_collection = client.collections.get(COLLECTION_NAME)
    # Use dynamic batching for simplicity
    with location_collection.batch.dynamic() as batch:
        print(f"Adding {len(locations_data)} objects to the batch...")
        for i, item in enumerate(locations_data):
            # Ensure tags are a list of strings if your schema expects text[]
            # If schema expects TEXT (like we defined), join them
            properties = {
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "tags": ", ".join(item.get("tags", [])) # Join tags for TEXT type
            }
            batch.add_object(properties=properties)
            if (i + 1) % 10 == 0:
                 print(f"  Added {i + 1} objects to batch...")

    print("Batch import process finished.")
    # Batch commits automatically upon exiting the 'with' block

    # Optional: Check for batch errors if needed
    if location_collection.batch.num_errors() > 0:
        print(f"Errors occurred during batch import: {location_collection.batch.errors}")

except Exception as e:
    print(f"Error during native Weaviate batch import: {e}")

# --- Verifying Data in Weaviate ---
print("\n--- Verifying Data in Weaviate ---")
try:
    # Ensure client is connected before querying
    if not client.is_live():
        print("Weaviate client is not live. Cannot verify data.")
    else:
        location_collection = client.collections.get(COLLECTION_NAME) # Use the variable
        response = location_collection.aggregate.over_all(total_count=True)
        print(f"Total objects found in '{COLLECTION_NAME}' collection via client: {response.total_count}")

        if response.total_count > 0:
            print("Fetching a few objects:")
            fetch_result = location_collection.query.fetch_objects(limit=3)
            for obj in fetch_result.objects:
                # Access properties safely
                name = obj.properties.get('name', 'N/A')
                print(f"  ID: {obj.uuid}, Name: {name}")

except Exception as e:
    print(f"Error querying Weaviate for verification: {e}")

# Close the client connection when done
client.close()
print("Weaviate client connection closed.")

# Optional: Quick test query
# Needs adjustment for v4 client if run here
# print("\\nTest Query (Requires v4 query syntax)") 