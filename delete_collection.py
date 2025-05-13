import os
import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Needed for client headers usually
COLLECTION_NAME = "Location"

# Check Environment Variables
if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
    print("ERROR: Ensure WEAVIATE_URL, WEAVIATE_API_KEY, and OPENAI_API_KEY are set in .env")
    sys.exit(1)

# --- Connect to Weaviate ---
print(f"Connecting to Weaviate at {WEAVIATE_URL}...")
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY} # Include OpenAI key if needed by Weaviate setup
    )
    client.is_ready() # Check connection
    print("Successfully connected to Weaviate.")
except Exception as e:
    print(f"FATAL: Failed to connect to Weaviate: {e}")
    sys.exit(1)

# --- Delete the Collection ---
try:
    print(f"Checking if collection '{COLLECTION_NAME}' exists...")
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' found. Attempting to delete...")
        client.collections.delete(COLLECTION_NAME)
        print(f"Successfully deleted collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Nothing to delete.")
except Exception as e:
    print(f"ERROR: An error occurred while trying to delete the collection: {e}")
finally:
    # --- Close Connection ---
    if 'client' in locals() and client.is_connected():
        client.close()
        print("Weaviate client connection closed.") 