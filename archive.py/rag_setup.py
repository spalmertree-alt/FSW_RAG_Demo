import time
from google import genai
from google.genai import types

# 1. SETUP: Configure your client
client = genai.Client(api_key="AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")

def setup_rag():
    print("--- Starting Gemini File Search RAG Setup ---")
    
    # 2. UPLOAD: Create a File Store and Upload a Document
    # Create the store
    file_store = client.file_search_stores.create(
        config={"display_name": "My_Knowledge_Base"}
    )
    print(f"Created Store: {file_store.name}")
    
    # Upload a file
    print("Uploading file...")
    upload_op = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=file_store.name,
        file="manual.pdf",
        config={"display_name": "Product Manual"}
    )
    
    # 3. WAIT: Polling for processing completion
    print("Processing file (Chunking & Embedding)...")
    while not upload_op.done:
        print(".", end="", flush=True)
        time.sleep(2)
        upload_op = client.operations.get(upload_op)
    
    print("\nFile processed successfully!")
    return file_store

# Setup RAG (create store and upload file)
file_store = setup_rag()

# Tier 1 Optimized Loop
# This allows for 120 requests/minute (safe buffer for Tier 1)
print("--- Querying the Model (Tier 1 Mode) ---")

questions = [
    "Describe in detail Overtaking situations",
    "What is the minimum visibility of Masthead Lights?",
    "What is the sound signal for a vessel at anchor?",
    # ... imagine 50 more questions here
]

for q in questions:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Flash is faster and cheaper
            contents=q,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[file_store.name]
                        )
                    )
                ]
            )
        )
        print(f"Q: {q} -> A: {response.text}")
        
        # CRITICAL: The "Heartbeat" Sleep
        # Sleep 0.5 seconds between requests. 
        # This ensures you never exceed ~120 RPM, keeping you safe.
        time.sleep(0.5) 

    except Exception as e:
        print(f"Error on '{q}': {e}")
        time.sleep(5) # Back off longer if we hit a bump