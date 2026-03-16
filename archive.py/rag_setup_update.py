import time
import os
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU" 
EXISTING_STORE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o" # Add your store ID here if you already created one to avoid duplicates

# Map files to specific descriptions to help the AI distinguish them
FILES_CONFIG = {
    "Rules_of_the_Road.pdf": "Official_Regulation_Textbook_Source",
    "ROTR_Slick_Sheet.pdf": "Quick_Reference_Guide",
    "Navigation_Rules_Standard_Size.pdf": "Standard_Navigation_Rules",
    "ROTR_Guide.pdf": "Learner_Guide_Book",
    "ror_test.csv": "Question_and_Answer_Database_CSV" # Crucial distinction
}

client = genai.Client(api_key=API_KEY)

def get_or_create_store(store_id=None):
    if store_id:
        print(f"--- Retrieving Existing Store: {store_id} ---")
        try:
            return client.file_search_stores.get(name=store_id)
        except Exception as e:
            print(f"Error retrieving store: {e}. Creating new.")
    
    print("--- Creating New Knowledge Base Store ---")
    return client.file_search_stores.create(
        config={"display_name": "ROTR_Knowledge_System"}
    )

def upload_files(store_name, files_map):
    print(f"--- Processing {len(files_map)} Files ---")
    operations = []

    for filename, description in files_map.items():
        if not os.path.exists(filename):
            print(f"⚠️ Missing file: {filename}")
            continue
            
        print(f"Uploading {filename} as '{description}'...")
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=filename,
                config={"display_name": description} # This helps the AI context switch
            )
            operations.append(op)
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n--- Waiting for Indexing ---")
    for op in operations:
        current_op = op
        while not current_op.done:
            print(".", end="", flush=True)
            time.sleep(2)
            current_op = client.operations.get(current_op)
    print("\nAll files processed.")

# --- EXECUTION ---
store = get_or_create_store(EXISTING_STORE_ID)
# Only upload if this is a NEW store or you added new files
if not EXISTING_STORE_ID:
    upload_files(store.name, FILES_CONFIG)

print(f"\n✅ SETUP COMPLETE. Use this Store ID in your App: {store.name}")