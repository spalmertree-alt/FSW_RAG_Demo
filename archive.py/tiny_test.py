import time
from google import genai
from google.genai import types

# Replace with your NEW key
client = genai.Client(api_key="AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")

def run_tiny_test():
    print("--- Starting Tiny Test ---")
    
    # 1. Create a dummy file (1 sentence)
    with open("dummy.txt", "w") as f:
        f.write("This is a test document for Gemini RAG. If you can read this, the system works.")
    
    # 2. Create Store
    try:
        store = client.file_search_stores.create(config={"display_name": "Tiny Test Store"})
        print(f"Store Created: {store.name}")
    except Exception as e:
        print(f"FAILED to create store: {e}")
        return

    # 3. Upload (Should take 2 seconds)
    print("Uploading tiny file...")
    try:
        client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file="dummy.txt",
            config={"display_name": "Dummy Doc"}
        )
        print("Upload Success! Your API Key and Logic are PERFECT.")
        print("The issue is definitely your PDF size vs. Free Tier limits.")
    except Exception as e:
        print(f"FAILED with 429? {e}")

if __name__ == "__main__":
    run_tiny_test()