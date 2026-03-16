import os
import google.generativeai as genai

# --- CONFIGURATION ---
# Ensure your API Key is set. Replace with your actual key or set as env variable.
# os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_API_KEY"
API_KEY = os.environ.get("AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")

# REPLACE THIS with the Store Name generated in your previous script
# It usually looks like: "corpora/1a2b3c..." or "fileSearchStores/..."
STORE_NAME = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o" 

def setup_tutor():
    genai.configure(api_key=API_KEY)
    
    # 1. Create the Tool configuration using your specific Store
    # This tells Gemini: "When you answer, look inside this folder first."
    file_search_tool = genai.protos.Tool(
        file_search=genai.protos.FileSearch(
            file_search_store_names=[STORE_NAME]
        )
    )

    # 2. Define the Model
    # Gemini 1.5 Flash is fast and great for tutoring. 
    # Use 1.5 Pro if you need very complex reasoning.
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-001',
        tools=[file_search_tool],
        system_instruction="""
            You are a friendly and helpful AI Tutor. 
            Answer questions based strictly on the provided course materials.
            If the answer is not in the documents, say "I couldn't find that in the course notes."
            Always quote the source when possible.
        """
    )
    return model

def start_chat_session(model):
    # Enable automatic function calling so the model handles the retrieval steps internally
    chat = model.start_chat(enable_automatic_function_calling=True)
    return chat

def print_citations(response):
    """Helper to extract and print which file the AI used."""
    for candidate in response.candidates:
        if candidate.citation_metadata:
            print("\n--- Sources Used ---")
            for source in candidate.citation_metadata.citation_sources:
                # This often returns the URI or partial text index
                if source.uri:
                    print(f"Reference: {source.uri}")

def main():
    if "YOUR_STORE_ID" in STORE_NAME:
        print("ERROR: You must update the STORE_NAME variable with your actual Corpus ID.")
        return

    print(f"initializing AI Tutor against Store: {STORE_NAME}...")
    
    try:
        model = setup_tutor()
        chat = start_chat_session(model)
        print("\n--- AI Tutor Ready (Type 'quit' to exit) ---")

        while True:
            user_query = input("\nStudent: ")
            if user_query.lower() in ["quit", "exit"]:
                break
            
            print("Tutor is reading documents...")
            response = chat.send_message(user_query)
            
            # Print the AI's answer
            print(f"\nTutor: {response.text}")
            
            # (Optional) Show citations
            print_citations(response)

    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Check that your API Key is correct and the Store Name exists.")

if __name__ == "__main__":
    main()