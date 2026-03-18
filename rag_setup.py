import time
import os
import sys
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# Set your API key here or via environment variable GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# If you already have a store, paste its ID here to skip creation.
# Leave empty to create a new store.
EXISTING_STORE_ID = ""

# Directory containing the 13 installation PDFs.
# Override from command line:  python rag_setup.py /path/to/your/pdfs
PDF_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\SeanPalmertree\OneDrive - LearnToWin, Inc\Desktop\FSW Docs"

# Map PDF filenames to display names that help the AI distinguish documents
FILES_CONFIG = {
    "Aruba_8325_IGSG_en_us.pdf": "Aruba_8325_Installation_and_Getting_Started_Guide",
    "Aruba_8325H_IGSG_en_us.pdf": "Aruba_8325H_Installation_and_Getting_Started_Guide",
    "EC-10108_Install_Guide.pdf": "EdgeConnect_EC-10108_Install_Guide",
    "EC-10108_StartUp_Guide.pdf": "EdgeConnect_EC-10108_StartUp_Guide",
    "error-event-message-guide.pdf": "Error_and_Event_Message_Reference_Guide",
    "HPE Aruba Networking CX 8325 Switch Series-a00059009enw.pdf": "HPE_Aruba_CX_8325_Switch_Series_Specs",
    "HPE Aruba Networking EdgeConnect SD-WAN QuickSpecs-a50004289enw.pdf": "HPE_Aruba_EdgeConnect_SD-WAN_QuickSpecs",
    "ngfw-administration.pdf": "NGFW_Administration_Guide",
    "Orch_UserGuide_R960.pdf": "Orchestrator_User_Guide_R960",
    "pa-1400-hardware-reference.pdf": "PA-1400_Hardware_Reference_Guide",
    "pa-1400-series.pdf": "PA-1400_Series_Guide",
    "pan-os-admin.pdf": "PAN-OS_Administration_Guide",
    "xr5610-om-en-us.pdf": "XR5610_Operations_Manual",
}

# --- CLIENT SETUP ---
if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable or edit API_KEY in this script.")

client = genai.Client(api_key=API_KEY)


def get_or_create_store(store_id=None):
    """Retrieve an existing store or create a new one."""
    if store_id:
        print(f"--- Retrieving Existing Store: {store_id} ---")
        try:
            return client.file_search_stores.get(name=store_id)
        except Exception as e:
            print(f"Error retrieving store: {e}. Creating new.")

    print("--- Creating New Knowledge Base Store ---")
    return client.file_search_stores.create(
        config={"display_name": "Network_Installation_Manuals"}
    )


def upload_files(store_name, files_map, pdf_dir):
    """Upload all PDFs to the file search store and wait for indexing."""
    print(f"\n--- Processing {len(files_map)} Files from '{pdf_dir}/' ---")
    operations = []

    for filename, description in files_map.items():
        filepath = os.path.join(pdf_dir, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠️  Missing file: {filepath}")
            continue

        print(f"  Uploading {filename} as '{description}'...")
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=filepath,
                config={"display_name": description},
            )
            operations.append((filename, op))
        except Exception as e:
            print(f"  ❌ Failed to upload {filename}: {e}")

    if not operations:
        print("\nNo files were uploaded. Check that PDFs exist in the pdfs/ directory.")
        return

    print(f"\n--- Waiting for Indexing ({len(operations)} files) ---")
    for filename, op in operations:
        current_op = op
        while not current_op.done:
            print(".", end="", flush=True)
            time.sleep(2)
            current_op = client.operations.get(current_op)
        print(f" ✅ {filename}")

    print("All files processed and indexed.")


# --- EXECUTION ---
if __name__ == "__main__":
    store = get_or_create_store(EXISTING_STORE_ID if EXISTING_STORE_ID else None)

    # Upload files if this is a new store (no EXISTING_STORE_ID provided)
    if not EXISTING_STORE_ID:
        upload_files(store.name, FILES_CONFIG, PDF_DIR)
    else:
        print("Using existing store. To re-upload files, clear EXISTING_STORE_ID.")

    print(f"\n{'='*60}")
    print(f"✅ SETUP COMPLETE")
    print(f"   Store ID: {store.name}")
    print(f"{'='*60}")
    print(f"\nAdd this to your Streamlit secrets (.streamlit/secrets.toml):")
    print(f'   STORE_ID = "{store.name}"')
