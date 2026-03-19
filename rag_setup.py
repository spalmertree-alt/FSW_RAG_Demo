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

# Source directory for upload. Two modes:
#
#   MODE 1 — Raw PDFs (original behaviour):
#     Point this at the directory containing the 13 PDFs.
#     Google will chunk them automatically (coarse chunking, no page labels).
#
#   MODE 2 — Pre-chunked text files (recommended):
#     First run rag_preprocess.py to generate labeled text chunks, then
#     point SOURCE_DIR at the output folder (default: ./rag_chunks).
#     This gives the model precise page-level context for every retrieved
#     passage, which dramatically improves citations and retrieval quality
#     for large reference documents (e.g., the 800-page error event guide).
#
# Override from command line:  python rag_setup.py /path/to/source
SOURCE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "rag_chunks")

# Map PDF filenames → display names (used in MODE 1 / raw PDF upload only).
# In MODE 2 the display_name is derived from each .txt filename automatically.
FILES_CONFIG = {
    "Aruba_8325_IGSG_en_us.pdf":                                          "Aruba_8325_Installation_and_Getting_Started_Guide",
    "Aruba_8325H_IGSG_en_us.pdf":                                         "Aruba_8325H_Installation_and_Getting_Started_Guide",
    "EC-10108_Install_Guide.pdf":                                         "EdgeConnect_EC-10108_Install_Guide",
    "EC-10108_StartUp_Guide.pdf":                                         "EdgeConnect_EC-10108_StartUp_Guide",
    "error-event-message-guide.pdf":                                      "Error_and_Event_Message_Reference_Guide",
    "HPE Aruba Networking CX 8325 Switch Series-a00059009enw.pdf":        "HPE_Aruba_CX_8325_Switch_Series_Specs",
    "HPE Aruba Networking EdgeConnect SD-WAN QuickSpecs-a50004289enw.pdf":"HPE_Aruba_EdgeConnect_SD-WAN_QuickSpecs",
    "ngfw-administration.pdf":                                            "NGFW_Administration_Guide",
    "Orch_UserGuide_R960.pdf":                                            "Orchestrator_User_Guide_R960",
    "pa-1400-hardware-reference.pdf":                                     "PA-1400_Hardware_Reference_Guide",
    "pa-1400-series.pdf":                                                 "PA-1400_Series_Guide",
    "pan-os-admin.pdf":                                                   "PAN-OS_Administration_Guide",
    "xr5610-om-en-us.pdf":                                                "XR5610_Operations_Manual",
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


def _upload_one(store_name, filepath, display_name):
    """Upload a single file and return the operation (or None on failure)."""
    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=filepath,
            config={"display_name": display_name},
        )
        return op
    except Exception as e:
        print(f"  ❌ Failed to upload {os.path.basename(filepath)}: {e}")
        return None


def _wait_for_operations(operations):
    """Poll all pending operations until they complete."""
    print(f"\n--- Waiting for Indexing ({len(operations)} files) ---")
    for label, op in operations:
        current_op = op
        while not current_op.done:
            print(".", end="", flush=True)
            time.sleep(2)
            current_op = client.operations.get(current_op)
        print(f" ✅ {label}")
    print("All files processed and indexed.")


def upload_pdfs(store_name, files_map, pdf_dir):
    """MODE 1: Upload raw PDFs — Google handles chunking internally."""
    print(f"\n--- MODE 1: Uploading {len(files_map)} PDFs from '{pdf_dir}/' ---")
    print("Note: for better retrieval quality, consider running rag_preprocess.py first.\n")
    operations = []

    for filename, description in files_map.items():
        filepath = os.path.join(pdf_dir, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠️  Missing: {filepath}")
            continue
        print(f"  Uploading {filename} as '{description}'...")
        op = _upload_one(store_name, filepath, description)
        if op:
            operations.append((filename, op))

    if operations:
        _wait_for_operations(operations)
    else:
        print("No files were uploaded. Check that PDFs exist in the source directory.")


def upload_chunks(store_name, chunks_dir):
    """MODE 2: Upload pre-chunked .txt files generated by rag_preprocess.py."""
    txt_files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".txt"))
    if not txt_files:
        print(f"No .txt files found in '{chunks_dir}'. Run rag_preprocess.py first.")
        return

    print(f"\n--- MODE 2: Uploading {len(txt_files)} pre-chunked files from '{chunks_dir}/' ---\n")
    operations = []

    for filename in txt_files:
        filepath = os.path.join(chunks_dir, filename)
        # Display name = filename without extension (already descriptive, e.g.
        # "Error_and_Event_Message_Reference_Guide_p0861-0865")
        display_name = os.path.splitext(filename)[0]
        print(f"  Uploading {filename}...")
        op = _upload_one(store_name, filepath, display_name)
        if op:
            operations.append((filename, op))

    if operations:
        _wait_for_operations(operations)
    else:
        print("No files were uploaded.")


def detect_mode(source_dir):
    """Return 'chunks' if the directory has .txt files, 'pdfs' if it has .pdf files."""
    entries = os.listdir(source_dir) if os.path.isdir(source_dir) else []
    if any(f.endswith(".txt") for f in entries):
        return "chunks"
    return "pdfs"


# --- EXECUTION ---
if __name__ == "__main__":
    store = get_or_create_store(EXISTING_STORE_ID if EXISTING_STORE_ID else None)

    if not EXISTING_STORE_ID:
        mode = detect_mode(SOURCE_DIR)
        if mode == "chunks":
            upload_chunks(store.name, SOURCE_DIR)
        else:
            upload_pdfs(store.name, FILES_CONFIG, SOURCE_DIR)
    else:
        print("Using existing store. To re-upload files, clear EXISTING_STORE_ID.")

    print(f"\n{'='*60}")
    print(f"✅ SETUP COMPLETE")
    print(f"   Store ID: {store.name}")
    print(f"{'='*60}")
    print(f"\nAdd this to your Streamlit secrets (.streamlit/secrets.toml):")
    print(f'   STORE_ID = "{store.name}"')
