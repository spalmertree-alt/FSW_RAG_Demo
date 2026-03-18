"""
Create a BRAND-NEW Gemini File Search store for the USN ATG Basic Phase course.

IMPORTANT: This script always creates a fresh, isolated store so documents
from different courses are never mixed.  Each course in the app connects to
exactly one store — the AI can only see documents inside that store.

Usage:
    1. Place the four ATG PDF files in this directory.
    2. Run:  python rag_setup_atg.py
    3. Copy the printed Store ID into .streamlit/secrets.toml as STORE_ID_3.
"""

import time
import os
import sys
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# CONFIGURATION  — edit these for your new course
# ──────────────────────────────────────────────

# Your Gemini API key (or set GEMINI_API_KEY env var)
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

# Directory where the ATG PDF files are stored
PDF_DIR = r"C:\Users\SeanPalmertree\OneDrive - LearnToWin, Inc\Desktop\ATG"

# Display name for the NEW store (shows in Gemini console).
STORE_DISPLAY_NAME = "USN_ATG_Basic_Phase_Knowledge_System"

# Map of local filenames -> display names for the store.
FILES_CONFIG = {
    "ATG User Guide App Z - FBP 26OCT23.pdf": "ATG_User_Guide_App_Z_FBP",
    "AUGM Appendix AB - READ-E 3 Guidance.pdf": "AUGM_Appendix_AB_READ-E_3_Guidance",
    "AUGM_App_AA_SBTT_25JAN19.pdf": "AUGM_App_AA_SBTT",
    "AUGM_App_ P_MOB-E.pdf": "AUGM_App_P_MOB-E",
}

# ──────────────────────────────────────────────
# SCRIPT LOGIC  — no need to edit below
# ──────────────────────────────────────────────

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set your API key in API_KEY or the GEMINI_API_KEY env var.")
        sys.exit(1)

    if not FILES_CONFIG:
        print("ERROR: Add at least one file to FILES_CONFIG before running.")
        sys.exit(1)

    # Verify all files exist before touching the API
    missing = [f for f in FILES_CONFIG if not os.path.exists(os.path.join(PDF_DIR, f))]
    if missing:
        print(f"ERROR: These files are missing from {os.getcwd()}:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    # ── Always create a NEW store ──────────────────────────
    print(f"--- Creating NEW File Search Store: {STORE_DISPLAY_NAME} ---")
    print("(This is intentionally separate from any existing stores.)")
    store = client.file_search_stores.create(
        config={"display_name": STORE_DISPLAY_NAME}
    )
    print(f"Store created: {store.name}\n")

    # ── Upload files ───────────────────────────────────────
    print(f"--- Uploading {len(FILES_CONFIG)} file(s) ---")
    operations = []

    for filename, description in FILES_CONFIG.items():
        filepath = os.path.join(PDF_DIR, filename)
        print(f"  Uploading {filename} as '{description}'...")
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store.name,
                file=filepath,
                config={"display_name": description},
            )
            operations.append((filename, op))
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Wait for indexing ──────────────────────────────────
    if operations:
        print("\n--- Waiting for indexing to complete ---")
        for filename, op in operations:
            current_op = op
            while not current_op.done:
                print(".", end="", flush=True)
                time.sleep(2)
                current_op = client.operations.get(current_op)
            print(f"  {filename} indexed.")
        print("All files indexed.\n")

    # ── Summary ────────────────────────────────────────────
    print("=" * 60)
    print("  SETUP COMPLETE — New isolated store ready")
    print("=" * 60)
    print(f"\n  Store ID:  {store.name}\n")
    print("  Next steps:")
    print("  1. Add this line to .streamlit/secrets.toml:")
    print(f'       STORE_ID_3 = "{store.name}"')
    print()
    print("  2. The app will pick it up automatically via COURSE_ATG")
    print("     in claude_demo.py (store_id_secret = STORE_ID_3_NAME).")
    print()
    print("  The maritime store (STORE_ID) and marksmanship store (STORE_ID_2)")
    print("  are untouched and fully separate.")
    print("  Switching courses in the app swaps which store the AI can see.")


if __name__ == "__main__":
    main()
