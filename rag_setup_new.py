"""
Create a BRAND-NEW Gemini File Search store for a new course.

IMPORTANT: This script always creates a fresh, isolated store so documents
from different courses are never mixed.  Each course in the app connects to
exactly one store — the AI can only see documents inside that store.

Usage:
    1. Place your PDF(s) in this directory.
    2. Edit the CONFIGURATION section below.
    3. Run:  python rag_setup_new.py
    4. Copy the printed Store ID into .streamlit/secrets.toml as STORE_ID_2.
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

# Display name for the NEW store (shows in Gemini console).
# Pick something descriptive so you can tell stores apart.
STORE_DISPLAY_NAME = "MCRP_3-01A_Knowledge_System"

# Map of local filenames -> display names for the store.
# Only documents listed here will be uploaded to the NEW store.
# The maritime PDFs live in a completely separate store — no overlap.
FILES_CONFIG = {
    "MCRP 3-01A.pdf": "MCRP_3-01A_Official_Reference",
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
    missing = [f for f in FILES_CONFIG if not os.path.exists(f)]
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
        print(f"  Uploading {filename} as '{description}'...")
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store.name,
                file=filename,
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
    print(f'       STORE_ID_2 = "{store.name}"')
    print()
    print("  2. The app will pick it up automatically via COURSE_NEW_SUBJECT")
    print("     in claude_demo.py (store_id_secret = STORE_ID_2_NAME).")
    print()
    print("  The maritime store (STORE_ID) is untouched and fully separate.")
    print("  Switching courses in the app swaps which store the AI can see.")


if __name__ == "__main__":
    main()
