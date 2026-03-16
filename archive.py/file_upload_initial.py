import os
import time

from google import genai
from google.genai import types

# --- CONFIGURATION ---------------------------------------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")
CORPUS_DISPLAY_NAME = "Maritime Rules of the Road (COLREGs)"
PDF_FILES = [
    "Rules_of_the_Road.pdf",
    "ROTR_Slick_Sheet.pdf",
    "ROTR_Guide.pdf",
    "Navigation_Rules_Standard_Size.pdf",
]

# Instantiate the Gemini client once.
client = genai.Client(api_key=API_KEY)


def upload_file(store_name: str, pdf_path: str) -> None:
    """Upload a single PDF into the File Search store."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' not found next to this script.")

    print(f"Uploading {pdf_path} ...")
    op = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=pdf_path,
        config={"display_name": os.path.basename(pdf_path)},
    )

    while not op.done:
        print(".", end="", flush=True)
        time.sleep(2)
        op = client.operations.get(op)

    print("\nUpload complete and indexed!")


def create_store(display_name: str) -> types.FileSearchStore:
    """Create a brand-new File Search store."""
    store = client.file_search_stores.create(config={"display_name": display_name})
    print(f"Created File Search Store: {store.name}")
    return store


def create_or_update_rag():
    """Create the store and ingest all configured PDFs."""
    store = create_store(CORPUS_DISPLAY_NAME)

    for pdf in PDF_FILES:
        try:
            upload_file(store.name, pdf)
        except Exception as exc:
            print(f"Failed to ingest '{pdf}': {exc}")

    print("\n--- SUCCESS ---")
    print(f"Store Name: {store.display_name}")
    print(f"Store ID:   {store.name}")
    print("Save the Store ID so you can reuse it later.")


if __name__ == "__main__":
    create_or_update_rag()