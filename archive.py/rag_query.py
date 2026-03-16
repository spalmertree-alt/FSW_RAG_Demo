import os

from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")
CORPUS_RESOURCE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o"
MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)


def query_rag_corpus(query_text: str) -> None:
    print(f"\n{'=' * 40}")
    print(f"Querying: '{query_text}'")
    print(f"{'=' * 40}")

    full_prompt = (
        f"Question: {query_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. You are a Maritime Law Expert. Answer ONLY using the provided documents.\n"
        "2. You MUST cite the specific Rule Number, Annex, or Section ID for every single claim.\n"
        "3. Format citations in brackets, e.g., [Rule 10(c)] or [Annex IV].\n"
        "4. If the answer is not in the documents, reply: 'I cannot find this in the official regulations.'\n"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[CORPUS_RESOURCE_ID]
                        )
                    )
                ],
            ),
        )

        print("\n>>> RAG ANSWER:")
        print(response.text)

        print("\n>>> CITATION VERIFICATION:")
        metadata = response.candidates[0].citation_metadata
        if metadata and metadata.citation_sources:
            for idx, citation in enumerate(metadata.citation_sources, start=1):
                print(f"  {idx}. Verified Source: {citation.uri}")
        else:
            print("(!) WARNING: No citation metadata returned. The model may be answering from general knowledge.")

    except Exception as exc:
        print(f"Error occurred: {exc}")


if __name__ == "__main__":
    query_rag_corpus("What is the lighting requirement for a vessel aground?")
    # query_rag_corpus("What are the distress signals listed in the Annex?")
