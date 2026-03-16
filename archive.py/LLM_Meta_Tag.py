import pandas as pd
import google.generativeai as genai
import concurrent.futures
from tqdm import tqdm

# --- CONFIGURATION ---
API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU"

# NOTE: Double check your model version availability. 
# Common versions: "gemini-1.5-flash", "gemini-2.0-flash-exp"
MODEL_NAME = "gemini-2.5-flash" 

INPUT_FILE = "ror_questions_tag.csv"
OUTPUT_FILE = "ror_questions_gemini_tagged_test.csv"
WORKER_COUNT = 30  # Adjust based on your Paid Tier limits (usually 20-50 is safe)

# 1. Configure Gemini
genai.configure(api_key=API_KEY)

# 2. Safety Settings
# We disable safety filters to prevent false positives on maritime terms like "collision"
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 3. Initialize Model
# We use a slightly higher temperature (0.1) to allow for smart tag generation, 
# but keep it low enough to ensure strict formatting (commas).
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"temperature": 0.1}, 
    safety_settings=safety_settings
)

# 4. Define the Helper Function (Prompt Logic)
def process_row(args):
    """
    Worker function to process a single row.
    args is a tuple: (index, question_text)
    """
    index, question_text = args
    
    # Updated Prompt for Open-Ended Tagging
    prompt = f"""
    Act as a Maritime Rules Expert and Curriculum Developer.
    Analyze the following exam question to generate accurate metadata tags.
    
    Tagging Strategy:
    1. Topic Hierarchy: Identify the broad category (e.g., Steering and Sailing, Lights and Shapes).
    2. Specific Rules: If a specific COLREGs rule applies, tag it (e.g., "Rule 14", "Rule 19").
    3. Scenario: Describe the situation (e.g., "Head-on", "Overtaking", "Restricted Visibility").
    4. Asset Types: Identify vessels/objects involved (e.g., "Power-driven", "Sailing Vessel", "Buoy").
    
    Formatting Rules:
    - Start EVERY tag with a hashtag (#).
    - Output ONLY the tags.
    - Separate tags with COMMAS.
    - Keep tags concise (1-3 words).
    - Do not number the tags.
    - Example output: #International, #Rule_15, #CrossingSituation, #PowerDriven
    
    Question: "{question_text}"
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean up result: remove trailing periods or newlines
        clean_text = response.text.strip().replace("\n", ", ")
        return index, clean_text
    except Exception as e:
        return index, "ERROR_API_FAIL"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 5. Load Data
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8", on_bad_lines='skip')
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        exit()

    results_dict = {}
    tasks = [(i, row["Question_Text"]) for i, row in df.iterrows()]

    print(f"Starting analysis with {WORKER_COUNT} parallel workers...")
    print(f"Using Model: {MODEL_NAME}")

    # 6. Run Parallel Processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = [executor.submit(process_row, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            index, result = future.result()
            results_dict[index] = result

    # 7. Map results back to DataFrame
    print("Mapping tags to DataFrame...")
    df["Meta_Tags"] = df.index.map(results_dict)

    # 8. Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Success! Saved to {OUTPUT_FILE}")
    
    # Optional: Print the first 3 rows to verify output format
    print("\n--- Sample Output ---")
    print(df[["Question_Text", "Meta_Tags"]].head(3))