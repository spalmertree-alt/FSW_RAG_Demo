import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# 1. Configure Gemini
# Replace with your actual API key
genai.configure(api_key="AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU")

# 2. Safety Settings
# We disable safety filters because maritime questions often discuss 
# "collisions," "explosions," or "death," which can trigger false positives.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 3. Initialize Model (Paid Tier Configuration)
# Temperature 0.0 forces the model to be strict/deterministic.
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"temperature": 0.0},
    safety_settings=safety_settings
)

# 4. Load your CSV (UPDATED to UTF-8)
# on_bad_lines='skip' ensures one messy row doesn't crash the script
try:
    df = pd.read_csv("ror_questions_tag.csv", encoding="utf-8", on_bad_lines='skip')
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print("Error: 'ror_questions.csv' not found. Check the file path.")
    exit()

# 5. Define Tags
tags = ["#inland", "#international", "#anchor", "#overtaking", "#lights", "#sound",
        "#horns", "#whistle", "#collision", "#passing", "#crossing", "#channel",
        "#dock", "#berth", "#sail", "#sailing", "#barge", "#head_on", "#anchorage",
        "#tug", "#astern", "#tow", "#towed", "#fog"]

# 6. Define the Prompting Function
def llm_tag_question(question_text):
    prompt = f"""
    You are a maritime rules expert. Analyze the following exam question.
    Select relevant tags from this exact list: {", ".join(tags)}.
    
    Rules:
    1. Only output tags from the provided list.
    2. If no tags apply, output "NO_TAGS".
    3. Output the tags separated by a single space.
    4. Do not provide introductions or explanations.
    
    Question: "{question_text}"
    """
    try:
        # Since you are on the Paid Tier, we do not need artificial delays (time.sleep)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # If the API fails (network issue), return ERROR so you can filter/retry later
        return "ERROR_API_FAIL"

# 7. Apply Gemini tagging with a Progress Bar
# We use tqdm so you can see the estimated time of completion
tqdm.pandas(desc="Tagging Rows")
df["Meta_Data"] = df["Question_Text"].progress_apply(llm_tag_question)

# 8. Save the result
df.to_csv("ror_questions_gemini_tagged.csv", index=False, encoding="utf-8")
print("Process Complete. Saved to ror_questions_gemini_tagged.csv")