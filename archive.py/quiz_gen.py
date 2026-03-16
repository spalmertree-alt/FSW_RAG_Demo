import google.generativeai as genai
import os
import time

# --- CONFIGURATION ---
genai.configure(api_key=os.environ["AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU"])

# Point this to your Question Bank file
# Ideally a text file or CSV where questions are clearly separated
QUESTION_BANK_FILE = "QuestionBank.txt" 
# Update this to the path of your question bank file

def upload_bank_to_context(file_path):
    """
    Uploads the entire Question Bank to the Gemini File API.
    Unlike RAG, this doesn't 'chunk' it; it gives the whole thing to the model.
    """
    print(f"Uploading Question Bank: {file_path}...")
    file_ref = genai.upload_file(file_path)
    
    # Wait for processing
    while file_ref.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(1)
        file_ref = genai.get_file(file_ref.name)
    
    print(f"\nReady. File Name: {file_ref.display_name}")
    return file_ref

def generate_quiz(topic, num_questions, file_ref):
    print(f"\n--- Generating {num_questions}-Question Quiz on '{topic}' ---")
    
    model = genai.GenerativeModel("gemini-1.5-flash")

    # The prompt explicitly instructs the model to act as a quiz generator
    # using ONLY the provided file.
    prompt = (
        f"You are an expert Maritime Examiner. "
        f"Using the provided Question Bank file, create a {num_questions}-question quiz "
        f"specifically about '{topic}'. \n\n"
        "Rules:\n"
        "1. Randomly select questions from the file that match the topic.\n"
        "2. Do not repeat questions.\n"
        "3. Provide the output in this format:\n"
        "   Q1: [Question Text]\n"
        "   A) [Option]\n"
        "   B) [Option]\n"
        "   ...\n"
        "   (Answer Key at the very bottom)\n"
    )

    response = model.generate_content(
        [file_ref, prompt] # We pass the file reference directly here
    )

    print(response.text)

if __name__ == "__main__":
    # 1. Upload the bank once (in a real app, you'd cache this object)
    if not os.path.exists(QUESTION_BANK_FILE):
        print(f"Error: {QUESTION_BANK_FILE} not found! Please add your question bank file.")
    else:
        bank_ref = upload_bank_to_context(QUESTION_BANK_FILE)
        
        # 2. Run the command
        # You can change these inputs to test different quizzes
        generate_quiz("Ship Lighting", 25, bank_ref)