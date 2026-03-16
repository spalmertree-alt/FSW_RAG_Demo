import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import base64
import json
import re

# --- CONFIGURATION ---
API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU"
STORE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o"
# CRITICAL: Use 1.5-flash. "2.5" does not exist yet.
MODEL_ID = "gemini-2.5-flash" 

# --- PAGE SETUP ---
st.set_page_config(page_title="RAG Engine", layout="wide")

# --- BACKGROUND LOADER ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #ddd;
        }}
        .stForm {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
        }}
        .stMetric {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# --- HELPER: ROBUST JSON EXTRACTOR ---
def extract_json_from_text(text):
    """
    Aggressively extracts a JSON array or object from a string using Regex.
    """
    if not text:
        return None
    text = text.strip()
    # 1. Try to find a JSON Array pattern: [ ... ]
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)
    # 2. If no array, try to find a JSON Object pattern: { ... }
    object_match = re.search(r'\{.*\}', text, re.DOTALL)
    if object_match:
        return f"[{object_match.group(0)}]"
    return text

# --- HELPER: CSV FUNCTIONS ---
@st.cache_data
def get_quiz_topics():
    csv_file = "ror_test.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if "Meta_Tags" in df.columns:
                all_tags = set()
                for tag_string in df["Meta_Tags"].dropna():
                    tags = [tag.strip() for tag in str(tag_string).split(',')]
                    all_tags.update(tags)
                return sorted(list(all_tags))
        except Exception:
            pass
    return ["General Rules"]

def get_random_questions_from_csv(selected_tags, num_questions):
    csv_file = "ror_test.csv"
    if not os.path.exists(csv_file):
        return None
    try:
        df = pd.read_csv(csv_file)
        def contains_any_tag(tag_string, selected_tags):
            if pd.isna(tag_string): return False
            tag_string = str(tag_string)
            tags_in_row = [tag.strip() for tag in tag_string.split(',')]
            return any(selected_tag in tags_in_row for selected_tag in selected_tags)
        
        filtered_df = df[df['Meta_Tags'].apply(lambda x: contains_any_tag(x, selected_tags))]
        if filtered_df.empty: return None
        
        sample_size = min(len(filtered_df), num_questions)
        sampled_df = filtered_df.sample(n=sample_size)
        return sampled_df.to_json(orient="records")
    except Exception as e:
        st.error(f"CSV Read Error: {e}")
        return None

# --- GEMINI CLIENT SETUP ---
@st.cache_resource
def get_client():
    if not API_KEY or "PASTE_YOUR" in API_KEY:
        st.error("CRITICAL: Check your API Key line in the code.")
        return None
    try:
        client = genai.Client(api_key=API_KEY)
        return client
    except Exception as e:
        st.error(f"Client Initialization Failed: {str(e)}")
        return None

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "current_quiz_source" not in st.session_state:
    st.session_state.current_quiz_source = None
if "client" not in st.session_state:
    st.session_state.client = get_client()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    mode = st.radio("Select Mode:", ["RAG Engine 🤖", "Quiz Generator 📝"])
    st.divider()
    st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.get("debug_mode", False))

# ==========================================
# MODE 1: RAG ENGINE (CHAT)
# ==========================================
if mode == "RAG Engine 🤖":
    st.title("📚 RAG Engine")
    st.caption(f"Powered by {MODEL_ID}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.client:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.client.models.generate_content(
                            model=MODEL_ID,
                            contents=f"You are a Maritime Expert. Search the files. Question: {prompt}",
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[STORE_ID]))]
                            )
                        )
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ==========================================
# MODE 2: QUIZ GENERATOR (HYBRID)
# ==========================================
elif mode == "Quiz Generator 📝":
    st.title("📝 Knowledge Check")
    
    # --- STEP 1: CHOOSE SOURCE ---
    source_type = st.radio(
        "Select Quiz Source:", 
        ["📂 From Database (CSV)", "🧠 Generate New (AI + Docs)"],
        horizontal=True
    )
    
    st.divider()

    # --- STEP 2: CONFIGURE BASED ON SOURCE ---
    
    # >>> OPTION A: CSV DATABASE <<<
    if source_type == "📂 From Database (CSV)":
        available_topics = get_quiz_topics()
        col1, col2 = st.columns(2)
        with col1:
            selected_tags = st.multiselect("Select Topic Tags:", options=available_topics, default=None)
        with col2:
            num_questions = st.number_input("Count:", min_value=1, max_value=25, value=5, key="csv_num")
        
        generate_btn = st.button("Load Quiz from CSV", type="primary")
        
        if generate_btn:
            if not selected_tags:
                st.error("Please select at least one tag.")
            else:
                with st.spinner("Fetching from database..."):
                    raw_data = get_random_questions_from_csv(selected_tags, num_questions)
                    if raw_data:
                        # Use AI just to format the CSV data cleanly
                        fmt_prompt = f"Format this CSV JSON into the standard quiz schema. JSON: {raw_data} Schema: [{{question, options[], correct_answer, reference}}]"
                        resp = st.session_state.client.models.generate_content(
                            model=MODEL_ID,
                            contents=fmt_prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json")
                        )
                        json_text = extract_json_from_text(resp.text)
                        st.session_state.quiz_data = json.loads(json_text)
                        st.session_state.user_answers = {} # Reset answers
                        st.session_state.current_quiz_source = f"CSV: {', '.join(selected_tags)}"
                    else:
                        st.error("No questions found for those tags.")

    # >>> OPTION B: AI GENERATION <<<
    else: 
        col1, col2 = st.columns(2)
        with col1:
            topic_1 = st.text_input("Primary Topic", "Steering and Sailing Rules")
            num_q = st.number_input("Total Questions (1-25)", 1, 25, 5, key="ai_num")
        with col2:
            topic_pct = st.slider(f"Percentage of questions for '{topic_1}'", 0, 100, 75)
            topic_2 = st.text_input("Secondary Topic", "Sound Signals")
            
        # Calc split
        count_1 = int(num_q * (topic_pct / 100))
        count_2 = num_q - count_1
        st.caption(f"Plan: {count_1} questions on '{topic_1}' | {count_2} questions on '{topic_2}'")

        generate_btn = st.button("Generate New Quiz", type="primary")
        
        if generate_btn:
            with st.spinner("Reading manuals and generating questions..."):
                response = None 
                try:
                    # PROMPT ENGINEERING
                    prompt = f"""
                    Create a {num_q} question multiple-choice quiz based ONLY on the provided File Store documents.
                    
                    DISTRIBUTION:
                    - {count_1} questions strictly about: {topic_1}
                    - {count_2} questions strictly about: {topic_2}
                    
                    FORMAT:
                    You must return a strictly valid JSON Array. Do not wrap in markdown blocks.
                    Schema:
                    [
                        {{
                            "question": "Question text...",
                            "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                            "correct_answer": "Full text of correct option",
                            "reference": "Rule # or Page #"
                        }}
                    ]
                    
                    Verify answers against the documents.
                    """
                    
                    response = st.session_state.client.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[STORE_ID]))]
                        )
                    )
                    
                    if st.session_state.debug_mode:
                        st.info("Raw AI Response:")
                        st.text(response.text)

                    cleaned_json = extract_json_from_text(response.text)
                    
                    if not cleaned_json:
                        raise ValueError("AI returned empty or non-JSON response.")

                    st.session_state.quiz_data = json.loads(cleaned_json)
                    st.session_state.user_answers = {}
                    st.session_state.current_quiz_source = f"AI Generated: {topic_1} & {topic_2}"
                    
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.warning("⚠️ Troubleshooting info:")
                    if response:
                        st.write("The AI returned the following text (which failed to parse as JSON):")
                        st.code(response.text)
                    else:
                        st.write("The AI returned no response object.")

    # --- STEP 3: RENDER QUIZ (SHARED LOGIC) ---
    if st.session_state.quiz_data:
        st.divider()
        st.subheader(f"📝 Quiz: {st.session_state.get('current_quiz_source', 'Custom')}")
        
        with st.form("quiz_form"):
            for idx, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"**{idx+1}. {q['question']}**")
                st.radio("Select:", q['options'], key=f"q_{idx}", label_visibility="collapsed", index=None)
                st.write("---")
            
            submitted = st.form_submit_button("Submit & Grade")
            
        if submitted:
            st.subheader("📊 Results")
            score = 0
            total = len(st.session_state.quiz_data)
            
            for idx, q_item in enumerate(st.session_state.quiz_data):
                user_choice = st.session_state.get(f"q_{idx}")
                correct_choice = q_item['correct_answer']
                question_text = q_item['question']
                
                # 1. DISPLAY RESULT STATUS
                if user_choice == correct_choice:
                    score += 1
                    st.success(f"**Q{idx+1}: Correct!**")
                    st.markdown(f"**Question:** {question_text}")
                    st.markdown(f"**Correct Answer:** {correct_choice}")
                else:
                    st.error(f"**Q{idx+1}: Incorrect.**")
                    st.markdown(f"**Question:** {question_text}")
                    st.markdown(f"**Your Answer:** {user_choice}")
                    st.markdown(f"**Correct Answer:** {correct_choice}")
                
                # 2. ALWAYS DISPLAY RAG EXPLANATION (Correct or Incorrect)
                with st.expander(f"📖 Explanation for Q{idx+1}"):
                    with st.spinner("Consulting Manuals..."):
                        explanation_prompt = f"""
                        Question: {question_text}
                        Correct Answer: {correct_choice}
                        Rule Ref: {q_item.get('reference', '')}
                        
                        Search the PDF manuals. Explain WHY this is the correct answer based on the official text.
                        """
                        
                        try:
                            expl_response = st.session_state.client.models.generate_content(
                                model=MODEL_ID,
                                contents=explanation_prompt,
                                config=types.GenerateContentConfig(
                                    tools=[
                                        types.Tool(
                                            file_search=types.FileSearch(
                                                file_search_store_names=[STORE_ID]
                                            )
                                        )
                                    ]
                                )
                            )
                            st.markdown(expl_response.text)
                        except Exception as e:
                            st.warning(f"Could not load explanation: {e}")

            # Final Score Display
            percentage = (score / total * 100) if total > 0 else 0
            st.metric(label="Final Score", value=f"{score} / {total} ({percentage:.0f}%)")