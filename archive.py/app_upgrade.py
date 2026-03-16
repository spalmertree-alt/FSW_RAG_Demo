import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import base64
import json
import re

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU" 

STORE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o" 
MODEL_ID = "gemini-2.5-flash"

# --- SAFETY SETTINGS ---
# We disable safety filters to prevent false positives on maritime terms like "collision"
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- SYSTEM INSTRUCTIONS ---
SYSTEM_INSTRUCTION = """
You are a Maritime Rules of the Road expert and Examiner.
CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base.
The knowledge base contains:
1. Rules_of_the_Road.pdf 
2. ROTR_Slick_Sheet.pdf 
3. Navigation_Rules_Standard_Size.pdf 
4. ROTR_Guide.pdf 
5. ror_test.csv 

RULES:
1. Search across ALL these documents for every question.
2. Do NOT answer from general knowledge. 
3. Always cite specific Rule numbers (e.g., "Rule 14(a)") or Annex sections.
4. If information is not found in any of these documents, state clearly that it is not available.
"""

# --- PAGE SETUP ---
st.set_page_config(page_title="RAG Engine", page_icon="⚓", layout="wide")

# --- STYLING & ASSETS ---
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
            background-attachment: fixed;
        }}
        .stChatMessage, .stForm {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# --- HELPER: ROBUST JSON EXTRACTOR ---
def extract_clean_json(text):
    if not text: return None
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx])
        
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return [json.loads(text[start_idx:end_idx])]
    except json.JSONDecodeError:
        return None
    return None

# --- HELPER: CSV FUNCTIONS ---
@st.cache_data
def get_quiz_topics(csv_file="ror_test.csv"):
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if "Meta_Tags" in df.columns:
                all_tags = set()
                for tag_string in df["Meta_Tags"].dropna():
                    tags = [tag.strip() for tag in str(tag_string).split(',')]
                    all_tags.update(tags)
                return sorted(list(all_tags))
        except: pass
    return ["General Rules"]

def get_random_questions_from_csv(selected_tags, num_questions, csv_file="ror_test.csv"):
    if not os.path.exists(csv_file): return None
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
    except: return None

# --- GEMINI CLIENT SETUP ---
@st.cache_resource
def get_client():
    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        st.error("⚠️ API Key missing.")
        return None
    try:
        return genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Client Init Failed: {str(e)}")
        return None

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am your Maritime Rules expert. Ask me any question about the Rules of the Road, or switch to the Quiz tab to test your knowledge."}
    ]
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "current_quiz_source" not in st.session_state:
    st.session_state.current_quiz_source = None
if "client" not in st.session_state:
    st.session_state.client = get_client()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚓ Maritime AI")
    mode = st.radio("Select Tool:", ["Chat Assistant 🤖", "Quiz Generator 📝"])
    st.divider()
    st.caption(f"Model: {MODEL_ID}")
    st.session_state.debug_mode = st.checkbox("Debug JSON", value=False)
    
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.rerun()

# ==========================================
# MODE 1: RAG ENGINE (CHAT)
# ==========================================
if mode == "Chat Assistant 🤖":
    st.title("📚 Regulations Chat")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about maritime rules..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        if st.session_state.client:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Searching manuals..."):
                        try:
                            tool_config = [types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[STORE_ID]
                                )
                            )]
                            
                            response = st.session_state.client.models.generate_content(
                                model=MODEL_ID,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=0.1, 
                                    system_instruction=SYSTEM_INSTRUCTION, 
                                    tools=tool_config,
                                    safety_settings=SAFETY_SETTINGS  # <--- SAFETY ADDED
                                )
                            )
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})

                            # --- FILE SEARCH CHECK ---
                            # Check if grounding metadata exists to confirm file search was used
                            file_search_used = False
                            if response.candidates and response.candidates[0].grounding_metadata:
                                file_search_used = True

                            if file_search_used:
                                st.caption("*File Search was used on Rules of the Roads documents*")
                            else:
                                st.caption("⚠️ *Note: File search may not have been triggered. The model may be using general knowledge.*")
                                
                        except Exception as e:
                            st.error(f"API Error: {str(e)}")

# ==========================================
# MODE 2: QUIZ GENERATOR
# ==========================================
elif mode == "Quiz Generator 📝":
    st.title("📝 Knowledge Check")
    
    source_type = st.radio(
        "Source:", 
        ["From Database (CSV)", "Generate New (AI + RAG)"],
        horizontal=True
    )
    st.divider()

    # --- CSV PATH ---
    if source_type == "From Database (CSV)":
        available_topics = get_quiz_topics()
        col1, col2 = st.columns(2)
        with col1:
            selected_tags = st.multiselect("Topics:", options=available_topics)
        with col2:
            num_questions = st.number_input("Count:", 1, 20, 5)
        
        if st.button("Load CSV Quiz", type="primary"):
            if not selected_tags:
                st.warning("Pick a topic.")
            else:
                with st.spinner("Formatting data..."):
                    raw_data = get_random_questions_from_csv(selected_tags, num_questions)
                    if raw_data:
                        fmt_prompt = f"""
                        Convert this CSV data into a JSON quiz.
                        Add "explanation" field for why the answer is correct based on Maritime Rules.
                        Raw Data: {raw_data} 
                        Schema: [{{question, options[], correct_answer, reference, explanation}}]
                        """
                        try:
                            resp = st.session_state.client.models.generate_content(
                                model=MODEL_ID,
                                contents=fmt_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    system_instruction=SYSTEM_INSTRUCTION,
                                    safety_settings=SAFETY_SETTINGS # <--- SAFETY ADDED
                                )
                            )
                            data = extract_clean_json(resp.text)
                            if data:
                                st.session_state.quiz_data = data
                                st.session_state.current_quiz_source = f"CSV: {', '.join(selected_tags)}"
                            else:
                                st.error("Failed to parse AI response.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("No questions found in CSV.")

    # --- AI GENERATION PATH ---
    else: 
        col1, col2 = st.columns(2)
        with col1:
            topic_1 = st.text_input("Primary Topic", "Steering and Sailing Rules")
            num_q = st.number_input("Total Questions (1-25)", 1, 25, 5, key="ai_num")
        with col2:
            topic_pct = st.slider(f"Percentage of questions for '{topic_1}'", 0, 100, 75)
            topic_2 = st.text_input("Secondary Topic", "Sound Signals")
            
        count_1 = int(num_q * (topic_pct / 100))
        count_2 = num_q - count_1
        st.caption(f"Plan: {count_1} questions on '{topic_1}' | {count_2} questions on '{topic_2}'")

        if st.button("Generate RAG Quiz", type="primary"):
            with st.spinner("Drafting questions..."):
                prompt = f"""
                Create a {num_q} question multiple-choice quiz based ONLY on the provided File Store documents.
                
                DISTRIBUTION:
                - {count_1} questions strictly about: {topic_1}
                - {count_2} questions strictly about: {topic_2}
                
                You MUST return a JSON Array.
                Schema:
                [
                    {{
                        "question": "Question text...",
                        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                        "correct_answer": "Full text of correct option",
                        "reference": "Rule #",
                        "explanation": "Concise explanation based on the rule."
                    }}
                ]
                """
                try:
                    tool_config = [types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[STORE_ID]
                        )
                    )]
                    
                    response = st.session_state.client.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            system_instruction=SYSTEM_INSTRUCTION,
                            tools=tool_config,
                            response_mime_type="application/json",
                            safety_settings=SAFETY_SETTINGS # <--- SAFETY ADDED
                        )
                    )
                    data = extract_clean_json(response.text)
                    if data:
                        st.session_state.quiz_data = data
                        st.session_state.current_quiz_source = f"AI: {topic_1} ({count_1}) & {topic_2} ({count_2})"
                    else:
                        st.error("AI did not return valid JSON.")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # --- RENDER QUIZ ---
    if st.session_state.quiz_data:
        st.subheader(f"📝 {st.session_state.get('current_quiz_source', 'Quiz')}")
        
        with st.form("quiz_form"):
            for idx, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"**{idx+1}. {q['question']}**")
                opts = q.get('options', [])
                st.radio("Select:", opts, key=f"q_{idx}", label_visibility="collapsed", index=None)
                st.write("---")
            
            submitted = st.form_submit_button("Submit Answers")
            
        if submitted:
            score = 0
            total = len(st.session_state.quiz_data)
            
            for idx, q_item in enumerate(st.session_state.quiz_data):
                user_choice = st.session_state.get(f"q_{idx}")
                correct_choice = q_item.get('correct_answer')
                question_text = q_item.get('question', 'Question missing')
                
                if user_choice == correct_choice:
                    score += 1
                    st.success(f"Q{idx+1}: Correct!")
                    st.markdown(f"**Question:** {question_text}")
                    st.markdown(f"**Your Answer (Correct):** {user_choice}")
                else:
                    st.error(f"Q{idx+1}: Incorrect.")
                    st.markdown(f"**Question:** {question_text}")
                    st.markdown(f"**Your Answer:** {user_choice}")
                    st.markdown(f"**Correct Answer:** {correct_choice}")
                
                with st.expander(f"📖 Reference: {q_item.get('reference', 'N/A')}"):
                    st.info(q_item.get('explanation', 'No explanation provided.'))
            
            if total > 0:
                pct = (score / total) * 100
            else:
                pct = 0
            
            st.metric("Final Score", f"{score}/{total} ({pct:.0f}%)")