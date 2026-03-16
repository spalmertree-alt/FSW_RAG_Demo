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
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# --- HELPER: CLEAN JSON ---
def clean_json_text(text):
    """Cleans the response text to ensure it is valid JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()

# --- HELPER: LOAD & FILTER CSV (Hybrid Logic) ---
@st.cache_data
def get_quiz_topics():
    """Parses individual tags from Meta_Tags column (comma-separated values) and returns unique, alphabetized tags."""
    csv_file = "ror_test.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if "Meta_Tags" in df.columns:
                # Parse all tags from comma-separated values
                all_tags = set()
                for tag_string in df["Meta_Tags"].dropna():
                    # Split by comma and clean each tag
                    tags = [tag.strip() for tag in str(tag_string).split(',')]
                    all_tags.update(tags)
                # Return alphabetized list of unique tags
                return sorted(list(all_tags))
        except Exception as e:
            st.error(f"Error parsing tags: {e}")
            pass
    return ["General Rules"]

def get_random_questions_from_csv(selected_tags, num_questions):
    """
    Filters questions where Meta_Tags contains any of the selected tags.
    Meta_Tags column contains comma-separated tags like "#Inland, #Rule9, #Crossing"
    """
    csv_file = "ror_test.csv"
    if not os.path.exists(csv_file):
        return None
    
    try:
        df = pd.read_csv(csv_file)
        
        # Filter rows where Meta_Tags contains any of the selected tags
        # Check if any selected tag appears in the Meta_Tags string
        def contains_any_tag(tag_string, selected_tags):
            if pd.isna(tag_string):
                return False
            tag_string = str(tag_string)
            # Split the tag string and check if any selected tag matches
            tags_in_row = [tag.strip() for tag in tag_string.split(',')]
            return any(selected_tag in tags_in_row for selected_tag in selected_tags)
        
        filtered_df = df[df['Meta_Tags'].apply(lambda x: contains_any_tag(x, selected_tags))]
        
        if filtered_df.empty:
            return None
        
        # Show breakdown by selected tags (count how many questions match each tag)
        if len(selected_tags) > 1:
            tag_counts = {}
            for tag in selected_tags:
                count = df[df['Meta_Tags'].apply(lambda x: contains_any_tag(x, [tag]))].shape[0]
                tag_counts[tag] = count
            breakdown = ", ".join([f"{tag}: {count}" for tag, count in tag_counts.items()])
            st.info(f"📊 Questions matching tags: {breakdown} (Total matching: {len(filtered_df)})")
        else:
            st.info(f"📊 Found {len(filtered_df)} question(s) matching tag: {selected_tags[0]}")
            
        # Randomly sample the requested number of questions
        # If we have fewer questions than requested, take all of them
        sample_size = min(len(filtered_df), num_questions)
        sampled_df = filtered_df.sample(n=sample_size, random_state=None)
        
        # Convert to a string format/dictionary to send to Gemini
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
if "client" not in st.session_state:
    client = get_client()
    if client:
        st.session_state.client = client
    else:
        st.session_state.client = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    mode = st.radio("Select Mode:", ["RAG Engine 🤖", "Quiz Generator 📝"])
    
    st.divider()
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode)

# ==========================================
# MODE 1: AI TUTOR (Standard RAG)
# ==========================================
if mode == "RAG Engine 🤖":
    st.title("📚 RAG Engine")
    st.caption("Ask questions about the Rules of the Road documents.")

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
                        system_instruction = (
                            "You are a Maritime Rules of the Road expert. "
                            "CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base. "
                            "The knowledge base contains the following documents: "
                            "1. Rules_of_the_Road.pdf (Official Regulation Textbook Source) "
                            "2. ROTR_Slick_Sheet.pdf (Quick Reference Guide) "
                            "3. Navigation_Rules_Standard_Size.pdf (Standard Navigation Rules) "
                            "4. ROTR_Guide.pdf (Learner Guide Book) "
                            "5. ror_test.csv (Question and Answer Database) "
                            "Search across ALL these documents for every question. "
                            "Do NOT answer from general knowledge. Always cite specific Rule numbers, Annex sections, or document references. "
                            "If information is not found in any of these documents, state clearly that it is not available in the provided documents."
                        )
                        full_prompt = f"{system_instruction}\n\nQuestion: {prompt}"
                        
                        response = st.session_state.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                tools=[
                                    types.Tool(
                                        file_search=types.FileSearch(
                                            file_search_store_names=[STORE_ID]
                                        )
                                    )
                                ]
                            )
                        )
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ==========================================
# MODE 2: QUIZ GENERATOR (Hybrid: Pandas + Gemini)
# ==========================================
elif mode == "Quiz Generator 📝":
    st.title("📝 Knowledge Check")
    
    available_topics = get_quiz_topics()

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        selected_tags = st.multiselect(
            "Select Tags (one or multiple):", 
            options=available_topics,
            default=None,
            help="Select individual tags (e.g., #Rule9, #Crossing). Questions matching any selected tag will be included. Tags are alphabetized."
        )
        # Show count of selected tags
        if selected_tags:
            st.caption(f"Selected: {len(selected_tags)} tag(s): {', '.join(selected_tags)}")
    with col2:
        num_questions = st.number_input("Number of Questions:", min_value=1, max_value=25, value=5)

    if st.button("Generate Quiz", type="primary"):
        if not selected_tags:
            st.error("Please select at least one tag.")
        else:
            tag_label = f"{len(selected_tags)} tag(s)" if len(selected_tags) > 1 else selected_tags[0]
            with st.spinner(f"Generating quiz from {tag_label}..."):
                # STEP 1: Use Pandas to get the raw data (Deterministic & Accurate)
                raw_data_json = get_random_questions_from_csv(selected_tags, num_questions)
                
                if not raw_data_json or raw_data_json == "[]":
                    st.error("No questions found in CSV for these tags. Please check your CSV 'Meta_Tags' column.")
                else:
                    # STEP 2: Use Gemini to Format the data (No File Search needed here, just formatting)
                    try:
                        formatting_prompt = f"""
                        I have some raw question data from a CSV. 
                        Please format this data into a clean JSON array for my quiz app.
                        
                        RAW DATA:
                        {raw_data_json}

                        INSTRUCTIONS:
                        1. Create a JSON array.
                        2. Map the CSV columns to this schema:
                           - "question": The question text
                           - "options": An array of the choices [A, B, C, D]
                           - "correct_answer": The text of the correct answer
                           - "reference": The Rule reference
                        3. RETURN ONLY JSON. NO MARKDOWN.
                        """
                        
                        response = st.session_state.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=formatting_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.1 # Low temp for strict formatting
                            )
                        )
                        
                        cleaned_text = clean_json_text(response.text)
                        st.session_state.quiz_data = json.loads(cleaned_text)
                        st.session_state.user_answers = {}
                        st.session_state.last_selected_tags = selected_tags  # Store for display 
                        
                    except Exception as e:
                        st.error(f"Error formatting quiz: {e}")
                        if 'response' in locals():
                            st.caption("Raw AI Response:")
                            st.code(response.text)

    # Render Quiz Form
    if st.session_state.quiz_data:
        st.divider()
        # Display selected tags from session state
        current_tags = st.session_state.get('last_selected_tags', [])
        st.subheader(f"Quiz: {', '.join(current_tags) if current_tags else 'Custom Tags'}")
        
        with st.form("quiz_form"):
            for idx, q_item in enumerate(st.session_state.quiz_data):
                st.markdown(f"**{idx+1}. {q_item['question']}**")
                st.radio(
                    "Select Answer:", 
                    q_item['options'], 
                    key=f"q_{idx}", 
                    label_visibility="collapsed",
                    index=None
                )
                st.write("---")
            
            submit_quiz = st.form_submit_button("Submit & Grade")

        # Grading (Uses RAG for Explanations)
        if submit_quiz:
            st.subheader("📊 Results")
            score = 0
            total = len(st.session_state.quiz_data)
            
            for idx, q_item in enumerate(st.session_state.quiz_data):
                user_choice = st.session_state.get(f"q_{idx}")
                correct_choice = q_item['correct_answer']
                question_text = q_item['question']
                
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
                    
                    # RAG EXPLANATION CALL
                    with st.expander(f"📖 Explanation for Q{idx+1}"):
                        with st.spinner("Consulting Manuals..."):
                            explanation_prompt = f"""
                            The user missed this question.
                            Question: {question_text}
                            Correct Answer: {correct_choice}
                            Rule Ref: {q_item.get('reference', '')}
                            
                            Search the PDF manuals. Explain WHY this is the correct answer based on the official text.
                            """
                            # We use the CLIENT with TOOLS here because we need RAG for the explanation
                            expl_response = st.session_state.client.models.generate_content(
                                model="gemini-2.5-flash",
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

            # Calculate percentage
            percentage = (score / total * 100) if total > 0 else 0
            st.metric(label="Final Score", value=f"{score} / {total} ({percentage:.0f}%)")