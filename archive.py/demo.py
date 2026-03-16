import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import base64
import json
import re
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError

# --- CONSTANTS & CONFIGURATION ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# API & Model Config
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU"

STORE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o" 
MODEL_ID = "gemini-2.5-flash" 

# UI Constants (Magic Strings)
MODE_CHAT = "Chat Assistant 🤖"
MODE_QUIZ = "Quiz Generator 📝"
SOURCE_CSV = "From Database (CSV)"
SOURCE_AI = "Generate New (AI + RAG)"

# --- SAFETY SETTINGS ---
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

# --- DATA MODELS (PYDANTIC) ---
class TopicDistribution(BaseModel):
    """Represents a single topic and its associated percentage weight."""
    topic_name: str = Field(..., description="Name of the topic/file to query in RAG.")
    percentage: int = Field(..., ge=0, le=100, description="Percentage of quiz devoted to this topic.")

class QuizRequest(BaseModel):
    """Validates configuration for AI+RAG Quiz Generator."""
    num_questions: int = Field(..., ge=1, le=25, description="Total questions (1-25).")
    topics: List[TopicDistribution] = Field(..., min_length=1, max_length=4, description="List of topics.")

    @field_validator('topics')
    @classmethod
    def validate_total_percentage(cls, topics: List[TopicDistribution], info: ValidationInfo) -> List[TopicDistribution]:
        active_topics = [t for t in topics if t.topic_name.strip()]
        if not active_topics:
             raise ValueError("At least one topic must be defined.")
        
        total = sum(t.percentage for t in active_topics)
        if total != 100:
            raise ValueError(f"Total percentage must equal 100%. Current: {total}%")
        return active_topics

# --- SERVICE LAYER ---
class QuizService:
    @staticmethod
    def calculate_distribution(request: QuizRequest) -> Dict[str, int]:
        """Calculates question count per topic handling rounding."""
        distribution = {}
        total_questions = request.num_questions
        topics = request.topics
        questions_assigned = 0
        
        for i, topic in enumerate(topics):
            if i == len(topics) - 1:
                count = total_questions - questions_assigned
            else:
                count = int(total_questions * (topic.percentage / 100))
            
            if count > 0:
                distribution[topic.topic_name] = count
                questions_assigned += count
        return distribution

# --- HELPER FUNCTIONS ---
def get_base64_of_bin_file(bin_file: str) -> Optional[str]:
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_background(png_file: str):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        st.markdown(f'''
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
        ''', unsafe_allow_html=True)

def extract_clean_json(text: str) -> Optional[Any]:
    """Robustly extracts JSON from LLM text responses."""
    if not text: return None
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()
    try:
        # Try finding array brackets
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx])
        
        # Try finding object braces
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return [json.loads(text[start_idx:end_idx])]
    except json.JSONDecodeError:
        return None
    return None

@st.cache_data
def get_quiz_topics(csv_file="ror_test.csv") -> List[str]:
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

def get_random_questions_from_csv(selected_tags: List[str], num_questions: int, csv_file="ror_test.csv") -> Optional[str]:
    if not os.path.exists(csv_file): return None
    try:
        df = pd.read_csv(csv_file)
        # Helper to check if row has matching tags
        def matches_tags(row_tags):
            if pd.isna(row_tags): return False
            row_tag_list = [t.strip() for t in str(row_tags).split(',')]
            return any(tag in row_tag_list for tag in selected_tags)
        
        filtered_df = df[df['Meta_Tags'].apply(matches_tags)]
        if filtered_df.empty: return None
        
        sample = filtered_df.sample(n=min(len(filtered_df), num_questions))
        return sample.to_json(orient="records")
    except Exception:
        return None

@st.cache_resource
def get_client():
    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        st.error("⚠️ API Key missing in configuration.")
        return None
    return genai.Client(api_key=API_KEY)

# --- PAGE SETUP ---
st.set_page_config(page_title="RAG Engine", page_icon="⚓", layout="wide")
set_background('background.png')

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am your Maritime Rules expert. Ask me about the Rules, or take a quiz!"}
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
    mode = st.radio("Select Tool:", [MODE_CHAT, MODE_QUIZ])
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
if mode == MODE_CHAT:
    st.title("📚 Regulations Chat")
    chat_container = st.container()
    
    # Display History
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input & Processing
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
                                file_search=types.FileSearch(file_search_store_names=[STORE_ID])
                            )]
                            
                            response = st.session_state.client.models.generate_content(
                                model=MODEL_ID,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=0.1, 
                                    system_instruction=SYSTEM_INSTRUCTION, 
                                    tools=tool_config,
                                    safety_settings=SAFETY_SETTINGS
                                )
                            )
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})

                            # Check grounding
                            is_grounded = any(c.grounding_metadata for c in response.candidates) if response.candidates else False
                            if is_grounded:
                                st.caption("🔍 *Sourced from Knowledge Base*")
                            else:
                                st.caption("⚠️ *General knowledge used (No direct citation found)*")
                                
                        except Exception as e:
                            st.error(f"API Error: {str(e)}")

# ==========================================
# MODE 2: QUIZ GENERATOR
# ==========================================
elif mode == MODE_QUIZ:
    st.title("📝 Knowledge Check")
    
    # --- CALLBACK TO RESET STATE ---
    def clear_quiz_state():
        st.session_state.quiz_data = None
        st.session_state.current_quiz_source = None

    # Clean UI State Switch with on_change
    source_type = st.radio(
        "Source:", 
        [SOURCE_CSV, SOURCE_AI], 
        horizontal=True,
        on_change=clear_quiz_state 
    )
    st.divider()

    # --- CSV PATH ---
    if source_type == SOURCE_CSV:
        available_topics = get_quiz_topics()
        c1, c2 = st.columns(2)
        with c1:
            selected_tags = st.multiselect("Topics:", options=available_topics)
        with c2:
            num_questions = st.number_input("Count:", 1, 25, 5)
        
        if st.button("Load CSV Quiz", type="primary"):
            if not selected_tags:
                st.warning("Please select at least one topic.")
            else:
                with st.spinner("Fetching data..."):
                    raw_data = get_random_questions_from_csv(selected_tags, num_questions)
                    if raw_data:
                        fmt_prompt = f"""
                        Convert this CSV data into a JSON quiz.
                        Add "explanation" field based on Maritime Rules.
                        Raw Data: {raw_data} 
                        Schema: [{{question, options[], correct_answer, reference, explanation}}]
                        """
                        try:
                            # Note: CSV formatting tasks usually handle JSON mode fine without tools
                            resp = st.session_state.client.models.generate_content(
                                model=MODEL_ID,
                                contents=fmt_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    system_instruction=SYSTEM_INSTRUCTION,
                                    safety_settings=SAFETY_SETTINGS
                                )
                            )
                            data = extract_clean_json(resp.text)
                            if data:
                                st.session_state.quiz_data = data
                                st.session_state.current_quiz_source = f"CSV: {', '.join(selected_tags)}"
                            else:
                                st.error("Failed to format CSV data.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("No matching questions found in CSV.")

    # --- AI GENERATION PATH ---
    else: 
        st.subheader("Distribution Configuration")
        st.caption("Total percentage must equal 100%.")

        topics_input = []
        # Input Matrix
        for i in range(1, 5):
            c1, c2 = st.columns([3, 1])
            with c1:
                t_name = st.text_input(f"Topic {i}", key=f"t_{i}", placeholder=f"e.g. Rule {i+5}")
            with c2:
                t_pct = st.number_input(f"%", 0, 100, 0, key=f"p_{i}")
            if t_name.strip():
                topics_input.append({"topic_name": t_name, "percentage": t_pct})

        num_q = st.number_input("Total Questions", 1, 25, 5)
        
        # Validation Feedback
        total_pct = sum(t['percentage'] for t in topics_input)
        if total_pct == 100:
            st.success(f"Total: {total_pct}% (Valid)")
        else:
            st.warning(f"Total: {total_pct}% (Target: 100%)")

        if st.button("Generate RAG Quiz", type="primary"):
            try:
                # 1. Logic Validation
                req = QuizRequest(num_questions=num_q, topics=topics_input)
                dist_map = QuizService.calculate_distribution(req)
                
                # 2. UI Feedback
                dist_msg = " | ".join([f"{k}: {v}" for k,v in dist_map.items()])
                st.info(f"Generating: {dist_msg}")

                # 3. Prompt Construction
                prompt_dist = "\n".join([f"- {c} questions strictly about: {t}" for t, c in dist_map.items()])
                prompt = f"""
                Create a {num_q} question multiple-choice quiz based ONLY on the provided File Store documents.
                DISTRIBUTION:
                {prompt_dist}
                
                OUTPUT FORMATTING:
                Return ONLY a raw JSON Array. No markdown formatting.
                Schema:
                [
                    {{
                        "question": "Question text...",
                        "options": ["A) ...", "B) ..."],
                        "correct_answer": "Full text of correct option",
                        "reference": "Rule #",
                        "explanation": "Concise explanation."
                    }}
                ]
                """

                # 4. Execution
                with st.spinner("Drafting questions..."):
                    tool_config = [types.Tool(
                        file_search=types.FileSearch(file_search_store_names=[STORE_ID])
                    )]
                    # NO json_mode here to support Tools
                    resp = st.session_state.client.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            system_instruction=SYSTEM_INSTRUCTION,
                            tools=tool_config,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                    
                    data = extract_clean_json(resp.text)
                    if data:
                        st.session_state.quiz_data = data
                        st.session_state.current_quiz_source = f"AI Generated: {dist_msg}"
                    else:
                        st.error("AI response invalid.")
                        if st.session_state.debug_mode: st.code(resp.text)

            except ValidationError as ve:
                for err in ve.errors():
                    st.error(f"Input Error: {err.get('msg', 'Invalid Data').replace('Value error, ', '')}")
            except Exception as e:
                st.error(f"System Error: {e}")

    # --- RENDER QUIZ & ADMIN ---
    if st.session_state.quiz_data:
        st.subheader(f"📝 {st.session_state.get('current_quiz_source', 'Quiz')}")
        
        with st.form("quiz_form"):
            score = 0
            for idx, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"**{idx+1}. {q.get('question', 'Error')}**")
                st.radio("Select:", q.get('options', []), key=f"q_{idx}", label_visibility="collapsed", index=None)
                st.write("---")
            
            # --- RESTORED FEEDBACK LOGIC ---
            if st.form_submit_button("Submit Answers"):
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
                
                pct = (score/total)*100 if total else 0
                st.metric("Score", f"{score}/{total} ({pct:.0f}%)")

        st.divider()
        with st.expander("🔐 Admin Controls (Export Quiz)"):
            try:
                json_str = json.dumps(st.session_state.quiz_data, indent=2)
                st.download_button(
                    "Download JSON", 
                    json_str, 
                    "quiz_export.json", 
                    "application/json"
                )
            except Exception as e:
                st.error(f"Export Error: {e}")