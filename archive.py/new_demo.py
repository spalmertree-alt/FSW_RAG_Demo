import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import json
import base64
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MaritimePOC")

# --- CONSTANTS ---
API_KEY_NAME = "GEMINI_API_KEY"
STORE_ID_NAME = "STORE_ID"
MODEL_ID = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
You are a Maritime Rules of the Road expert and Examiner.
CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base.
RULES:
1. Search across ALL these documents for every question.
2. Do NOT answer from general knowledge. 
3. Always cite specific Rule numbers (e.g., "Rule 14(a)") or Annex sections.
4. If information is not found, state clearly that it is not available.
"""

# --- DOMAIN LAYER (Models) ---

class QuizQuestion(BaseModel):
    """Represents a standardized quiz question."""
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer: str = Field(..., description="The text of the correct answer")
    reference: str = Field(..., description="Rule citation")
    explanation: str = Field(..., description="Reasoning for the answer")

class TopicDistribution(BaseModel):
    """Configuration for AI generation topics."""
    topic_name: str
    percentage: int

class QuizRequest(BaseModel):
    """Validates configuration for Quiz Generation."""
    num_questions: int = Field(..., ge=1, le=25)
    topics: List[TopicDistribution] = Field(default_factory=list)

    @field_validator('topics')
    @classmethod
    def validate_total_percentage(cls, topics: List[TopicDistribution], info: ValidationInfo) -> List[TopicDistribution]:
        if not topics:
            return []
        total = sum(t.percentage for t in topics)
        if total != 100:
            raise ValueError(f"Total percentage must equal 100%. Current: {total}%")
        return topics

# --- INFRASTRUCTURE LAYER (External IO) ---

class GenAIProvider:
    """Encapsulates Google GenAI interactions."""
    
    def __init__(self, api_key: str, store_id: str):
        self.client = genai.Client(api_key=api_key)
        self.store_id = store_id
        self.model = MODEL_ID

    def generate_content(
        self, 
        prompt: str, 
        system_instr: str, 
        use_rag: bool = True,
        response_schema: Any = None
    ) -> str:
        try:
            tools = []
            if use_rag and self.store_id:
                tools = [types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[self.store_id])
                )]

            config_args = {
                "temperature": 0.1,
                "system_instruction": system_instr,
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                ]
            }
            if tools: config_args["tools"] = tools
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].content.parts:
                    parts_text = [p.text for p in response.candidates[0].content.parts if p.text]
                    if parts_text: return "".join(parts_text)
                raise ValueError("The AI returned an empty response.")
            
            return response.text

        except Exception as e:
            logger.error(f"GenAI generation failed: {e}")
            raise RuntimeError(str(e))

class DataManager:
    """Handles File I/O operations."""
    
    @staticmethod
    def load_csv_sample(filepath: str, selected_tags: List[str], sample_size: int) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath): return []
        try:
            df = pd.read_csv(filepath)
            if "Meta_Tags" not in df.columns: return []
            
            def matches_tags(row_tags):
                if pd.isna(row_tags): return False
                row_tag_list = [t.strip() for t in str(row_tags).split(',')]
                return any(tag in row_tag_list for tag in selected_tags)

            filtered_df = df[df['Meta_Tags'].apply(matches_tags)]
            if filtered_df.empty: return []
            
            sample = filtered_df.sample(n=min(len(filtered_df), sample_size))
            return sample.to_dict(orient="records")
        except Exception: return []

    @staticmethod
    def get_topics_from_csv(filepath: str) -> List[str]:
        if not os.path.exists(filepath): return ["General Rules"]
        try:
            df = pd.read_csv(filepath)
            if "Meta_Tags" in df.columns:
                all_tags = set()
                for tag_string in df["Meta_Tags"].dropna():
                    tags = [tag.strip() for tag in str(tag_string).split(',')]
                    all_tags.update(tags)
                return sorted(list(all_tags))
        except Exception: return ["General Rules"]
        return ["General Rules"]

    @staticmethod
    def get_base64_image(filepath: str) -> Optional[str]:
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f: data = f.read()
            return base64.b64encode(data).decode()
        except Exception: return None

# --- SERVICE LAYER (Business Logic) ---

class JSONParser:
    @staticmethod
    def parse(text: Optional[str]) -> List[Dict[str, Any]]:
        if not text: raise ValueError("AI response empty.")
        clean_text = re.sub(r'```json\s*', '', text)
        clean_text = re.sub(r'```', '', clean_text).strip()
        
        start = clean_text.find('[')
        end = clean_text.rfind(']') + 1
        
        if start != -1 and end != 0:
            return json.loads(clean_text[start:end])
        raise ValueError("No JSON array found.")

class QuizService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def calculate_topic_distribution(self, request: QuizRequest) -> Dict[str, int]:
        distribution = {}
        total = request.num_questions
        assigned = 0
        active = [t for t in request.topics if t.percentage > 0]
        
        for i, topic in enumerate(active):
            if i == len(active) - 1: count = total - assigned
            else: count = int(total * (topic.percentage / 100))
            if count > 0:
                distribution[topic.topic_name] = count
                assigned += count
        return distribution

    def generate_from_ai(self, request: QuizRequest) -> List[QuizQuestion]:
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} questions about: {t}" for t, c in dist.items()])
        prompt = f"""
        Create a {request.num_questions} question multiple-choice quiz based ONLY on the provided File Store.
        DISTRIBUTION: {dist_str}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        raw = self.ai.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True)
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

    def generate_from_csv(self, raw_data: List[Dict], count: int) -> List[QuizQuestion]:
        prompt = f"""
        Convert this CSV data to JSON quiz. Use RAG for explanation/reference if missing.
        Data: {json.dumps(raw_data)}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        raw = self.ai.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True)
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

# --- PRESENTATION LAYER ---

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Maritime Expert Ready."}]
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "current_quiz_source" not in st.session_state:
        st.session_state.current_quiz_source = None

def set_background():
    bin_str = DataManager.get_base64_image('background.png')
    if bin_str:
        st.markdown(f'''<style>
        .stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; }}
        </style>''', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="RAG Engine", page_icon="⚓", layout="wide")
    set_background()
    init_session_state()
    
    # 1. Dependency Injection
    try:
        ai_provider = GenAIProvider(st.secrets[API_KEY_NAME], st.secrets[STORE_ID_NAME])
        quiz_service = QuizService(ai_provider)
    except KeyError:
        st.error("Missing Secrets"); st.stop()

    # 2. Sidebar
    with st.sidebar:
        st.title("⚓ Maritime AI")
        mode = st.radio("Tool:", ["Chat 🤖", "Quiz 📝"])
        if st.button("Reset Session"):
            st.session_state.messages = []
            st.session_state.quiz_data = None
            st.rerun()
        st.divider()
        st.caption("✅ v3.0 - Features Active") # CHECK FOR THIS

    # 3. Routing
    if mode == "Chat 🤖":
        st.title("Regulations Chat")
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": p})
            st.chat_message("user").write(p)
            with st.spinner("Thinking..."):
                r = ai_provider.generate_content(p, SYSTEM_INSTRUCTION)
                st.session_state.messages.append({"role": "assistant", "content": r})
                st.chat_message("assistant").write(r)

    elif mode == "Quiz 📝":
        st.title("Knowledge Check")
        
        # CLEAR STATE ON CHANGE
        def clear(): st.session_state.quiz_data = None
        source = st.radio("Source:", ["CSV", "AI Gen"], horizontal=True, on_change=clear)
        
        # CONFIGURATION
        if source == "CSV":
            tags = st.multiselect("Topics:", DataManager.get_topics_from_csv("ror_test.csv"))
            n = st.number_input("Count:", 1, 25, 5)
            if st.button("Load CSV") and tags:
                data = DataManager.load_csv_sample("ror_test.csv", tags, n)
                if data:
                    st.session_state.quiz_data = quiz_service.generate_from_csv(data, n)
                    st.session_state.current_quiz_source = "CSV"
                    st.rerun()
        else:
            topics = []
            for i in range(1, 4):
                c1, c2 = st.columns([3, 1])
                t = c1.text_input(f"Topic {i}")
                p = c2.number_input(f"%", 0, 100, 0, key=f"p{i}")
                if t: topics.append(TopicDistribution(topic_name=t, percentage=p))
            n = st.number_input("Count", 1, 25, 5)
            if st.button("Generate AI Quiz"):
                try:
                    req = QuizRequest(num_questions=n, topics=topics)
                    st.session_state.quiz_data = quiz_service.generate_from_ai(req)
                    st.session_state.current_quiz_source = "AI"
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

        # EXECUTION (The missing features should be here)
        if st.session_state.quiz_data:
            st.divider()
            st.subheader(f"Quiz: {st.session_state.current_quiz_source}")
            questions = st.session_state.quiz_data
            
            with st.form("quiz"):
                score = 0
                answers = {}
                for i, q in enumerate(questions):
                    st.markdown(f"**{i+1}. {q.question}**") # Feature: Q Text
                    answers[i] = st.radio("Opt", q.options, key=f"q{i}", label_visibility="collapsed")
                    st.write("---")
                
                if st.form_submit_button("Submit"):
                    for i, q in enumerate(questions):
                        correct = q.correct_answer
                        user = answers.get(i)
                        if user == correct:
                            score += 1
                            st.success(f"Q{i+1} Correct!")
                            st.write(f"**Question:** {q.question}")
                            st.write(f"**Your Answer:** {user}")
                        else:
                            st.error(f"Q{i+1} Incorrect")
                            st.write(f"**Question:** {q.question}")
                            st.write(f"**Your Answer:** {user}")
                            st.write(f"**Correct:** {correct}")
                        with st.expander("Explanation"): st.info(f"{q.explanation} ({q.reference})")
                    
                    pct = (score/len(questions))*100
                    st.metric("Score", f"{score}/{len(questions)} ({pct:.0f}%)") # Feature: %
            
            # Feature: JSON Export
            with st.expander("Admin Export"):
                st.download_button("Download JSON", json.dumps([q.model_dump() for q in questions], indent=2), "quiz.json")

if __name__ == "__main__":
    main()