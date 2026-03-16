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

# --- UPDATED SYSTEM INSTRUCTIONS (v7.1) ---
# Enhanced to prevent Recitation Check errors by emphasizing paraphrasing
SYSTEM_INSTRUCTION = """
You are a Maritime Rules of the Road expert and Examiner.
CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base.

RULES FOR OPERATION:
1. Search across ALL these documents for every question.
2. Do NOT answer from general knowledge. 
3. Always cite specific Rule numbers (e.g., "Rule 14(a)") or Annex sections.
4. ENSURE ACCURACY by using the exact terminology found in the documents. 
5. When explaining a Rule, PARAPHRASE the content in your own words rather than copying verbatim.
6. Use brief excerpts (1-2 sentences maximum) when directly referencing rule text.
7. DO NOT reproduce large contiguous blocks of text. Instead, summarize and explain concepts.
8. Focus on explaining the meaning and application of rules, not reproducing the exact wording.
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
                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }
            if tools: config_args["tools"] = tools
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            if response.text:
                return response.text
            
            # Defensive check for empty text but valid candidates
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    parts_text = [p.text for p in candidate.content.parts if p.text]
                    if parts_text:
                        return "".join(parts_text)
                
                # Check for specific stop reasons
                if candidate.finish_reason:
                    reason = str(candidate.finish_reason)
                    if "RECITATION" in reason:
                        raise ValueError("The AI attempted to copy too much text verbatim. (Recitation Check).")
                    raise ValueError(f"AI Generation stopped. Reason: {reason}")

            raise ValueError("The AI returned an empty response (No candidates or text found).")

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
        # UPDATED PROMPT: Emphasize conciseness to avoid Recitation error
        prompt = f"""
        Create a {request.num_questions} question multiple-choice quiz based ONLY on the provided File Store.
        DISTRIBUTION: {dist_str}
        
        OUTPUT SCHEMA (JSON Array): 
        [
            {{
                "question": "Question text...", 
                "options": ["A) ...", "B) ..."], 
                "correct_answer": "Full text of option", 
                "reference": "Rule #", 
                "explanation": "Concise explanation referencing the specific rule text."
            }}
        ]
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

class RemediationService:
    """Generates lesson plans based on incorrect quiz answers."""
    
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def generate_lesson(self, incorrect_questions: List[QuizQuestion]) -> str:
        unique_refs = sorted(list(set(q.reference for q in incorrect_questions if q.reference)))
        
        if not unique_refs:
            topics_context = "General Maritime Rules"
        else:
            topics_context = ", ".join(unique_refs)

        # UPDATED PROMPT: Avoid verbatim copying - use paraphrased summaries instead
        prompt = f"""
        The student failed a quiz on the following specific Maritime Rules/Topics: {topics_context}.
        
        TASK:
        Create a targeted Remediation Lesson Plan based ONLY on the provided File Store documents.
        
        CRITICAL INSTRUCTIONS:
        - DO NOT copy large blocks of text verbatim from the documents
        - Paraphrase and summarize the rule content in your own words
        - Use brief excerpts (1-2 sentences max) when referencing specific rule text
        - Focus on explaining concepts rather than reproducing text
        
        STRUCTURE:
        1. **Executive Summary**: Briefly list the rules the student needs to review.
        2. **Deep Dive**: For each topic ({topics_context}):
           - **Rule Summary**: Provide a concise summary of what the rule states (paraphrased, not verbatim).
           - **Simplified Explanation**: Explain it in plain English with examples.
           - **Common Pitfalls**: Why do students usually miss this?
           - **Memory Aid**: Provide a mnemonic or trick to remember it.
        3. **Study Plan**: Suggest specific next steps.
        
        FORMAT: Markdown.
        """
        
        return self.ai.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True)

# --- PRESENTATION LAYER ---

def init_session_state():
    defaults = {
        "messages": [{"role": "assistant", "content": "👋 Maritime Expert Ready."}],
        "quiz_data": None,
        "current_quiz_source": None,
        "incorrect_questions": [],
        "remediation_text": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

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
        remediation_service = RemediationService(ai_provider)
    except KeyError:
        st.error("Missing Secrets"); st.stop()

    # 2. Sidebar
    with st.sidebar:
        st.title("⚓ Maritime AI")
        mode = st.radio("Tool:", ["Chat 🤖", "Quiz 📝"])
        if st.button("Reset Session"):
            st.session_state.messages = []
            st.session_state.quiz_data = None
            st.session_state.incorrect_questions = []
            st.session_state.remediation_text = None
            st.rerun()
        st.divider()
        st.caption("✅ v7.1 - Enhanced Recitation Prevention") 

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
        def clear(): 
            st.session_state.quiz_data = None
            st.session_state.incorrect_questions = []
            st.session_state.remediation_text = None

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
            for i in range(1, 5):
                c1, c2 = st.columns([3, 1])
                t = c1.text_input(f"Topic {i}")
                p = c2.number_input(f"%", 0, 100, 0, key=f"p{i}")
                if t: topics.append(TopicDistribution(topic_name=t, percentage=p))
            
            n = st.number_input("Count", 1, 25, 5)
            
            current_total = sum(t.percentage for t in topics)
            st.progress(min(current_total, 100) / 100)
            if current_total == 100:
                st.success(f"Total: {current_total}% (Valid)")
            else:
                st.warning(f"Total: {current_total}% (Target: 100%)")

            if st.button("Generate AI Quiz"):
                st.info("Generating Quiz... Please wait.")
                try:
                    req = QuizRequest(num_questions=n, topics=topics)
                    with st.spinner("Processing Documents..."):
                        st.session_state.quiz_data = quiz_service.generate_from_ai(req)
                        st.session_state.current_quiz_source = "AI"
                        st.rerun()
                except Exception as e: st.error(f"Error: {e}")

        # EXECUTION
        if st.session_state.quiz_data:
            st.divider()
            st.subheader(f"Quiz: {st.session_state.current_quiz_source}")
            questions = st.session_state.quiz_data
            
            with st.form("quiz"):
                score = 0
                answers = {}
                for i, q in enumerate(questions):
                    st.markdown(f"**{i+1}. {q.question}**")
                    answers[i] = st.radio("Opt", q.options, key=f"q{i}", label_visibility="collapsed", index=None)
                    st.write("---")
                
                # --- SUBMIT LOGIC ---
                if st.form_submit_button("Submit"):
                    # 1. Reset state for new run
                    current_incorrect = []
                    
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
                            # Track missed question
                            current_incorrect.append(q)
                            
                        with st.expander("Explanation"): st.info(f"{q.explanation} ({q.reference})")
                    
                    # 2. Update Persistent State
                    st.session_state.incorrect_questions = current_incorrect
                    st.session_state.remediation_text = None # Reset old lesson
                    
                    pct = (score/len(questions))*100
                    st.metric("Score", f"{score}/{len(questions)} ({pct:.0f}%)")
            
            # --- REMEDIATION ZONE (Outside Form) ---
            if st.session_state.incorrect_questions:
                st.divider()
                st.subheader("🎓 Remediation Zone")
                st.warning(f"You missed {len(st.session_state.incorrect_questions)} questions. Let's create a personalized lesson plan.")
                
                if st.button("Generate Remediation Lesson"):
                    with st.spinner("Analyzing weak points and generating lesson..."):
                        lesson = remediation_service.generate_lesson(st.session_state.incorrect_questions)
                        st.session_state.remediation_text = lesson
                
                if st.session_state.remediation_text:
                    st.markdown("---")
                    st.markdown(st.session_state.remediation_text)
            
            # --- ADMIN EXPORT ---
            with st.expander("Admin Export"):
                st.download_button("Download JSON", json.dumps([q.model_dump() for q in questions], indent=2), "quiz.json")

if __name__ == "__main__":
    main()