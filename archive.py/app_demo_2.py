import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import json
import base64
import re
import logging
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Iterator
from io import BytesIO
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- THIRD PARTY IMPORTS ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False
    st_canvas = None

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
TEMPERATURE = 0.1
REMEDIATION_BATCH_SIZE = 3
MAX_QUIZ_QUESTIONS = 25
MIN_QUIZ_QUESTIONS = 1
DEFAULT_QUIZ_COUNT = 5
CACHE_TTL_SECONDS = 3600
REMEDIATION_QUIZ_MIN = 5
REMEDIATION_QUIZ_MAX = 25
MAX_PARALLEL_WORKERS = 3
API_TIMEOUT_SECONDS = 60
BACKGROUND_IMAGE_CACHE_TTL = 86400
MIN_AUDIO_SIZE_BYTES = 100
WAV_HEADER_SIZE = 12
AUDIO_TEMPERATURE = 0.2
RESPONSE_CACHE_TTL = 1800
ENABLE_STREAMING = True

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

try:
    import streamlit as st
    IN_STREAMLIT = True
except ImportError:
    IN_STREAMLIT = False

SYSTEM_INSTRUCTION = """
You are a Maritime Rules of the Road expert and Examiner.
CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base.

RULES FOR OPERATION:
1. Search across ALL these documents for every question.
2. Do NOT answer from general knowledge. 
3. Always cite specific Rule numbers (e.g., "Rule 14(a)") or Annex sections.
4. Ensure accuracy by using the exact terminology found in the documents.
5. When quoting rule text verbatim, use brief excerpts (1-3 sentences maximum per rule).
6. Intersperse verbatim quotes with your own explanations and analysis.
7. DO NOT reproduce entire paragraphs or pages verbatim - select only the most relevant sentences.
"""

# --- DOMAIN LAYER (Models) ---

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The question text")
    options: List[str] = Field(..., description="List of possible answers")
    correct_answer: str = Field(..., description="The text of the correct answer")
    reference: str = Field(..., description="Rule citation")
    explanation: str = Field(..., description="Reasoning for the answer")

class TopicDistribution(BaseModel):
    topic_name: str
    percentage: int

class QuizRequest(BaseModel):
    num_questions: int = Field(..., ge=1, le=25)
    topics: List[TopicDistribution] = Field(default_factory=list)

    @field_validator('topics')
    @classmethod
    def validate_total_percentage(cls, topics: List[TopicDistribution], info: ValidationInfo) -> List[TopicDistribution]:
        if not topics: return []
        total = sum(t.percentage for t in topics)
        if total != 100:
            raise ValueError(f"Total percentage must equal 100%. Current: {total}%")
        return topics

class FreeResponseQuestion(BaseModel):
    id: str
    topic: str
    question_text: str
    ideal_answer: str

class VisualQuestion(BaseModel):
    """Represents a visual identification question."""
    id: str
    question_text: str
    image_path: str  # Path to the base image presented to student
    rubric_path: str # Path to the image showing the correct answer

class GradingResult(BaseModel):
    score: int
    status: str
    reasoning: str

# --- INFRASTRUCTURE LAYER (External IO) ---

class PDFGenerator:
    _BOLD_PATTERN = re.compile(r'\*\*(.*?)\*\*')
    _NUMBERED_PATTERN = re.compile(r'^\d+\.')
    
    @staticmethod
    def create_study_guide(content: str) -> Optional[BytesIO]:
        if not HAS_REPORTLAB: return None
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        story = []
        styles.add(ParagraphStyle(name='RuleTitle', parent=styles['Heading2'], spaceAfter=6, textColor=colors.darkblue))
        styles.add(ParagraphStyle(name='NormalJustified', parent=styles['Normal'], alignment=TA_JUSTIFY, spaceAfter=6))
        story.append(Paragraph("Maritime Remediation Study Guide", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Generated by Maritime AI PoC", styles['Normal']))
        story.append(Spacer(1, 24))
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            formatted_line = PDFGenerator._BOLD_PATTERN.sub(r'<b>\1</b>', line)
            if line.startswith('#'):
                clean_text = line.lstrip('#').strip().replace('**', '')
                story.append(Paragraph(clean_text, styles['RuleTitle']))
                story.append(Spacer(1, 6))
            elif line.startswith(('- ', '* ')):
                clean_text = formatted_line[2:].strip()
                story.append(Paragraph(f"• {clean_text}", styles['NormalJustified']))
            elif PDFGenerator._NUMBERED_PATTERN.match(line):
                story.append(Paragraph(formatted_line, styles['NormalJustified']))
            else:
                story.append(Paragraph(formatted_line, styles['NormalJustified']))
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f"PDF Generation failed: {e}")
            return None

class GenAIProvider:
    def __init__(self, api_key: str, store_id: str):
        self.client = genai.Client(api_key=api_key)
        self.store_id = store_id
        self.model = MODEL_ID
        self._response_cache: Dict[str, tuple[str, float]] = {}
    
    def __del__(self):
        try:
            if hasattr(self, 'client'): self.client.close()
        except: pass

    def _get_config(self, system_instr: str, temperature: float = TEMPERATURE) -> types.GenerateContentConfig:
        tools = []
        if self.store_id:
            tools = [types.Tool(file_search=types.FileSearch(file_search_store_names=[self.store_id]))]
        return types.GenerateContentConfig(temperature=temperature, system_instruction=system_instr, tools=tools, safety_settings=SAFETY_SETTINGS)
    
    def _get_cache_key(self, prompt: str, system_instr: str, use_rag: bool) -> str:
        cache_string = f"{prompt}|{system_instr}|{use_rag}|{self.model}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp < RESPONSE_CACHE_TTL:
                return response
            del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        self._response_cache[cache_key] = (response, time.time())
        if len(self._response_cache) > 100:
            oldest_key = min(self._response_cache.keys(), key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]

    def generate_content(
        self, prompt: str, system_instr: str, use_rag: bool = True,
        timeout: int = API_TIMEOUT_SECONDS, stream: bool = False, use_cache: bool = True
    ) -> Union[str, Iterator[str]]:
        cache_key = None
        if use_cache and not stream:
            cache_key = self._get_cache_key(prompt, system_instr, use_rag)
            cached = self._get_cached_response(cache_key)
            if cached: return cached
        try:
            config_args = self._get_config(system_instr).model_dump()
            # Fix: Ensure tools config is passed properly
            if not use_rag: config_args['tools'] = []
            
            if stream and ENABLE_STREAMING:
                if use_cache and not cache_key: cache_key = self._get_cache_key(prompt, system_instr, use_rag)
                response_stream = self.client.models.generate_content_stream(model=self.model, contents=prompt, config=config_args)
                return self._stream_response(response_stream, cache_key)
            else:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config_args)
                full_text = response.text if response.text else None
                if not full_text and response.candidates:
                    cand = response.candidates[0]
                    if cand.content and cand.content.parts:
                        full_text = "".join([p.text for p in cand.content.parts if p.text])
                    if not full_text and cand.finish_reason:
                        if "RECITATION" in str(cand.finish_reason): raise ValueError("RECITATION_ERROR")
                        raise ValueError(f"AI Generation stopped: {cand.finish_reason}")
                if not full_text: raise ValueError("AI response empty.")
                if use_cache: 
                    cache_key = self._get_cache_key(prompt, system_instr, use_rag)
                    self._cache_response(cache_key, full_text)
                return full_text
        except Exception as e:
            if "RECITATION_ERROR" in str(e): raise e
            logger.error(f"GenAI error: {e}")
            raise RuntimeError(str(e))
    
    def _stream_response(self, response_stream: Iterator, cache_key: Optional[str] = None) -> Iterator[str]:
        full_text_parts = []
        try:
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_text_parts.append(chunk.text)
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    cand = chunk.candidates[0]
                    if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_text_parts.append(part.text)
                                yield part.text
                    if hasattr(cand, 'finish_reason') and cand.finish_reason and "RECITATION" in str(cand.finish_reason):
                        raise ValueError("RECITATION_ERROR")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        if cache_key and full_text_parts: self._cache_response(cache_key, "".join(full_text_parts))

    def generate_content_with_audio(self, prompt: str, audio_bytes: bytes, mime_type: str, system_instr: str, use_rag: bool = False) -> str:
        try:
            parts = [prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)]
            config = self._get_config(system_instr, temperature=AUDIO_TEMPERATURE)
            if not use_rag: config.tools = []
            response = self.client.models.generate_content(model=self.model, contents=parts, config=config)
            if response.text: return filter_citation_metadata(response.text)
            if response.candidates and response.candidates[0].content.parts:
                return filter_citation_metadata("".join([p.text for p in response.candidates[0].content.parts if p.text]))
            raise ValueError("No response for audio.")
        except Exception as e:
            logger.error(f"Audio error: {e}")
            raise RuntimeError(str(e))

    def generate_content_multimodal(self, prompt: str, images: List[bytes], system_instr: str) -> str:
        """
        Sends multiple images and a text prompt to the model.
        Used for Visual Grading (Comparing Student Drawing vs Rubric).
        """
        try:
            parts = [prompt]
            for img_bytes in images:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            
            # Disable RAG for visual comparison to focus on image analysis
            config = types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=system_instr,
                tools=[], # No RAG needed for visual comparison
                safety_settings=SAFETY_SETTINGS
            )
            
            response = self.client.models.generate_content(model=self.model, contents=parts, config=config)
            
            if response.text: return response.text
            if response.candidates and response.candidates[0].content.parts:
                return "".join([p.text for p in response.candidates[0].content.parts if p.text])
            raise ValueError("No response for visual input.")
        except Exception as e:
            logger.error(f"Visual grading error: {e}")
            raise RuntimeError(str(e))

class DataManager:
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def _load_full_csv(filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath): return pd.DataFrame()
        try: return pd.read_csv(filepath)
        except: 
            try: return pd.read_csv(filepath, encoding='latin-1')
            except: return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS) 
    def load_csv_sample(filepath: str, selected_tags: List[str], sample_size: int) -> List[Dict[str, Any]]:
        df = DataManager._load_full_csv(filepath)
        if df.empty or "Meta_Tags" not in df.columns: return []
        def matches_tags(row_tags):
            if pd.isna(row_tags): return False
            row_tag_list = [t.strip() for t in str(row_tags).split(',')]
            return any(tag in row_tag_list for tag in selected_tags)
        filtered_df = df[df['Meta_Tags'].apply(matches_tags)]
        if filtered_df.empty: return []
        sample = filtered_df.sample(n=min(len(filtered_df), sample_size), random_state=42)
        return sample.to_dict(orient="records")

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def get_topics_from_csv(filepath: str) -> List[str]:
        df = DataManager._load_full_csv(filepath)
        if df.empty or "Meta_Tags" not in df.columns: return ["General Rules"]
        try:
            all_tags = set()
            for tag_string in df["Meta_Tags"].dropna():
                tags = [tag.strip() for tag in str(tag_string).split(',')]
                all_tags.update(tags)
            return sorted(list(all_tags))
        except: return ["General Rules"]

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def load_frq_bank(filepath: str = "frq_bank.csv") -> List[FreeResponseQuestion]:
        df = DataManager._load_full_csv(filepath)
        if df.empty: return []
        df.columns = df.columns.str.strip()
        questions = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            q_id = str(row_dict.get('ID', '')).strip()
            text = str(row_dict.get('Question_Text', '')).strip()
            if q_id and text:
                questions.append(FreeResponseQuestion(
                    id=q_id, 
                    topic=str(row_dict.get('Topic', 'General')).strip(),
                    question_text=text,
                    ideal_answer=str(row_dict.get('Ideal_Answer', '')).strip()
                ))
        return questions

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def load_visual_bank(filepath: str = "visual_bank.csv") -> List[VisualQuestion]:
        """Loads Visual Questions from CSV."""
        df = DataManager._load_full_csv(filepath)
        if df.empty: return []
        df.columns = df.columns.str.strip()
        questions = []
        base_dir = os.path.dirname(os.path.abspath(filepath)) or "."
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            q_id = str(row_dict.get('ID', '')).strip()
            text = str(row_dict.get('Question_Text', '')).strip()
            raw_img = str(row_dict.get('Image_Path', '')).strip().strip('"').strip("'")
            raw_rubric = str(row_dict.get('Rubric_Path', '')).strip().strip('"').strip("'")
            img_path = os.path.join(base_dir, raw_img) if not os.path.isabs(raw_img) else raw_img
            rubric_path = os.path.join(base_dir, raw_rubric) if not os.path.isabs(raw_rubric) else raw_rubric
            
            if q_id and text and os.path.exists(img_path) and os.path.exists(rubric_path):
                questions.append(VisualQuestion(
                    id=q_id,
                    question_text=text,
                    image_path=img_path,
                    rubric_path=rubric_path
                ))
        return questions

    @staticmethod
    @st.cache_data(ttl=BACKGROUND_IMAGE_CACHE_TTL)
    def get_base64_image(filepath: str) -> Optional[str]:
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f: data = f.read()
            return base64.b64encode(data).decode()
        except: return None

# --- SERVICE LAYER (Business Logic) ---

def generate_with_retry(ai_provider, prompt, safe_prompt, error_context, use_cache=True):
    try:
        return ai_provider.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True, use_cache=use_cache)
    except Exception as e:
        if "RECITATION_ERROR" in str(e):
            logger.warning(f"[{error_context}] Recitation error. Retrying.")
            return ai_provider.generate_content(safe_prompt, SYSTEM_INSTRUCTION, use_rag=True, use_cache=False)
        raise e

def read_audio_bytes(audio_input) -> Optional[bytes]:
    if not audio_input: return None
    try:
        if hasattr(audio_input, 'read'):
            audio_input.seek(0)
            return audio_input.read()
        return bytes(audio_input)
    except: return None

def validate_audio_bytes(audio_bytes, min_size=MIN_AUDIO_SIZE_BYTES):
    if not audio_bytes: return False, "No audio data."
    if len(audio_bytes) < min_size: return False, "Audio too small."
    return True, None

def validate_config():
    if not st.secrets.get(API_KEY_NAME):
        st.error("Missing API Key.")
        return False
    return True

def filter_citation_metadata(text):
    if not text: return text
    return re.sub(r'\[cite:\s*[^\]]+\]', '', text, flags=re.IGNORECASE).strip()

class JSONParser:
    @staticmethod
    def parse(text):
        if not text: raise ValueError("Empty response.")
        clean = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        clean = re.sub(r'```', '', clean).strip()
        start, end = clean.find('['), clean.rfind(']') + 1
        if start == -1 or end == 0:
            s, e = clean.find('{'), clean.rfind('}') + 1
            if s != -1 and e != 0: return [json.loads(clean[s:e])]
            raise ValueError("No valid JSON.")
        return json.loads(clean[start:end])

class QuizService:
    def __init__(self, ai_provider): self.ai = ai_provider
    def calculate_topic_distribution(self, request):
        dist = {}
        active = [t for t in request.topics if t.percentage > 0]
        if not active: return dist
        total, assigned = request.num_questions, 0
        for t in active[:-1]:
            c = (total * t.percentage) // 100
            if c > 0: dist[t.topic_name], assigned = c, assigned + c
        if active: dist[active[-1].topic_name] = total - assigned
        return dist
    def _gen(self, p1, p2): return generate_with_retry(self.ai, p1, p2, "Quiz")
    def generate_from_ai(self, req):
        dist = self.calculate_topic_distribution(req)
        d_str = ",".join([f"{c}x{t}" for t,c in dist.items()])
        p1 = f"Create {req.num_questions} quiz questions from File Store. Topics: {d_str}. JSON Schema: [{{question, options[], correct_answer, reference, explanation}}]"
        p2 = f"Create {req.num_questions} questions. Paraphrase. Topics: {d_str}. JSON Schema: [{{question, options[], correct_answer, reference, explanation}}]"
        return [QuizQuestion(**i) for i in JSONParser.parse(self._gen(p1, p2))]
    def generate_scenarios(self, req):
        dist = self.calculate_topic_distribution(req)
        d_str = ",".join([f"{c}x{t}" for t,c in dist.items()])
        p1 = f"Create {req.num_questions} nav scenario questions from File Store. Topics: {d_str}. JSON Schema: [{{question, options[], correct_answer, reference, explanation}}]"
        p2 = f"Create {req.num_questions} nav scenarios. Paraphrase. Topics: {d_str}. JSON Schema: [{{question, options[], correct_answer, reference, explanation}}]"
        return [QuizQuestion(**i) for i in JSONParser.parse(self._gen(p1, p2))]
    def generate_from_csv(self, data, count):
        opt_data = [{k:v for k,v in r.items() if k in ['Question','Option_A','Correct_Answer','Explanation']} for r in data[:count*2]]
        d_json = json.dumps(opt_data)
        p1 = f"Convert CSV to JSON (max {count}). Data: {d_json}. Schema: [{{question, options[], correct_answer, reference, explanation}}]"
        p2 = f"Convert CSV to JSON. Paraphrase. Data: {d_json}. Schema: ..."
        return [QuizQuestion(**i) for i in JSONParser.parse(self._gen(p1, p2))]

class RemediationService:
    def __init__(self, ai): self.ai = ai
    def _gen_batch(self, batch):
        topics = ",".join(batch[:5])
        return generate_with_retry(self.ai, f"Remediation deep dive: {topics}. Markdown.", f"Remediation summary: {topics}. Paraphrase.", "Remediation")
    def generate_lesson(self, incorrect):
        refs = sorted({q.reference for q in incorrect if q.reference})
        if not refs: return "No topics."
        doc = ["# Remediation Plan", f"**Topics:** {', '.join(refs)}", "---"]
        batches = [refs[i:i+3] for i in range(0, len(refs), 3)]
        with ThreadPoolExecutor(3) as exe:
            f_to_b = {exe.submit(self._gen_batch, b): b for b in batches}
            res = []
            for f in as_completed(f_to_b):
                try: res.append((f_to_b[f], f.result()))
                except: pass
        res.sort(key=lambda x: refs.index(x[0][0]) if x[0] else 0)
        for _, c in res: doc.extend([c, "\n---\n"])
        return "\n\n".join(doc)

class OralBoardService:
    def __init__(self, ai): self.ai = ai
    def generate_scenario(self): return generate_with_retry(self.ai, "Complex Maritime Oral scenario.", "Unique Oral scenario. Paraphrase.", "Oral")
    def grade_response(self, sc, aud): return self.ai.generate_content_with_audio(f"Grade response. Scenario: {sc[:200]}. Output: Transcript, Grade, Critique.", aud, "audio/wav", SYSTEM_INSTRUCTION, True)

class GradingService:
    def __init__(self, ai): self.ai = ai
    def grade_submission(self, q, ans):
        prompt = f"Grade Maritime Exam.\nQ: {q.question_text}\nIdeal: {q.ideal_answer}\nStudent: {ans}\nOutput JSON: {{score: 0-100, reasoning: str}}"
        try:
            txt = self.ai.generate_content(prompt, "Strict grader.", False)
            d = json.loads(re.sub(r'```json\s*|```', '', txt).strip())
            s = int(d.get("score", 0))
            stat = "Correct" if s >= 80 else "Flagged" if s >= 50 else "Incorrect"
            return GradingResult(score=s, status=stat, reasoning=d.get("reasoning",""))
        except Exception as e: return GradingResult(score=0, status="Error", reasoning=str(e))

class VisualGradingService:
    """Handles grading of visual drawing tasks."""
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def grade_drawing(self, question: VisualQuestion, drawing_data: np.ndarray) -> GradingResult:
        """
        Composites the drawing onto the base image and compares with rubric.
        
        Args:
            question: The visual question object
            drawing_data: RGBA numpy array from streamlit-drawable-canvas
            
        Returns:
            GradingResult
        """
        try:
            # 1. Load Base Image
            base_image = Image.open(question.image_path).convert("RGBA")
            
            # 2. Process Drawing (Resize if needed to match base)
            # Canvas might be different size, but usually fits background.
            # We assume canvas was sized to image.
            drawing_image = Image.fromarray(drawing_data.astype('uint8'), 'RGBA')
            
            # Resize drawing to match base if dimensions differ
            if drawing_image.size != base_image.size:
                drawing_image = drawing_image.resize(base_image.size)
            
            # 3. Composite (Overlay drawing on base)
            student_composite = Image.alpha_composite(base_image, drawing_image)
            
            # 4. Convert to Bytes for API
            student_bytes_io = BytesIO()
            student_composite.save(student_bytes_io, format="PNG")
            student_bytes = student_bytes_io.getvalue()
            
            rubric_bytes_io = BytesIO()
            with open(question.rubric_path, "rb") as f:
                rubric_bytes_io.write(f.read())
            rubric_bytes = rubric_bytes_io.getvalue()
            
            # 5. Send to AI
            prompt = f"""
            You are a Visual Grading Assistant for a Maritime Exam.
            
            **TASK:**
            Compare the two images provided.
            - **Image 1:** Student's Answer (The student has drawn/circled an area on the chart/diagram).
            - **Image 2:** Rubric / Answer Key (Shows the correct area highlighted).
            
            **QUESTION ASKED:** {question.question_text}
            
            **GRADING CRITERIA:**
            1. Did the student circle/highlight the SAME object or area as shown in the Rubric?
            2. Is the marking reasonably accurate (it captures the intended target)?
            3. Ignore minor sloppiness in drawing lines. Focus on the location.
            
            **OUTPUT FORMAT (JSON ONLY):**
            {{
                "score": <100 for correct location, 0 for incorrect>,
                "reasoning": "<Concise explanation>"
            }}
            """
            
            # Send: [Student Image, Rubric Image] + Prompt
            response_text = self.ai.generate_content_multimodal(
                prompt=prompt,
                images=[student_bytes, rubric_bytes],
                system_instr="You are a strict visual grader."
            )
            
            # 6. Parse Result
            clean_text = re.sub(r'```json\s*|```', '', response_text).strip()
            data = json.loads(clean_text)
            score = int(data.get("score", 0))
            
            status = "Correct" if score >= 80 else "Incorrect"
            return GradingResult(score=score, status=status, reasoning=data.get("reasoning", ""))
            
        except Exception as e:
            logger.error(f"Visual grading error: {e}")
            return GradingResult(score=0, status="Error", reasoning=f"System Error: {str(e)}")

# --- PRESENTATION LAYER ---

def init_session_state():
    defaults = {
        "messages": [{"role": "assistant", "content": "👋 Maritime Expert Ready."}],
        "quiz_data": None,
        "current_quiz_source": None,
        "incorrect_questions": [],
        "remediation_text": None,
        "oral_scenario": None,
        "oral_grade": None,
        "frq_bank": [],
        "frq_results": {},
        "visual_bank": [], # Store visual questions
        "visual_results": {} 
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

def set_background():
    bin_str = DataManager.get_base64_image('background.png')
    if bin_str:
        st.markdown(f'''<style>.stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; }}</style>''', unsafe_allow_html=True)

def render_visual_quiz(visual_service: VisualGradingService):
    st.title("🎨 Visual Identification Quiz")
    st.caption("Draw on the image to identify the correct vessel, light, or feature.")
    
    if not HAS_CANVAS:
        st.error("⚠️ `streamlit-drawable-canvas` is missing. Please run `pip install streamlit-drawable-canvas`.")
        return

    # Load Questions
    if not st.session_state.visual_bank:
        v_questions = DataManager.load_visual_bank()
        if not v_questions:
            st.warning("⚠️ No visual questions found. Ensure `visual_bank.csv` exists and images are in the path.")
            with st.expander("Setup Instructions"):
                st.markdown("""
                1. Create `visual_bank.csv` with columns: `ID`, `Question_Text`, `Image_Path`, `Rubric_Path`.
                2. Ensure image files (PNG/JPG) exist in the project folder.
                """)
            return
        st.session_state.visual_bank = v_questions

    # Render Active Question
    # For PoC, just show a selector or iterate. Let's do a selector for simplicity.
    q_options = {f"Q{q.id}: {q.question_text}": q for q in st.session_state.visual_bank}
    selected_label = st.selectbox("Select Question:", list(q_options.keys()))
    question = q_options[selected_label]
    
    st.markdown(f"**Task:** {question.question_text}")
    
    # Load Base Image for Canvas
    try:
        base_img = Image.open(question.image_path)
        # Resize for display consistency if needed (max width)
        max_width = 600
        if base_img.width > max_width:
            ratio = max_width / base_img.width
            base_img = base_img.resize((max_width, int(base_img.height * ratio)))
            
        # Canvas Component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Translucent orange
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=base_img,
            update_streamlit=True,
            height=base_img.height,
            width=base_img.width,
            drawing_mode="freedraw",
            key=f"canvas_{question.id}"
        )
        
        if st.button("Submit Drawing", key=f"v_btn_{question.id}"):
            if canvas_result.image_data is not None:
                with st.spinner("Analyzing your drawing against the rubric..."):
                    result = visual_service.grade_drawing(question, canvas_result.image_data)
                    st.session_state.visual_results[question.id] = result
            else:
                st.warning("Please draw on the image first.")
        
        # Display Result
        if question.id in st.session_state.visual_results:
            res = st.session_state.visual_results[question.id]
            if res.status == "Correct":
                st.success(f"✅ Correct! (Score: {res.score})")
            else:
                st.error(f"❌ Incorrect. (Score: {res.score})")
            st.info(f"**Feedback:** {res.reasoning}")
            
            with st.expander("View Answer Key (Rubric)"):
                st.image(question.rubric_path, caption="Correct Answer Highlighted")

    except Exception as e:
        err_msg = str(e)
        if "image_to_url" in err_msg:
            st.error("Visual Quiz requires a compatibility fix for your Streamlit version.")
            st.markdown("""
            The `streamlit-drawable-canvas` package uses an outdated Streamlit API. Install the maintained fork:

            ```
            pip uninstall streamlit-drawable-canvas
            pip install streamlit-drawable-canvas-fix
            ```

            Then restart the app and try again.
            """)
        else:
            st.error(f"Error loading image: {e}")

def main():
    st.set_page_config(page_title="RAG Engine", page_icon="⚓", layout="wide")
    set_background()
    init_session_state()
    if not validate_config(): st.stop()
    
    try:
        ai_provider = GenAIProvider(st.secrets[API_KEY_NAME], st.secrets[STORE_ID_NAME])
        quiz_service = QuizService(ai_provider)
        remediation_service = RemediationService(ai_provider)
        oral_service = OralBoardService(ai_provider)
        grading_service = GradingService(ai_provider)
        visual_service = VisualGradingService(ai_provider) # New Service
    except KeyError:
        st.error("Missing Secrets"); st.stop()

    with st.sidebar:
        st.title("⚓ Maritime AI")
        mode = st.radio("Tool:", ["Chat 🤖", "Quiz 📝", "Free Response ✍️", "Visual Quiz 🎨", "Oral Board 🎙️"])
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.caption("✅ v19.0 - Visual Grading Active")

    # Routing Logic
    if mode == "Chat 🤖":
        st.title("Regulations Chat")
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": p})
            st.chat_message("user").write(p)
            with st.spinner("Thinking..."):
                try:
                    if ENABLE_STREAMING:
                        ph = st.empty()
                        full = ""
                        for c in ai_provider.generate_content(p, SYSTEM_INSTRUCTION, True, stream=True):
                            full += c
                            ph.write(full + "▌")
                        ph.write(full)
                        r = full
                    else: r = ai_provider.generate_content(p, SYSTEM_INSTRUCTION)
                    st.session_state.messages.append({"role": "assistant", "content": r})
                except Exception as e: st.error(f"Error: {e}")

    elif mode == "Quiz 📝":
        st.title("Knowledge Check")
        def clear(): 
            st.session_state.quiz_data = None
            st.session_state.incorrect_questions = []
            st.session_state.remediation_text = None
        source = st.radio("Source:", ["CSV", "AI Gen"], horizontal=True, on_change=clear, key="q_src")
        
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
            style = st.radio("Style:", ["Standard Rules", "Applied Scenarios"], horizontal=True)
            topics = []
            for i in range(1, 5):
                c1, c2 = st.columns([3, 1])
                t = c1.text_input(f"Topic {i}")
                p = c2.number_input(f"%", 0, 100, 0, key=f"p{i}")
                if t: topics.append(TopicDistribution(topic_name=t, percentage=p))
            n = st.number_input("Count", 1, 25, 5)
            if st.button("Generate"):
                req = QuizRequest(num_questions=n, topics=topics)
                if style == "Standard Rules": st.session_state.quiz_data = quiz_service.generate_from_ai(req)
                else: st.session_state.quiz_data = quiz_service.generate_scenarios(req)
                st.session_state.current_quiz_source = f"AI ({style})"
                st.rerun()

        if st.session_state.quiz_data:
            with st.form("quiz"):
                score, answers = 0, {}
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**{i+1}. {q.question}**")
                    answers[i] = st.radio("Opt", q.options, key=f"q{i}", label_visibility="collapsed", index=None)
                    st.write("---")
                if st.form_submit_button("Submit"):
                    inc = []
                    for i, q in enumerate(st.session_state.quiz_data):
                        if answers.get(i) == q.correct_answer: score += 1; st.success("Correct")
                        else: st.error("Incorrect"); st.write(f"Correct: {q.correct_answer}"); inc.append(q)
                        st.info(q.explanation)
                    st.session_state.incorrect_questions = inc
                    st.metric("Score", f"{score}/{len(st.session_state.quiz_data)}")
            
            if st.session_state.incorrect_questions:
                st.divider()
                if st.button("Generate Remediation"):
                    st.session_state.remediation_text = remediation_service.generate_lesson(st.session_state.incorrect_questions)
                if st.session_state.remediation_text:
                    st.markdown(st.session_state.remediation_text)
                    if HAS_REPORTLAB:
                        pdf = PDFGenerator.create_study_guide(st.session_state.remediation_text)
                        if pdf: st.download_button("PDF", pdf, "remediation.pdf", "application/pdf")

    elif mode == "Free Response ✍️":
        render_frq_section(grading_service)
    
    elif mode == "Visual Quiz 🎨":
        render_visual_quiz(visual_service)

    elif mode == "Oral Board 🎙️":
        render_oral_board(oral_service)

if __name__ == "__main__":
    main()