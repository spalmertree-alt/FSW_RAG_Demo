import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import json
import base64
import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- PDF GENERATION IMPORTS ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# --- VISUAL QUIZ (drawable canvas) ---
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
API_TIMEOUT_SECONDS = 120
BACKGROUND_IMAGE_CACHE_TTL = 86400  # 24 hours
MIN_AUDIO_SIZE_BYTES = 100
WAV_HEADER_SIZE = 12
AUDIO_TEMPERATURE = 0.2  # For audio grading

# Safety settings for GenAI API
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Check if we're in Streamlit context
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

class FreeResponseQuestion(BaseModel):
    """Represents a Free Response Question from the curriculum bank."""
    id: str
    topic: str
    question_text: str
    ideal_answer: str

class GradingResult(BaseModel):
    """Represents the AI grading of a free response."""
    score: int
    status: str  # "Correct", "Incorrect", "Flagged"
    reasoning: str

class VisualQuestion(BaseModel):
    """Represents a visual identification question (draw on image)."""
    id: str
    question_text: str
    image_path: str
    rubric_path: str

# --- INFRASTRUCTURE LAYER (External IO) ---

class PDFGenerator:
    """Handles PDF generation using ReportLab."""
    
    # Pre-compiled regex patterns for performance
    _BOLD_PATTERN = re.compile(r'\*\*(.*?)\*\*')
    _NUMBERED_PATTERN = re.compile(r'^\d+\.')
    
    @staticmethod
    def create_study_guide(content: str) -> Optional[BytesIO]:
        """
        Generate a PDF study guide from markdown content.
        
        Args:
            content: Markdown-formatted text content
            
        Returns:
            BytesIO buffer containing PDF data, or None if generation fails
        """
        if not HAS_REPORTLAB:
            return None

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
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
    """Encapsulates Google GenAI interactions."""
    
    def __init__(self, api_key: str, store_id: str):
        """
        Initialize the GenAI provider.
        
        Args:
            api_key: Google Gemini API key
            store_id: File search store ID for RAG
        """
        self.client = genai.Client(api_key=api_key)
        self.store_id = store_id
        self.model = MODEL_ID
        self._response_cache: Dict[str, tuple[str, float]] = {}

    def __del__(self):
        try:
            if hasattr(self, 'client'): self.client.close()
        except: pass

    def _get_config(self, system_instr: str, temperature: float = TEMPERATURE) -> types.GenerateContentConfig:
        """Helper to create config with tools and safety settings."""
        tools = []
        if self.store_id:
            tools = [types.Tool(
                file_search=types.FileSearch(file_search_store_names=[self.store_id])
            )]

        return types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instr,
            tools=tools,
            safety_settings=SAFETY_SETTINGS
        )
    
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
        self, 
        prompt: str, 
        system_instr: str, 
        use_rag: bool = True,
        response_schema: Any = None,
        timeout: int = API_TIMEOUT_SECONDS,
        stream: bool = False,
        use_cache: bool = True
    ) -> Union[str, Iterator[str]]:
        """
        Generate content using Google GenAI with optional RAG.
        """
        cache_key = None
        if use_cache and not stream:
            cache_key = self._get_cache_key(prompt, system_instr, use_rag)
            cached = self._get_cached_response(cache_key)
            if cached: return cached

        try:
            tools = []
            if use_rag and self.store_id:
                tools = [types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[self.store_id])
                )]

            config_args = {
                "temperature": TEMPERATURE,
                "system_instruction": system_instr,
                "safety_settings": SAFETY_SETTINGS
            }
            if tools: 
                config_args["tools"] = tools
            
            if stream and ENABLE_STREAMING:
                if use_cache and not cache_key: cache_key = self._get_cache_key(prompt, system_instr, use_rag)
                response_stream = self.client.models.generate_content_stream(model=self.model, contents=prompt, config=types.GenerateContentConfig(**config_args))
                return self._stream_response(response_stream, cache_key)
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args)
                )
                
                if response.text:
                    full_text = response.text
                elif response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        parts_text = [p.text for p in candidate.content.parts if p.text]
                        if parts_text:
                            full_text = "".join(parts_text)
                    
                    if not full_text and candidate.finish_reason:
                        reason = str(candidate.finish_reason)
                        if "RECITATION" in reason:
                            raise ValueError("RECITATION_ERROR")
                        raise ValueError(f"AI Generation stopped. Reason: {reason}")
                else:
                    raise ValueError("The AI returned an empty response (No candidates or text found).")

                if use_cache and cache_key:
                    self._cache_response(cache_key, full_text)
                return full_text

        except Exception as e:
            if "RECITATION_ERROR" in str(e):
                raise e 
            logger.error(f"GenAI generation failed: {e}")
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

    def generate_content_with_audio(
        self,
        prompt: str,
        audio_bytes: bytes,
        mime_type: str,
        system_instr: str,
        use_rag: bool = False
    ) -> str:
        """
        Generates content using Multimodal inputs (Text + Audio).
        """
        try:
            if not audio_bytes:
                raise ValueError("Audio bytes are empty. Please ensure you recorded audio.")
            
            if len(audio_bytes) < MIN_AUDIO_SIZE_BYTES:
                raise ValueError(f"Audio file is too small ({len(audio_bytes)} bytes). Please record a longer audio clip.")
            
            if mime_type == "audio/wav":
                if len(audio_bytes) < WAV_HEADER_SIZE:
                    raise ValueError("Audio file is too small to be a valid WAV file.")
                if not (audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'):
                    logger.warning("Audio file may not be a valid WAV format, but attempting to process anyway.")
            
            logger.info(f"Processing audio: {len(audio_bytes)} bytes, MIME type: {mime_type}, RAG: {use_rag}")
            
            parts = [
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ]
            
            tools = []
            if use_rag and self.store_id:
                tools = [types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[self.store_id])
                )]
            
            config = types.GenerateContentConfig(
                temperature=AUDIO_TEMPERATURE,
                system_instruction=system_instr,
                tools=tools,
                safety_settings=SAFETY_SETTINGS
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config=config
            )
            
            if response.text:
                return filter_citation_metadata(response.text)
            
            if response.candidates:
                if response.candidates[0].content.parts:
                    parts_text = [p.text for p in response.candidates[0].content.parts if p.text]
                    if parts_text:
                        return filter_citation_metadata("".join(parts_text))
                
                if response.candidates[0].finish_reason:
                    reason = str(response.candidates[0].finish_reason)
                    if "RECITATION" in reason:
                        logger.warning("Audio grading hit RECITATION error")
                        raise ValueError("RECITATION_ERROR")
                    raise ValueError(f"AI Generation stopped. Reason: {reason}")
            
            raise ValueError("No response generated for audio input.")

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            if "RECITATION_ERROR" in str(e):
                raise e
            raise RuntimeError(f"Audio grading failed: {str(e)}")

    def generate_content_multimodal(self, prompt: str, images: List[bytes], system_instr: str) -> str:
        """Send images + text to model. Used for Visual Quiz grading (no RAG)."""
        try:
            parts = [prompt]
            for img_bytes in images:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            config = types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=system_instr,
                tools=[],
                safety_settings=SAFETY_SETTINGS
            )
            response = self.client.models.generate_content(
                model=self.model, contents=parts, config=config
            )
            if response.text:
                return response.text
            if response.candidates and response.candidates[0].content.parts:
                return "".join([p.text for p in response.candidates[0].content.parts if p.text])
            raise ValueError("No response for visual input.")
        except Exception as e:
            logger.error(f"Visual grading error: {e}")
            raise RuntimeError(str(e))

class DataManager:
    """Handles File I/O operations."""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def _load_full_csv(filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            return pd.DataFrame()
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.warning(f"CSV loading error: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS) 
    def load_csv_sample(filepath: str, selected_tags: List[str], sample_size: int) -> List[Dict[str, Any]]:
        df = DataManager._load_full_csv(filepath)
        if df.empty or "Meta_Tags" not in df.columns:
            return []
        
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
        if df.empty or "Meta_Tags" not in df.columns:
            return ["General Rules"]
        try:
            all_tags = set()
            for tag_string in df["Meta_Tags"].dropna():
                tags = [tag.strip() for tag in str(tag_string).split(',')]
                all_tags.update(tags)
            return sorted(list(all_tags))
        except Exception: return ["General Rules"]

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def load_frq_bank(filepath: str = "frq_bank.csv") -> List[FreeResponseQuestion]:
        # Resolve file path - try current directory first
        if not os.path.isabs(filepath):
            full_path = os.path.join(os.getcwd(), filepath)
        else:
            full_path = filepath
        
        # Check if file exists first
        if not os.path.exists(full_path):
            logger.error(f"FRQ bank file not found: {full_path}. Current working directory: {os.getcwd()}")
            # Try alternative paths
            alt_paths = [
                filepath,  # Try as-is
                os.path.join(".", filepath),  # Try relative
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    full_path = alt_path
                    logger.info(f"Found file at alternative path: {full_path}")
                    break
            else:
                logger.error(f"Could not find frq_bank.csv in any location. Searched: {[full_path] + alt_paths}")
                return []
        
        logger.info(f"Loading FRQ bank from: {full_path}")
        df = DataManager._load_full_csv(full_path)
        if df.empty:
            logger.error(f"CSV file {full_path} is empty or could not be loaded")
            return []
        
        # Log what we found
        logger.info(f"Loaded CSV with {len(df)} rows. Columns: {list(df.columns)}")
        
        questions = []
        try:
            # Normalize column names (case-insensitive, strip whitespace)
            column_map = {col.lower().strip(): col for col in df.columns}
            logger.info(f"Column mapping: {column_map}")
            
            # Check for required columns
            required_cols = ['id', 'question_text']
            missing_cols = [col for col in required_cols if col not in column_map]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}")
                return []
            
            rows_processed = 0
            rows_skipped = 0
            for idx, row in df.iterrows():
                rows_processed += 1
                # Convert pandas Series to dict for safe access
                row_dict = row.to_dict()
                
                # Get column names (handle case variations)
                id_col = column_map.get('id', 'ID')
                topic_col = column_map.get('topic', 'Topic')
                question_col = column_map.get('question_text', 'Question_Text')
                answer_col = column_map.get('ideal_answer', 'Ideal_Answer')
                
                # Handle NaN values and convert to strings
                q_id = str(row_dict.get(id_col, '')) if pd.notna(row_dict.get(id_col)) else ''
                topic = str(row_dict.get(topic_col, 'General')) if pd.notna(row_dict.get(topic_col)) else 'General'
                question_text = str(row_dict.get(question_col, '')) if pd.notna(row_dict.get(question_col)) else ''
                ideal_answer = str(row_dict.get(answer_col, '')) if pd.notna(row_dict.get(answer_col)) else ''
                
                # Skip rows with empty required fields
                if not q_id or not question_text:
                    rows_skipped += 1
                    logger.warning(f"Skipping row {idx+1}: missing ID or Question_Text. ID={q_id}, Q={question_text[:50] if question_text else 'None'}")
                    continue
                
                try:
                    questions.append(FreeResponseQuestion(
                        id=q_id,
                        topic=topic,
                        question_text=question_text,
                        ideal_answer=ideal_answer
                    ))
                except Exception as validation_error:
                    logger.error(f"Validation error on row {idx+1}: {validation_error}")
                    continue
            
            logger.info(f"Successfully loaded {len(questions)} questions from {full_path} (processed {rows_processed} rows, skipped {rows_skipped})")
            if len(questions) == 0 and rows_processed > 0:
                logger.warning(f"No valid questions found after processing {rows_processed} rows. Check column names and data format.")
            return questions
        except Exception as e:
            logger.error(f"Error parsing FRQ bank from {filepath}: {e}", exc_info=True)
            return []

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
        except Exception: return None

# --- SERVICE LAYER (Business Logic) ---

def generate_with_retry(
    ai_provider: GenAIProvider,
    prompt: str,
    safe_prompt: str,
    error_context: str = "generation",
    use_cache: bool = True
) -> str:
    try:
        return ai_provider.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True, use_cache=use_cache)
    except Exception as e:
        if "RECITATION_ERROR" in str(e):
            logger.warning(f"[{error_context}] Hit Recitation error. Retrying.")
            return ai_provider.generate_content(safe_prompt, SYSTEM_INSTRUCTION, use_rag=True, use_cache=False)
        raise e

def read_audio_bytes(audio_input) -> Optional[bytes]:
    if not audio_input: return None
    try:
        if hasattr(audio_input, 'read'):
            audio_input.seek(0)
            return audio_input.read()
        return bytes(audio_input)
    except Exception as e:
        logger.error(f"Error reading audio: {e}")
        return None

def validate_audio_bytes(audio_bytes: Optional[bytes], min_size: int = MIN_AUDIO_SIZE_BYTES) -> tuple[bool, Optional[str]]:
    if not audio_bytes: return False, "No audio data received."
    if len(audio_bytes) < min_size: return False, "Audio file too small."
    return True, None

def validate_config() -> bool:
    if not st.secrets.get(API_KEY_NAME):
        st.error("Missing API Key.")
        return False
    return True

def filter_citation_metadata(text: str) -> str:
    if not text: return text
    citation_pattern = re.compile(r'\[cite:\s*[^\]]+\]', re.IGNORECASE)
    cleaned = citation_pattern.sub('', text)
    return cleaned.strip()

class JSONParser:
    @staticmethod
    def parse(text: Optional[str]) -> List[Dict[str, Any]]:
        if not text: raise ValueError("AI response empty.")
        clean_text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```', '', clean_text).strip()
        
        start = clean_text.find('[')
        end = clean_text.rfind(']') + 1
        
        if start == -1 or end == 0:
            obj_start = clean_text.find('{')
            obj_end = clean_text.rfind('}') + 1
            if obj_start != -1 and obj_end != 0:
                try: return [json.loads(clean_text[obj_start:obj_end])]
                except: pass
            raise ValueError("No valid JSON found.")
        
        return json.loads(clean_text[start:end])

class QuizService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def calculate_topic_distribution(self, request: QuizRequest) -> Dict[str, int]:
        distribution = {}
        total = request.num_questions
        active = [t for t in request.topics if t.percentage > 0]
        if not active: return distribution
        
        assigned = 0
        for i, topic in enumerate(active[:-1]):
            count = (total * topic.percentage) // 100
            if count > 0:
                distribution[topic.topic_name] = count
                assigned += count
        
        if active: distribution[active[-1].topic_name] = total - assigned
        return distribution

    def _generate_with_retry(self, prompt: str, safe_prompt: str, use_cache: bool = True) -> str:
        return generate_with_retry(self.ai, prompt, safe_prompt, "Quiz generation", use_cache=use_cache)

    def generate_from_ai(self, request: QuizRequest) -> List[QuizQuestion]:
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} questions about: {t}" for t, c in dist.items()])
        prompt = f"""
        Create a {request.num_questions} question multiple-choice quiz based ONLY on the provided File Store.
        DISTRIBUTION: {dist_str}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        prompt_safe = f"""
        Create a {request.num_questions} question quiz. Paraphrase content to avoid recitation.
        DISTRIBUTION: {dist_str}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        raw = self._generate_with_retry(prompt, prompt_safe)
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]
    
    def generate_scenarios(self, request: QuizRequest) -> List[QuizQuestion]:
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} scenarios involving: {t}" for t, c in dist.items()])
        prompt = f"""
        Create {request.num_questions} navigational scenario questions based on File Store.
        DISTRIBUTION: {dist_str}
        INSTRUCTIONS: Vivid scenarios, bridge perspective.
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        prompt_safe = f"""
        Create {request.num_questions} navigational scenario questions. Paraphrase content.
        DISTRIBUTION: {dist_str}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        raw = self._generate_with_retry(prompt, prompt_safe)
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

    def generate_from_csv(self, raw_data: List[Dict], count: int) -> List[QuizQuestion]:
        prompt = f"""
        Convert this CSV data to JSON quiz.
        Data: {json.dumps(raw_data)}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        prompt_safe = f"""
        Convert this CSV data to JSON quiz. Paraphrase content.
        Data: {json.dumps(raw_data)}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        raw = self._generate_with_retry(prompt, prompt_safe)
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

class RemediationService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def _generate_batch_content(self, batch_refs: List[str]) -> str:
        topics = ", ".join(batch_refs)
        prompt = f"Create 'Deep Dive' remediation for: {topics}. Markdown format."
        prompt_safe = f"Create summary remediation for: {topics}. Paraphrase text."
        try:
            return generate_with_retry(self.ai, prompt, prompt_safe, "Remediation Batch")
        except Exception as e:
            return f"**Error: {str(e)}**"

    def generate_lesson(self, incorrect_questions: List[QuizQuestion]) -> str:
        unique_refs = sorted({q.reference for q in incorrect_questions if q.reference})
        if not unique_refs: return "No topics found."

        full_doc = ["# Remediation Lesson Plan", f"**Focus Topics:** {', '.join(unique_refs)}", "---"]
        batches = [unique_refs[i:i+REMEDIATION_BATCH_SIZE] for i in range(0, len(unique_refs), REMEDIATION_BATCH_SIZE)]
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_batch = {executor.submit(self._generate_batch_content, b): b for b in batches}
            results = []
            for future in as_completed(future_to_batch):
                try: results.append((future_to_batch[future], future.result()))
                except: pass
        
        results.sort(key=lambda x: unique_refs.index(x[0][0]) if x[0] else 0)
        for _, content in results:
            full_doc.append(content)
            full_doc.append("\n---\n")
            
        return "\n\n".join(full_doc)

class OralBoardService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def generate_scenario(self) -> str:
        prompt = "Generate a complex Oral Board scenario for Maritime Officer. Subject: Rules of the Road."
        safe = "Generate a unique Oral Board scenario. Paraphrase."
        return generate_with_retry(self.ai, prompt, safe, "Oral Scenario")

    def grade_response(self, scenario: str, audio_bytes: bytes) -> str:
        prompt = f"Grade this audio response to scenario: {scenario}. Output: Transcript, Grade (PASS/FAIL), Critique."
        return self.ai.generate_content_with_audio(prompt, audio_bytes, "audio/wav", SYSTEM_INSTRUCTION, use_rag=True)

class GradingService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def grade_submission(self, question: FreeResponseQuestion, student_answer: str) -> GradingResult:
        prompt = f"""
        Grade Maritime Exam.
        Q: {question.question_text}
        Ideal: {question.ideal_answer}
        Student: {student_answer}
        Output JSON: {{score: 0-100, reasoning: str}}
        """
        try:
            txt = self.ai.generate_content(prompt, "Strict grader.", False)
            d = json.loads(re.sub(r'```json\s*|```', '', txt).strip())
            s = int(d.get("score", 0))
            stat = "Correct" if s >= 80 else "Flagged" if s >= 50 else "Incorrect"
            return GradingResult(score=s, status=stat, reasoning=d.get("reasoning",""))
        except Exception as e: return GradingResult(score=0, status="Error", reasoning=str(e))

class VisualGradingService:
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def grade_drawing(self, question: VisualQuestion, drawing_data: np.ndarray) -> GradingResult:
        try:
            base_image = Image.open(question.image_path).convert("RGBA")
            drawing_image = Image.fromarray(drawing_data.astype('uint8'), 'RGBA')
            if drawing_image.size != base_image.size:
                drawing_image = drawing_image.resize(base_image.size)
            student_composite = Image.alpha_composite(base_image, drawing_image)
            
            student_bytes_io = BytesIO()
            student_composite.save(student_bytes_io, format="PNG")
            student_bytes = student_bytes_io.getvalue()
            
            rubric_bytes_io = BytesIO()
            with open(question.rubric_path, "rb") as f:
                rubric_bytes_io.write(f.read())
            rubric_bytes = rubric_bytes_io.getvalue()
            
            prompt = f"Compare images. Image 1: Student answer (drawing). Image 2: Answer key. Q: {question.question_text}. JSON: {{score: 100/0, reasoning: str}}"
            response_text = self.ai.generate_content_multimodal(prompt, [student_bytes, rubric_bytes], "Strict visual grader.")
            
            clean_text = re.sub(r'```json\s*|```', '', response_text).strip()
            data = json.loads(clean_text)
            score = int(data.get("score", 0))
            status = "Correct" if score >= 80 else "Incorrect"
            return GradingResult(score=score, status=status, reasoning=data.get("reasoning", ""))
        except Exception as e:
            logger.error(f"Visual grading error: {e}")
            return GradingResult(score=0, status="Error", reasoning=f"System Error: {str(e)}")

# --- PRESENTATION LAYER ---

def set_background():
    bin_str = DataManager.get_base64_image('background.png')
    if bin_str:
        st.markdown(f'''<style>
        .stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; }}
        </style>''', unsafe_allow_html=True)

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
        "visual_bank": [], 
        "visual_results": {} 
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

def render_visual_quiz(visual_service: VisualGradingService):
    st.subheader("🎨 Visual Identification Quiz")
    st.caption("Draw on the image to identify the correct vessel, light, or feature.")
    
    if not HAS_CANVAS:
        st.error("⚠️ `streamlit-drawable-canvas` is missing.")
        return

    if not st.session_state.visual_bank:
        v_questions = DataManager.load_visual_bank()
        if not v_questions:
            st.warning("⚠️ No visual questions found. Ensure `visual_bank.csv` exists.")
            return
        st.session_state.visual_bank = v_questions

    q_options = {f"Q{q.id}: {q.question_text}": q for q in st.session_state.visual_bank}
    selected_label = st.selectbox("Select Question:", list(q_options.keys()))
    question = q_options[selected_label]
    
    st.markdown(f"**Task:** {question.question_text}")
    
    try:
        base_img = Image.open(question.image_path)
        max_width = 600
        if base_img.width > max_width:
            ratio = max_width / base_img.width
            base_img = base_img.resize((max_width, int(base_img.height * ratio)))
            
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
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
                with st.spinner("Analyzing..."):
                    result = visual_service.grade_drawing(question, canvas_result.image_data)
                    st.session_state.visual_results[question.id] = result
            else:
                st.warning("Please draw on the image first.")
        
        if question.id in st.session_state.visual_results:
            res = st.session_state.visual_results[question.id]
            if res.status == "Correct":
                st.success(f"✅ Correct! (Score: {res.score})")
            else:
                st.error(f"❌ Incorrect. (Score: {res.score})")
            st.info(f"**Feedback:** {res.reasoning}")
            
            with st.expander("View Answer Key"):
                st.image(question.rubric_path, caption="Correct Answer")

    except Exception as e:
        st.error(f"Error loading image: {e}")

def render_frq_section(grading_service: GradingService):
    st.subheader("✍️ Free Response Practice")
    
    if not st.session_state.frq_bank:
        frq_questions = DataManager.load_frq_bank()
        if not frq_questions:
            st.warning("⚠️ No questions found.")
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                # Simplified debug info to match paste
                st.write(f"Looking for: frq_bank.csv")
            
            # Create dummy CSV instructions for user
            with st.expander("How to set up Free Response Questions"):
                st.markdown("""
                Create a file named **frq_bank.csv** in your project folder with these columns:
                `ID`, `Topic`, `Question_Text`, `Ideal_Answer`
                
                **Example Content:**
                ```csv
                ID,Topic,Question_Text,Ideal_Answer
                1,Rule 5,Define a proper look-out.,"Maintain proper look-out by sight and hearing..."
                ```
                """)
            return
        st.session_state.frq_bank = frq_questions

    for q in st.session_state.frq_bank:
        with st.container():
            st.markdown(f"**Q{q.id}: {q.question_text}**")
            
            # Input key unique to question ID
            user_input = st.text_area(f"Your Answer ({q.topic}):", key=f"frq_input_{q.id}")
            
            if st.button(f"Submit Answer {q.id}", key=f"btn_{q.id}"):
                if not user_input.strip():
                    st.warning("Please type an answer first.")
                else:
                    with st.spinner("Analyzing response..."):
                        result = grading_service.grade_submission(q, user_input)
                        st.session_state.frq_results[q.id] = result
            
            # Display Result if available
            if q.id in st.session_state.frq_results:
                res = st.session_state.frq_results[q.id]
                
                if res.status == "Correct":
                    st.success(f"✅ Correct ({res.score}%)")
                elif res.status == "Flagged for Review":
                    st.warning(f"⚠️ Flagged for Manual Grading ({res.score}%)")
                else:
                    st.error(f"❌ Incorrect ({res.score}%)")
                
                st.info(f"**AI Reasoning:** {res.reasoning}")
                with st.expander("Show Official Answer"):
                    st.write(q.ideal_answer)
            
            st.divider()

def render_oral_board(oral_service: OralBoardService):
    st.subheader("🎙️ Oral Board Simulator")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Generate New Scenario"):
            with st.spinner("Preparing..."):
                st.session_state.oral_scenario = oral_service.generate_scenario()
                st.session_state.oral_grade = None
                st.rerun()

    if st.session_state.oral_scenario:
        st.info(st.session_state.oral_scenario)
        audio_input = st.audio_input("Record Answer", key="audio")
        if audio_input and not st.session_state.oral_grade:
            with st.spinner("Grading..."):
                try:
                    bytes_data = read_audio_bytes(audio_input)
                    if bytes_data:
                        st.session_state.oral_grade = oral_service.grade_response(st.session_state.oral_scenario, bytes_data)
                except Exception as e: st.error(str(e))
        
        if st.session_state.oral_grade:
            st.divider()
            st.markdown(st.session_state.oral_grade)

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
        visual_service = VisualGradingService(ai_provider)
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

    if mode == "Chat 🤖":
        st.title("Regulations Chat")
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": p})
            st.chat_message("user").write(p)
            with st.spinner("Thinking..."):
                try:
                    if ENABLE_STREAMING:
                        ph, full = st.empty(), ""
                        for c in ai_provider.generate_content(p, SYSTEM_INSTRUCTION, True, stream=True):
                            full += c; ph.write(full + "▌")
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
        
        t1, t2 = st.tabs(["Configuration", "Active Quiz"])
        
        with t1:
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

        with t2:
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
            else:
                st.info("Go to 'Configuration' tab to start a quiz.")

    elif mode == "Free Response ✍️":
        render_frq_section(grading_service)
    
    elif mode == "Visual Quiz 🎨":
        render_visual_quiz(visual_service)

    elif mode == "Oral Board 🎙️":
        render_oral_board(oral_service)

if __name__ == "__main__":
    main()