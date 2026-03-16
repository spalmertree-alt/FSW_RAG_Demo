import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import json
import base64
import re
import logging
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

    def generate_content(
        self, 
        prompt: str, 
        system_instr: str, 
        use_rag: bool = True,
        response_schema: Any = None,
        timeout: int = API_TIMEOUT_SECONDS
    ) -> str:
        """
        Generate content using Google GenAI with optional RAG.
        
        Note: Google GenAI has rate limits. For high-volume usage, consider:
        - Adding delays between requests
        - Implementing exponential backoff
        - Caching responses where appropriate
        
        Args:
            prompt: User prompt text
            system_instr: System instruction text
            use_rag: Whether to use RAG file search
            response_schema: Optional response schema (not currently used)
            timeout: Request timeout in seconds (for future implementation)
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If RECITATION_ERROR occurs or response is empty
            RuntimeError: If API call fails
        """
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
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args)
            )
            
            if response.text:
                return response.text
            
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    parts_text = [p.text for p in candidate.content.parts if p.text]
                    if parts_text:
                        return "".join(parts_text)
                
                if candidate.finish_reason:
                    reason = str(candidate.finish_reason)
                    if "RECITATION" in reason:
                        raise ValueError("RECITATION_ERROR")
                    raise ValueError(f"AI Generation stopped. Reason: {reason}")

            raise ValueError("The AI returned an empty response (No candidates or text found).")

        except Exception as e:
            if "RECITATION_ERROR" in str(e):
                raise e 
            logger.error(f"GenAI generation failed: {e}")
            raise RuntimeError(str(e))

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
        
        Args:
            prompt: Text prompt
            audio_bytes: Audio data bytes
            mime_type: MIME type of audio (e.g., "audio/wav")
            system_instr: System instruction
            use_rag: Whether to enable RAG (default False for audio grading to avoid citation metadata)
        """
        try:
            # Validate audio bytes
            if not audio_bytes:
                raise ValueError("Audio bytes are empty. Please ensure you recorded audio.")
            
            if len(audio_bytes) < MIN_AUDIO_SIZE_BYTES:
                raise ValueError(f"Audio file is too small ({len(audio_bytes)} bytes). Please record a longer audio clip.")
            
            # Verify it's a valid WAV file (check for WAV header: "RIFF" at start, "WAVE" at offset 8)
            if mime_type == "audio/wav":
                if len(audio_bytes) < WAV_HEADER_SIZE:
                    raise ValueError("Audio file is too small to be a valid WAV file.")
                if not (audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'):
                    logger.warning("Audio file may not be a valid WAV format, but attempting to process anyway.")
            
            logger.info(f"Processing audio: {len(audio_bytes)} bytes, MIME type: {mime_type}, RAG: {use_rag}")
            
            # Construct the parts: Prompt text first, then audio data
            # Some APIs expect the text instruction before the media
            parts = [
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ]
            
            # Create config with optional RAG
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
            
            # Note: For audio grading, RAG is typically disabled to avoid citation metadata
            # The model uses its knowledge + the scenario context provided in the prompt
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config=config
            )
            
            if response.text:
                # Filter out citation metadata before returning
                return filter_citation_metadata(response.text)
            
            if response.candidates:
                # Fallback logic for empty text
                if response.candidates[0].content.parts:
                    parts_text = [p.text for p in response.candidates[0].content.parts if p.text]
                    if parts_text:
                        # Filter out citation metadata before returning
                        return filter_citation_metadata("".join(parts_text))
                
                # Check for finish reasons
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

class DataManager:
    """Handles File I/O operations."""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def _load_full_csv(filepath: str) -> pd.DataFrame:
        """
        Load and cache the full CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame containing CSV data, or empty DataFrame if file doesn't exist
        """
        if not os.path.exists(filepath):
            return pd.DataFrame()
        try:
            return pd.read_csv(filepath)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"CSV loading error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS) 
    def load_csv_sample(filepath: str, selected_tags: List[str], sample_size: int) -> List[Dict[str, Any]]:
        """
        Load CSV sample filtered by tags.
        
        Args:
            filepath: Path to CSV file
            selected_tags: List of tags to filter by
            sample_size: Number of samples to return
            
        Returns:
            List of dictionaries representing sampled rows
        """
        df = DataManager._load_full_csv(filepath)
        if df.empty or "Meta_Tags" not in df.columns:
            return []
        
        def matches_tags(row_tags):
            if pd.isna(row_tags):
                return False
            row_tag_list = [t.strip() for t in str(row_tags).split(',')]
            return any(tag in row_tag_list for tag in selected_tags)

        filtered_df = df[df['Meta_Tags'].apply(matches_tags)]
        if filtered_df.empty:
            return []
        
        sample = filtered_df.sample(n=min(len(filtered_df), sample_size), random_state=42)
        return sample.to_dict(orient="records")

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
    def get_topics_from_csv(filepath: str) -> List[str]:
        """
        Extract unique topic tags from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Sorted list of unique topic tags
        """
        df = DataManager._load_full_csv(filepath)
        if df.empty or "Meta_Tags" not in df.columns:
            return ["General Rules"]
        
        try:
            all_tags = set()
            for tag_string in df["Meta_Tags"].dropna():
                tags = [tag.strip() for tag in str(tag_string).split(',')]
                all_tags.update(tags)
            return sorted(list(all_tags))
        except Exception as e:
            logger.warning(f"Error extracting topics: {e}")
            return ["General Rules"]

    @staticmethod
    @st.cache_data(ttl=BACKGROUND_IMAGE_CACHE_TTL)
    def get_base64_image(filepath: str) -> Optional[str]:
        """
        Load image file and convert to base64 string.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Base64-encoded string, or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Image loading error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading image: {e}")
            return None

# --- SERVICE LAYER (Business Logic) ---

def generate_with_retry(
    ai_provider: GenAIProvider,
    prompt: str,
    safe_prompt: str,
    error_context: str = "generation"
) -> str:
    """
    Shared retry logic for handling RECITATION_ERROR across all services.
    
    Args:
        ai_provider: GenAI provider instance
        prompt: Initial prompt
        safe_prompt: Paraphrased prompt for retry
        error_context: Context string for logging
        
    Returns:
        Generated text response
        
    Raises:
        ValueError: If RECITATION_ERROR occurs on both attempts
        RuntimeError: If other errors occur
    """
    try:
        logger.info(f"[{error_context}] Processing request: {len(prompt)} chars")
        return ai_provider.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True)
    except Exception as e:
        if "RECITATION_ERROR" in str(e):
            logger.warning(f"[{error_context}] Hit Recitation error. Retrying with safe prompt.")
            try:
                return ai_provider.generate_content(safe_prompt, SYSTEM_INSTRUCTION, use_rag=True)
            except Exception as e2:
                if "RECITATION_ERROR" in str(e2):
                    logger.error(f"[{error_context}] Safe prompt also hit RECITATION_ERROR.")
                    raise ValueError("RECITATION_ERROR")
                raise e2
        raise e

def read_audio_bytes(audio_input) -> Optional[bytes]:
    """
    Safely read audio bytes from Streamlit audio input.
    
    Args:
        audio_input: Streamlit audio input object
        
    Returns:
        Audio bytes or None if invalid
    """
    if not audio_input:
        return None
    
    try:
        if hasattr(audio_input, 'read'):
            if hasattr(audio_input, 'seek'):
                audio_input.seek(0)
            return audio_input.read()
        elif hasattr(audio_input, 'getvalue'):
            return audio_input.getvalue()
        else:
            return bytes(audio_input)
    except Exception as e:
        logger.error(f"Error reading audio: {e}")
        return None

def validate_audio_bytes(audio_bytes: Optional[bytes], min_size: int = MIN_AUDIO_SIZE_BYTES) -> tuple[bool, Optional[str]]:
    """
    Validate audio bytes.
    
    Args:
        audio_bytes: Audio bytes to validate
        min_size: Minimum size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not audio_bytes:
        return False, "No audio data received. Please record your answer again."
    
    if len(audio_bytes) < min_size:
        return False, f"Audio file is very small ({len(audio_bytes)} bytes). Please record a longer response."
    
    # Validate WAV header if needed
    if len(audio_bytes) >= WAV_HEADER_SIZE:
        if not (audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'):
            logger.warning("Audio file may not be a valid WAV format")
    
    return True, None

def validate_config() -> bool:
    """
    Validate that all required configuration is present.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        if not st.secrets.get(API_KEY_NAME):
            st.error("Missing API Key. Please check your secrets configuration.")
            return False
        if not st.secrets.get(STORE_ID_NAME):
            st.warning("Missing Store ID. RAG features will be disabled.")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def filter_citation_metadata(text: str) -> str:
    """
    Remove citation metadata (e.g., [cite: 1, 2, 3...]) from AI responses.
    Preserves all other formatting and content.
    
    Args:
        text: Text that may contain citation metadata
        
    Returns:
        Text with citation metadata removed, preserving formatting
    """
    if not text:
        return text
    
    # Remove citation patterns like [cite: 1, 2, 3] or [cite:1,2,3] or [cite:...]
    # Only match [cite: ...] patterns, be careful not to remove other bracket content
    # Use non-greedy matching to avoid removing too much
    citation_pattern = re.compile(r'\[cite:\s*[^\]]+\]', re.IGNORECASE)
    cleaned = citation_pattern.sub('', text)
    
    # Remove standalone citation lines (lines that are only citation metadata)
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip lines that are only citation metadata
        stripped = line.strip()
        if stripped and not re.match(r'^\[cite:\s*[^\]]+\]$', stripped, re.IGNORECASE):
            filtered_lines.append(line)
        elif not stripped:
            # Keep empty lines to preserve formatting
            filtered_lines.append(line)
    
    cleaned = '\n'.join(filtered_lines)
    
    # Clean up multiple consecutive blank lines (max 2)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # If the cleaned text is mostly just numbers and commas (citation metadata), return original
    # This handles cases where the entire response is citation metadata
    if re.match(r'^[\d,\s\[\]]+$', cleaned.strip()):
        logger.warning("Response appears to be only citation metadata, returning original")
        return text
    
    return cleaned

class JSONParser:
    """Parses JSON from AI responses with robust error handling."""
    
    @staticmethod
    def parse(text: Optional[str]) -> List[Dict[str, Any]]:
        """
        Parse JSON array from AI response text.
        
        Args:
            text: Raw AI response text (may contain markdown code blocks)
            
        Returns:
            List of dictionaries parsed from JSON
            
        Raises:
            ValueError: If text is empty or no valid JSON found
        """
        if not text:
            raise ValueError("AI response empty.")
        
        # Remove markdown code blocks
        clean_text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```', '', clean_text).strip()
        
        # Try to find JSON array
        start = clean_text.find('[')
        end = clean_text.rfind(']') + 1
        
        if start == -1 or end == 0:
            # Try to find JSON object and wrap in array
            obj_start = clean_text.find('{')
            obj_end = clean_text.rfind('}') + 1
            if obj_start != -1 and obj_end != 0:
                try:
                    obj = json.loads(clean_text[obj_start:obj_end])
                    return [obj]
                except json.JSONDecodeError:
                    pass
            raise ValueError("No JSON array or object found in response.")
        
        json_str = clean_text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}. Response preview: {json_str[:200]}")

class QuizService:
    """Service for generating quiz questions using AI."""
    
    def __init__(self, ai_provider: GenAIProvider):
        """
        Initialize quiz service.
        
        Args:
            ai_provider: GenAI provider instance
        """
        self.ai = ai_provider

    def calculate_topic_distribution(self, request: QuizRequest) -> Dict[str, int]:
        """
        Calculate question distribution across topics.
        
        Args:
            request: QuizRequest with topics and question count
            
        Returns:
            Dictionary mapping topic names to question counts
        """
        distribution = {}
        total = request.num_questions
        active = [t for t in request.topics if t.percentage > 0]
        
        if not active:
            return distribution
        
        # Calculate base count per percentage point
        assigned = 0
        for i, topic in enumerate(active[:-1]):  # All but last
            count = (total * topic.percentage) // 100
            if count > 0:
                distribution[topic.topic_name] = count
                assigned += count
        
        # Last topic gets remainder
        if active:
            distribution[active[-1].topic_name] = total - assigned
        
        return distribution

    def _generate_with_retry(
        self, 
        prompt: str, 
        safe_prompt: str,
        error_context: str = "generation"
    ) -> str:
        """
        Generate content with automatic retry on RECITATION_ERROR.
        
        Args:
            prompt: Initial prompt
            safe_prompt: Paraphrased prompt for retry
            error_context: Context string for logging
            
        Returns:
            Generated text response
        """
        return generate_with_retry(self.ai, prompt, safe_prompt, error_context)

    def generate_from_ai(self, request: QuizRequest) -> List[QuizQuestion]:
        """
        Generate quiz questions using AI with RAG.
        
        Args:
            request: QuizRequest containing number of questions and topic distribution
            
        Returns:
            List of QuizQuestion objects
            
        Raises:
            ValueError: If AI generation fails or returns invalid JSON
            RuntimeError: If API call fails
        """
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} questions about: {t}" for t, c in dist.items()])
        
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
                "explanation": "Concise explanation."
            }}
        ]
        """
        
        prompt_safe = f"""
        Create a {request.num_questions} question multiple-choice quiz based on the provided File Store.
        DISTRIBUTION: {dist_str}
        
        CRITICAL INSTRUCTION: The previous attempt was blocked for quoting the text too closely.
        1. Rephrase the questions and options so they are distinct from the source text (Paraphrase).
        2. Test the concept/rule, do not just ask for the definition.
        3. Keep explanations concise and synthesized.
        
        OUTPUT SCHEMA (JSON Array): 
        [
            {{
                "question": "Question text (Paraphrased)...", 
                "options": ["A) ...", "B) ..."], 
                "correct_answer": "Full text of option", 
                "reference": "Rule #", 
                "explanation": "Synthesized explanation."
            }}
        ]
        """
        
        raw = self._generate_with_retry(prompt, prompt_safe, "Quiz generation")
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]
    
    def generate_scenarios(self, request: QuizRequest) -> List[QuizQuestion]:
        """
        Generate navigational scenario quiz questions.
        
        Args:
            request: QuizRequest containing number of questions and topic distribution
            
        Returns:
            List of QuizQuestion objects with scenario-based questions
            
        Raises:
            ValueError: If AI generation fails or returns invalid JSON
            RuntimeError: If API call fails
        """
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} scenarios involving: {t}" for t, c in dist.items()])
        
        prompt = f"""
        Create a {request.num_questions} question *Navigational Scenario Quiz* based ONLY on the provided File Store.
        
        DISTRIBUTION:
        {dist_str}
        
        INSTRUCTIONS:
        1. **Do not ask simple definitions.**
        2. **Create vivid scenarios:** Describe what the Officer of the Watch sees (lights, day shapes, sound signals) or the radar plot.
        3. **Bridge Perspective:** Phrase questions as "You are a power-driven vessel steering 090. You see..."
        4. **Application:** Ask for the required action or identification of status.
        
        OUTPUT SCHEMA (JSON Array): 
        [
            {{
                "question": "Detailed scenario text...", 
                "options": ["A) ...", "B) ..."], 
                "correct_answer": "Full text of correct option", 
                "reference": "Rule #", 
                "explanation": "Concise explanation citing the rule."
            }}
        ]
        """
        
        prompt_safe = f"""
        Create a {request.num_questions} question *Navigational Scenario Quiz* based on the provided File Store.
        DISTRIBUTION: {dist_str}
        
        CRITICAL INSTRUCTION: Do NOT copy scenario text from the books. Invent NEW scenarios that apply the same rules.
        1. Paraphrase all descriptions.
        2. Use different vessel names or slight variations in bearing/range to ensure uniqueness.
        
        OUTPUT SCHEMA (JSON Array): 
        [
            {{
                "question": "Detailed scenario text (Unique)...", 
                "options": ["A) ...", "B) ..."], 
                "correct_answer": "Full text of correct option", 
                "reference": "Rule #", 
                "explanation": "Concise explanation citing the rule."
            }}
        ]
        """
        
        raw = self._generate_with_retry(prompt, prompt_safe, "Scenario generation")
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

    def generate_from_csv(self, raw_data: List[Dict], count: int) -> List[QuizQuestion]:
        """
        Generate quiz questions from CSV data.
        
        Args:
            raw_data: List of dictionaries from CSV rows
            count: Number of questions to generate
            
        Returns:
            List of QuizQuestion objects
        """
        prompt = f"""
        Convert this CSV data to JSON quiz. Use RAG for explanation/reference if missing.
        Data: {json.dumps(raw_data)}
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        
        prompt_safe = f"""
        Convert this CSV data to JSON quiz. Use RAG for explanation/reference if missing.
        Data: {json.dumps(raw_data)}
        
        CRITICAL INSTRUCTION: Paraphrase questions and options to avoid direct copying from source.
        Create unique questions that test the same concepts but use different wording.
        
        OUTPUT SCHEMA (JSON Array): [{{question, options[], correct_answer, reference, explanation}}]
        """
        
        raw = self._generate_with_retry(prompt, prompt_safe, "CSV quiz generation")
        return [QuizQuestion(**i) for i in JSONParser.parse(raw)]

class RemediationService:
    """Generates lesson plans with batch processing for scalability."""
    
    def __init__(self, ai_provider: GenAIProvider):
        """
        Initialize remediation service.
        
        Args:
            ai_provider: GenAI provider instance
        """
        self.ai = ai_provider

    def _generate_with_retry(self, prompt: str, safe_prompt: str) -> str:
        """Helper method to retry generation with a safe prompt if RECITATION_ERROR occurs."""
        return generate_with_retry(self.ai, prompt, safe_prompt, "Remediation batch generation")

    def _generate_batch_content(self, batch_refs: List[str]) -> str:
        """
        Generate remediation content for a batch of rule references.
        
        Args:
            batch_refs: List of rule references to process
            
        Returns:
            Markdown-formatted remediation content
        """
        topics_context = ", ".join(batch_refs)
        
        prompt = f"""
        Create a "Deep Dive" remediation section for these specific Maritime Rules: {topics_context}.
        Based ONLY on the provided File Store documents.
        
        For each topic:
        1. **Header**: Rule Number/Name.
        2. **Official Rule Text**: Provide the exact verbatim text of the key sentence (1-3 sentences max).
        3. **Explanation**: Plain English summary.
        4. **Common Pitfalls**: Why students miss this.
        5. **Memory Aid**: Mnemonic or trick.
        
        OUTPUT FORMAT: Markdown. Do not include intro or outro.
        """
        
        prompt_safe = f"""
        Create a "Deep Dive" remediation section for these specific Maritime Rules: {topics_context}.
        
        CRITICAL: Paraphrase all rule text. Do not quote source text directly. Create original summaries.
        
        For each topic:
        1. **Header**: Rule Number/Name.
        2. **Rule Summary**: Summarize the rule concisely (Cite Rule #). Paraphrase - do not quote verbatim.
        3. **Explanation**: Plain English summary.
        4. **Common Pitfalls**: Why students miss this.
        5. **Memory Aid**: Mnemonic or trick.
        
        OUTPUT FORMAT: Markdown. Do not include intro or outro.
        """
        
        try:
            return self._generate_with_retry(prompt, prompt_safe)
        except Exception as e:
            logger.error(f"Error generating batch content for {topics_context}: {e}")
            return f"**Error generating content for {topics_context}: {str(e)}**"

    def generate_lesson(self, incorrect_questions: List[QuizQuestion]) -> str:
        """
        Generate a personalized remediation lesson plan.
        
        Args:
            incorrect_questions: List of questions the user got wrong
            
        Returns:
            Markdown-formatted lesson plan text
        """
        unique_refs = sorted({q.reference for q in incorrect_questions if q.reference})
        
        if not unique_refs:
            return "No specific rules identified for remediation."

        full_doc = [
            "# Remediation Lesson Plan",
            f"**Focus Topics:** {', '.join(unique_refs)}",
            "---"
        ]

        batches = [unique_refs[i:i+REMEDIATION_BATCH_SIZE] 
                   for i in range(0, len(unique_refs), REMEDIATION_BATCH_SIZE)]
        
        # Parallel processing with progress tracking
        progress_bar = st.progress(0) if IN_STREAMLIT else None
        status_text = st.empty() if IN_STREAMLIT else None
        
        try:
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(self._generate_batch_content, batch): batch 
                    for batch in batches
                }
                
                batch_results = []
                completed = 0
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_content = future.result()
                        batch_results.append((batch, batch_content))
                        completed += 1
                        
                        if progress_bar and status_text:
                            status_text.text(f"Processing batch {completed}/{len(batches)}: {', '.join(batch)}")
                            progress_bar.progress(completed / len(batches))
                    except Exception as e:
                        logger.error(f"Error processing batch {batch}: {e}")
                        batch_results.append((batch, f"**Error processing batch: {str(e)}**"))
            
            # Sort results to maintain order
            batch_results.sort(key=lambda x: unique_refs.index(x[0][0]) if x[0] and x[0][0] in unique_refs else 0)
            
            for batch, batch_content in batch_results:
                full_doc.append(batch_content)
                full_doc.append("\n---\n")
        finally:
            if progress_bar:
                progress_bar.empty()
            if status_text:
                status_text.empty()

        full_doc.append("## Recommended Study Plan")
        full_doc.append("1. **Review** the specific rule citations above in your official handbook.")
        full_doc.append("2. **Practice** these topics using the 'Remediation Quiz' button below.")
        full_doc.append("3. **Visualize** the scenarios described in the 'Common Pitfalls' section.")

        return "\n\n".join(full_doc)

class OralBoardService:
    """Handles Oral Examination logic (Audio grading)."""
    
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def _generate_with_retry(self, prompt: str, safe_prompt: str) -> str:
        """Helper method to retry generation with a safe prompt if RECITATION_ERROR occurs."""
        return generate_with_retry(self.ai, prompt, safe_prompt, "Oral Board scenario generation")

    def generate_scenario(self) -> str:
        """Generates a complex oral exam scenario."""
        prompt = """
        Generate a specific, complex 'Oral Board' scenario for a Maritime Officer candidate.
        Subject: Rules of the Road / Collision Avoidance.
        
        Format the output clearly:
        **Scenario:** [Describe vessel status, visibility, other vessels, radar info, sound signals]
        **Question:** [What is the specific situation and what is your required action?]
        """
        
        prompt_safe = """
        Generate a specific, complex 'Oral Board' scenario for a Maritime Officer candidate.
        Subject: Rules of the Road / Collision Avoidance.
        
        CRITICAL: Paraphrase all content. Do not quote source text directly. Create original scenario descriptions.
        
        Format the output clearly:
        **Scenario:** [Describe vessel status, visibility, other vessels, radar info, sound signals]
        **Question:** [What is the specific situation and what is your required action?]
        """
        
        return self._generate_with_retry(prompt, prompt_safe)

    def grade_response(self, scenario: str, audio_bytes: bytes) -> str:
        """Grades the audio response against the scenario."""
        prompt = f"""
        You are a USCG Licensing Examiner.
        
        **SCENARIO GIVEN TO CANDIDATE:**
        {scenario}
        
        **TASK:**
        Listen to the candidate's audio response provided in the attachment.
        
        **OUTPUT FORMAT:**
        1. **Transcript:** Write down exactly what the candidate said.
        2. **Grade:** PASS or FAIL.
        3. **Critique:**
           - Did they identify the correct Rule?
           - Did they state the correct action?
           - Was the answer delivered confidently?
        4. **Correct Answer:** What was the expected response?
        """
        
        response = self.ai.generate_content_with_audio(
            prompt=prompt, 
            audio_bytes=audio_bytes, 
            mime_type="audio/wav", # Streamlit audio recorder output
            system_instr=SYSTEM_INSTRUCTION,
            use_rag=True  # Enable RAG for proper maritime rules context
        )
        
        # Filter out citation metadata but preserve all formatting and content
        cleaned = filter_citation_metadata(response)
        
        # Ensure we have valid content - if filtering removed everything meaningful, return original
        if not cleaned or (len(cleaned.strip()) < 50 and len(response.strip()) > 100):
            logger.warning("Citation filter may have been too aggressive, using original response")
            return response
        
        return cleaned

# --- PRESENTATION LAYER ---

def init_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        "messages": [{"role": "assistant", "content": "👋 Maritime Expert Ready."}],
        "quiz_data": None,
        "current_quiz_source": None,
        "incorrect_questions": [],
        "remediation_text": None,
        "oral_scenario": None,  # For Oral Board
        "oral_grade": None      # For Oral Board
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def set_background():
    """Set background image for the Streamlit app."""
    bin_str = DataManager.get_base64_image('background.png')
    if bin_str:
        st.markdown(f'''<style>
        .stApp {{ background-image: url("data:image/png;base64,{bin_str}"); background-size: cover; }}
        </style>''', unsafe_allow_html=True)

def render_oral_board(oral_service: OralBoardService):
    st.title("🎙️ Oral Board Simulator")
    st.caption("Simulate a USCG Licensing Exam. Speak your answer clearly.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔄 Generate New Scenario", type="primary"):
            with st.spinner("Examiner is preparing a scenario..."):
                scenario = oral_service.generate_scenario()
                st.session_state.oral_scenario = scenario
                st.session_state.oral_grade = None # Reset previous grade
                st.rerun()

    if st.session_state.oral_scenario:
        st.markdown("### 📋 Examiner's Scenario")
        st.info(st.session_state.oral_scenario)
        
        st.markdown("### 🗣️ Your Response")
        audio_input = st.audio_input("Record your answer", key="audio_recorder")
        
        if audio_input:
            # Check if we already graded this specific audio (prevent re-run on simple interaction)
            if not st.session_state.oral_grade:
                with st.spinner("Examiner is listening and grading..."):
                    try:
                        # Read audio bytes using helper function
                        audio_bytes = read_audio_bytes(audio_input)
                        
                        if audio_bytes:
                            logger.info(f"Audio input type: {type(audio_input)}, size: {len(audio_bytes)} bytes")
                        
                        # Validate audio bytes
                        is_valid, error_msg = validate_audio_bytes(audio_bytes)
                        
                        if not is_valid:
                            st.error(f"❌ {error_msg}")
                        else:
                            logger.info(f"Processing audio response: {len(audio_bytes)} bytes, first 20 bytes: {audio_bytes[:20]}")
                            grade = oral_service.grade_response(st.session_state.oral_scenario, audio_bytes)
                            st.session_state.oral_grade = grade
                    except ValueError as e:
                        if "RECITATION_ERROR" in str(e):
                            st.warning("⚠️ The examiner encountered an issue. Please try recording your answer again.")
                            logger.warning("RECITATION_ERROR in audio grading")
                        else:
                            st.error(f"❌ Validation error: {e}")
                    except Exception as e:
                        st.error(f"❌ We encountered an issue while grading your response. Please try again or contact support if the problem persists.")
                        logger.error(f"Audio grading exception: {e}", exc_info=True)
            
            if st.session_state.oral_grade:
                st.divider()
                st.subheader("👨‍✈️ Examiner's Evaluation")
                st.markdown(st.session_state.oral_grade)

def main():
    """Main application entry point."""
    st.set_page_config(page_title="RAG Engine", page_icon="⚓", layout="wide")
    set_background()
    init_session_state()
    
    # Validate configuration
    if not validate_config():
        st.stop()
    
    try:
        ai_provider = GenAIProvider(st.secrets[API_KEY_NAME], st.secrets[STORE_ID_NAME])
        quiz_service = QuizService(ai_provider)
        remediation_service = RemediationService(ai_provider)
        oral_service = OralBoardService(ai_provider)
    except KeyError:
        st.error("Missing required secrets. Please check your configuration.")
        st.stop()

    with st.sidebar:
        st.title("⚓ Maritime AI")
        mode = st.radio("Tool:", ["Chat 🤖", "Quiz 📝", "Oral Board 🎙️"])
        if st.button("Reset Session"):
            st.session_state.messages = []
            st.session_state.quiz_data = None
            st.session_state.incorrect_questions = []
            st.session_state.remediation_text = None
            st.session_state.oral_scenario = None
            st.session_state.oral_grade = None
            st.rerun()
        st.divider()
        st.caption("✅ v16.0 - Oral Board Active") 

    if mode == "Chat 🤖":
        st.title("Regulations Chat")
        for m in st.session_state.messages:
            st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": p})
            st.chat_message("user").write(p)
            with st.spinner("Thinking..."):
                try:
                    r = ai_provider.generate_content(p, SYSTEM_INSTRUCTION)
                    st.session_state.messages.append({"role": "assistant", "content": r})
                    st.chat_message("assistant").write(r)
                except Exception as e:
                    error_msg = "We encountered an issue processing your request. Please try again or contact support if the problem persists."
                    logger.error(f"Chat error: {e}", exc_info=True)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.chat_message("assistant").write(error_msg)

    elif mode == "Quiz 📝":
        st.title("Knowledge Check")
        
        def clear(): 
            """Clear quiz-related session state."""
            st.session_state.quiz_data = None
            st.session_state.incorrect_questions = []
            st.session_state.remediation_text = None

        source = st.radio("Source:", ["CSV", "AI Gen"], horizontal=True, on_change=clear, key="source_radio")
        
        if source == "CSV":
            tags = st.multiselect("Topics:", DataManager.get_topics_from_csv("ror_test.csv"))
            n = st.number_input("Count:", MIN_QUIZ_QUESTIONS, MAX_QUIZ_QUESTIONS, DEFAULT_QUIZ_COUNT)
            if st.button("Load CSV") and tags:
                try:
                    data = DataManager.load_csv_sample("ror_test.csv", tags, n)
                    if data:
                        with st.spinner("Generating quiz from CSV..."):
                            st.session_state.quiz_data = quiz_service.generate_from_csv(data, n)
                            st.session_state.current_quiz_source = "CSV"
                            st.rerun()
                    else:
                        st.warning("No data found for selected tags.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            quiz_style = st.radio("Quiz Style:", ["Standard Rules", "Applied Scenarios"], horizontal=True)
            st.divider()
            
            topics = []
            for i in range(1, 5):
                c1, c2 = st.columns([3, 1])
                t = c1.text_input(f"Topic {i}")
                p = c2.number_input(f"%", 0, 100, 0, key=f"p{i}")
                if t:
                    topics.append(TopicDistribution(topic_name=t, percentage=p))
            
            n = st.number_input("Count", MIN_QUIZ_QUESTIONS, MAX_QUIZ_QUESTIONS, DEFAULT_QUIZ_COUNT)
            
            current_total = sum(t.percentage for t in topics)
            st.progress(min(current_total, 100) / 100)
            if current_total == 100:
                st.success(f"Total: {current_total}% (Valid)")
            else:
                st.warning(f"Total: {current_total}% (Target: 100%)")

            if st.button(f"Generate {quiz_style}"):
                st.info("Generating Quiz... Please wait.")
                try:
                    req = QuizRequest(num_questions=n, topics=topics)
                    with st.spinner("Processing Documents..."):
                        if quiz_style == "Standard Rules":
                            st.session_state.quiz_data = quiz_service.generate_from_ai(req)
                            st.session_state.current_quiz_source = "AI (Standard)"
                        else:
                            st.session_state.quiz_data = quiz_service.generate_scenarios(req)
                            st.session_state.current_quiz_source = "AI (Scenarios)"
                        st.rerun()
                except ValueError as e:
                    # Fallback caught in Service, this catches irrecoverable errors
                    if "RECITATION_ERROR" in str(e):
                        st.error("The AI failed to paraphrase after retrying. Please select different topics.")
                    else:
                        st.error(f"Error: {e}")
                except Exception as e: 
                    st.error(f"Error: {e}")

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
                
                if st.form_submit_button("Submit"):
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
                            current_incorrect.append(q)
                            
                        with st.expander("Explanation"):
                            st.info(f"{q.explanation} ({q.reference})")
                    
                    st.session_state.incorrect_questions = current_incorrect
                    st.session_state.remediation_text = None
                    pct = (score/len(questions))*100
                    st.metric("Score", f"{score}/{len(questions)} ({pct:.0f}%)")
            
            if st.session_state.incorrect_questions:
                st.divider()
                st.subheader("🎓 Remediation Zone")
                st.warning(f"You missed {len(st.session_state.incorrect_questions)} questions. Let's create a personalized lesson plan.")
                
                if st.button("Generate Remediation Lesson"):
                    with st.spinner("Analyzing weak points and generating lesson..."):
                        try:
                            lesson = remediation_service.generate_lesson(st.session_state.incorrect_questions)
                            st.session_state.remediation_text = lesson
                        except Exception as e:
                            st.error("We encountered an issue generating the remediation lesson. Please try again or contact support if the problem persists.")
                            logger.error(f"Remediation lesson generation error: {e}", exc_info=True)
                
                if st.session_state.remediation_text:
                    st.markdown("---")
                    st.markdown(st.session_state.remediation_text)
                    
                    if HAS_REPORTLAB:
                        pdf_data = PDFGenerator.create_study_guide(st.session_state.remediation_text)
                        if pdf_data:
                            st.download_button(
                                label="📄 Download Study Guide (PDF)",
                                data=pdf_data,
                                file_name="remediation_plan.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.caption("Install 'reportlab' to enable PDF export.")

                    st.divider()
                    st.subheader("📝 Practice Makes Perfect")
                    
                    missed_refs = sorted({q.reference for q in st.session_state.incorrect_questions if q.reference})
                    
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.info(f"Target Topics: {', '.join(missed_refs)}" if missed_refs else "Target Topics: General Review")
                    with c2:
                        rem_n = st.number_input("Questions:", REMEDIATION_QUIZ_MIN, REMEDIATION_QUIZ_MAX, DEFAULT_QUIZ_COUNT, key="rem_n")
                    
                    rem_style = st.radio("Remediation Style:", ["Standard Rules", "Applied Scenarios"], horizontal=True, key="rem_style")

                    if st.button("Start Remediation Quiz"):
                        try:
                            topics = []
                            if missed_refs:
                                count = len(missed_refs)
                                base_pct = 100 // count
                                remainder = 100 % count
                                for i, ref in enumerate(missed_refs):
                                    pct = base_pct + (1 if i < remainder else 0)
                                    topics.append(TopicDistribution(topic_name=ref, percentage=pct))
                            else:
                                topics.append(TopicDistribution(topic_name="General Maritime Rules", percentage=100))
                            
                            req = QuizRequest(num_questions=rem_n, topics=topics)
                            
                            with st.spinner("Generating targeted practice quiz..."):
                                if rem_style == "Standard Rules":
                                    new_quiz = quiz_service.generate_from_ai(req)
                                else:
                                    new_quiz = quiz_service.generate_scenarios(req)
                                
                                st.session_state.quiz_data = new_quiz
                                st.session_state.current_quiz_source = f"Remediation ({rem_style})"
                                st.session_state.incorrect_questions = []
                                st.session_state.remediation_text = None
                                st.rerun()
                                
                        except ValueError as e:
                            if "RECITATION_ERROR" in str(e):
                                st.error("The AI attempted to copy too much text. Please try generating the quiz again.")
                            else:
                                st.error(f"Error: {e}")
                        except Exception as e:
                            st.error("We encountered an issue generating the remediation quiz. Please try again or contact support if the problem persists.")
                            logger.error(f"Remediation quiz generation error: {e}", exc_info=True)

            with st.expander("Admin Export"):
                st.download_button("Download JSON", json.dumps([q.model_dump() for q in questions], indent=2), "quiz.json")

    elif mode == "Oral Board 🎙️":
        render_oral_board(oral_service)

if __name__ == "__main__":
    main()
