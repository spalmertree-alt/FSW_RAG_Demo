import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os
import json
import base64
import re
import logging
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from io import BytesIO
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from functools import wraps

# --- AUDIO RECORDER (Voice Chat) ---
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False
    audio_recorder = None

# --- DRAWABLE CANVAS (Visual Quiz) ---
from drawing_canvas import drawing_canvas

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
STORE_ID_NAME = "STORE_ID"       # File store key in secrets (13 installation PDFs)
ADMIN_PASSWORD_NAME = "ADMIN_PASSWORD"
QUERY_LOG_FILE = os.path.join(os.path.dirname(__file__), "query_log.jsonl")
HIGH_INTEREST_THRESHOLD = 3   # queries on same topic to flag as High Interest
KNOWLEDGE_GAP_THRESHOLD = 2   # not-found responses to flag as Knowledge Gap

# Model Configuration - Easy switching between models
MODEL_FLASH = "gemini-2.5-flash"  # Fast, cost-effective for most tasks
MODEL_PRO = "gemini-2.5-pro"      # Slower, better for complex reasoning

# Default model for general use (recommended: flash for speed)
MODEL_ID = MODEL_FLASH

# Task-specific model overrides (set to None to use default MODEL_ID)
MODEL_FOR_CHAT = None           # Uses MODEL_ID (flash recommended)
MODEL_FOR_QUIZ = None           # Uses MODEL_ID (flash recommended)
MODEL_FOR_GRADING = MODEL_PRO   # Pro for nuanced grading analysis
MODEL_FOR_ORAL = None           # Uses MODEL_ID (flash recommended)
MODEL_FOR_VISUAL = None         # Uses MODEL_ID (flash for speed)
MODEL_FOR_VOICE = None          # Uses MODEL_ID (flash for voice chat)

# Difficulty Levels for AI-Generated Quizzes
DIFFICULTY_EASY = "Easy"
DIFFICULTY_MEDIUM = "Medium"
DIFFICULTY_HARD = "Hard"
DIFFICULTY_LEVELS = [DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD]

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
RESPONSE_CACHE_TTL = 1800  # 30 minutes for response caching
ENABLE_STREAMING = True  # Enable streaming for chat mode
MAX_RETRY_ATTEMPTS = 3  # Max retries for rate limiting
RETRY_BASE_DELAY = 2  # Base delay in seconds for exponential backoff
MIN_ANSWER_LENGTH = 10  # Minimum characters for free response answers

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
You are a network infrastructure installation and administration expert.
Your knowledge base covers Aruba/HPE switches (CX 8325 series), HPE EdgeConnect SD-WAN,
Orchestrator, Palo Alto Networks firewalls (PA-1400, PAN-OS, NGFW), and related error references.
CRITICAL: You MUST use the File Search tool to search ALL documents in the knowledge base.

RULES FOR OPERATION:
1. Search across ALL documents for every question.
2. Do NOT answer from general knowledge. ONLY use information returned by the File Search tool.
3. If the File Search tool returns NO relevant results, respond:
   "I'm sorry, I could not find information about that topic in the current manuals. Please try rephrasing your question."
   Do NOT attempt to answer from your own knowledge.
4. MANDATORY SOURCE CITATION: Every answer MUST end with a source line in this exact format:
   **Source: [document name], Page [page number]**
   If multiple sources are used, list each one. Never omit this line.
5. Reproduce technical procedures, CLI commands, warnings, and specifications verbatim from the source document.
6. When synthesizing information across multiple documents, clearly attribute each source inline.
7. Do not add any information that is not present in the retrieved documents.
8. Use proper networking terminology and abbreviations (e.g., VLAN, OSPF, SD-WAN, NGFW, HA).
9. You have full memory of this conversation. Use prior messages as context when answering follow-up questions.
   If asked "what document is that from?" or similar, refer back to the source you cited in your previous answer.
"""

# Open LLM mode system instructions (used when RAG is disabled via Master Switch)
SYSTEM_INSTRUCTION_OPEN = """
You are a network infrastructure installation and administration expert.
You have full access to your training knowledge on Aruba/HPE switches, EdgeConnect SD-WAN,
Palo Alto Networks firewalls, and general network infrastructure.

RULES FOR OPERATION:
1. Use your full knowledge base to answer networking and installation questions.
2. You may draw on general networking knowledge, vendor documentation, and best practices.
3. Always cite specific manual names, sections, or references when applicable.
4. Provide thorough, accurate answers using your complete training data.
5. If a question goes beyond the specific manuals, you may provide relevant networking context.
6. You are NOT limited to a specific document set - use all available knowledge.
7. Use proper networking terminology and abbreviations.
"""

VOICE_CHAT_INSTRUCTION = ""
VOICE_CHAT_INSTRUCTION_OPEN = ""


# --- COURSE CONFIGURATION ---

class CourseConfig:
    """Bundles all course-specific settings so the app can switch subjects."""

    def __init__(
        self,
        key: str,
        name: str,
        icon: str,
        store_id_secret: str,
        system_instruction: str,
        system_instruction_open: str,
        voice_instruction: str,
        voice_instruction_open: str,
        mc_csv: str,
        frq_csv: str,
        visual_csv: str,
        background_image: str,
        rule_knowledge_areas: Dict[str, Tuple[str, str]],
        subject_label: str,
        examiner_persona: str,
        oral_subject: str,
        oral_board_caption: str = "Speak your answer clearly.",
        voice_placeholder: str = "Ask a question...",
        visual_caption: str = "Draw on the image to answer the question.",
        grading_synonym_hint: str = "",
        oral_grading_concept: str = "key concept or rule",
    ):
        self.key = key
        self.name = name
        self.icon = icon
        self.store_id_secret = store_id_secret
        self.system_instruction = system_instruction
        self.system_instruction_open = system_instruction_open
        self.voice_instruction = voice_instruction
        self.voice_instruction_open = voice_instruction_open
        self.mc_csv = mc_csv
        self.frq_csv = frq_csv
        self.visual_csv = visual_csv
        self.background_image = background_image
        self.rule_knowledge_areas = rule_knowledge_areas
        self.subject_label = subject_label          # e.g. "Network Infrastructure Installation"
        self.examiner_persona = examiner_persona    # e.g. "Network Infrastructure Expert"
        self.oral_subject = oral_subject            # e.g. "Network Installation & Administration"
        self.oral_board_caption = oral_board_caption
        self.voice_placeholder = voice_placeholder
        self.visual_caption = visual_caption
        self.grading_synonym_hint = grading_synonym_hint
        self.oral_grading_concept = oral_grading_concept


# Installation Assistant course ------------------------------------------------
INSTALLATION_KNOWLEDGE_AREAS: Dict[str, Tuple[str, str]] = {
    # Aruba / HPE Networking
    "aruba_8325_igsg": ("Aruba 8325 IGSG", "Aruba Switches"),
    "aruba_8325h_igsg": ("Aruba 8325H IGSG", "Aruba Switches"),
    "hpe_cx8325": ("HPE Aruba CX 8325 Switch Series", "Aruba Switches"),
    "hpe_edgeconnect_sdwan": ("HPE Aruba EdgeConnect SD-WAN QuickSpecs", "Aruba SD-WAN"),
    # EdgeConnect / Orchestrator
    "ec_10108_install": ("EC-10108 Install Guide", "EdgeConnect"),
    "ec_10108_startup": ("EC-10108 StartUp Guide", "EdgeConnect"),
    "orch_r960": ("Orchestrator User Guide R960", "EdgeConnect"),
    "xr5610": ("XR5610 Operations Manual", "EdgeConnect"),
    # Palo Alto Networks
    "pa_1400_hw_ref": ("PA-1400 Hardware Reference", "Palo Alto"),
    "pa_1400_series": ("PA-1400 Series", "Palo Alto"),
    "pan_os_admin": ("PAN-OS Administration", "Palo Alto"),
    "ngfw_admin": ("NGFW Administration", "Palo Alto"),
    # General
    "error_event_msg": ("Error & Event Message Reference", "Troubleshooting"),
}

COURSE_INSTALLATION = CourseConfig(
    key="installation",
    name="Network Installation Assistant",
    icon="🔧",
    store_id_secret=STORE_ID_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
    system_instruction_open=SYSTEM_INSTRUCTION_OPEN,
    voice_instruction=VOICE_CHAT_INSTRUCTION,
    voice_instruction_open=VOICE_CHAT_INSTRUCTION_OPEN,
    mc_csv="",
    frq_csv="",
    visual_csv="",
    background_image="background.png",
    rule_knowledge_areas=INSTALLATION_KNOWLEDGE_AREAS,
    subject_label="Network Infrastructure Installation",
    examiner_persona="Network Infrastructure Expert",
    oral_subject="Network Installation & Administration",
    oral_board_caption="",
    voice_placeholder="Ask about installation, configuration, or troubleshooting...",
    visual_caption="",
    grading_synonym_hint='Accept valid networking synonyms (e.g. "firewall rule" for "security policy" is acceptable if the concept is correct).',
    oral_grading_concept="procedure, configuration, or concept",
)

# Course registry
COURSES: Dict[str, CourseConfig] = {
    "installation": COURSE_INSTALLATION,
}

DEFAULT_COURSE_KEY = "installation"


def get_active_course() -> CourseConfig:
    """Return the currently selected course config."""
    key = st.session_state.get("active_course", DEFAULT_COURSE_KEY)
    return COURSES.get(key, COURSES[DEFAULT_COURSE_KEY])


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
    difficulty: str = Field(default=DIFFICULTY_MEDIUM, description="Target difficulty level")

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, difficulty: str) -> str:
        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty level. Must be one of: {DIFFICULTY_LEVELS}")
        return difficulty

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
    """Represents a visual identification question."""
    id: str
    question_text: str
    image_path: str
    rubric_path: str


class TopicPerformance(BaseModel):
    """Tracks performance for a specific topic (Weak Topic Identification)."""
    topic: str
    total_questions: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    last_attempt: Optional[datetime] = None

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return (self.correct_answers / self.total_questions) * 100

    @property
    def is_weak(self) -> bool:
        """Topic is weak if accuracy < 70% with at least 3 attempts."""
        return self.total_questions >= 3 and self.accuracy < 70

    @property
    def needs_more_data(self) -> bool:
        """Topic needs more data if < 3 attempts and accuracy is below 70%."""
        return self.total_questions < 3 and self.accuracy < 70 and self.total_questions > 0

    @property
    def is_strong(self) -> bool:
        """Topic is strong if accuracy is 100% with at least 3 attempts."""
        return self.total_questions >= 3 and self.accuracy == 100


# --- ANALYTICS SERVICE (Weak Topic Identification) ---

# Legacy alias kept for any external references; code now uses course config
RULE_KNOWLEDGE_AREAS = INSTALLATION_KNOWLEDGE_AREAS


class AnalyticsService:
    """Manages topic performance tracking and weak topic identification.

    All data is scoped per-course so switching courses preserves each course's analytics.
    """

    # Regex pattern to extract rule numbers from references
    _RULE_PATTERN = re.compile(r'[Rr]ule\s*(\d+)', re.IGNORECASE)
    _ANNEX_PATTERN = re.compile(r'[Aa]nnex\s*([IViv]+)', re.IGNORECASE)

    @staticmethod
    def _perf_key() -> str:
        """Session-state key scoped to the active course."""
        course = get_active_course()
        return f"topic_performance_{course.key}"

    @staticmethod
    def _get_knowledge_areas() -> Dict[str, Tuple[str, str]]:
        """Return the knowledge-area mapping for the active course."""
        return get_active_course().rule_knowledge_areas

    @staticmethod
    def extract_topic_from_reference(reference: str) -> Tuple[str, str]:
        """Extract a meaningful topic name and knowledge area from a rule reference."""
        knowledge_areas = AnalyticsService._get_knowledge_areas()
        course = get_active_course()

        if not reference:
            return (f"General {course.subject_label}", "General")

        # Try to match Annex first
        annex_match = AnalyticsService._ANNEX_PATTERN.search(reference)
        if annex_match:
            annex_num = annex_match.group(1).lower()
            key = f"annex {annex_num}"
            if key in knowledge_areas:
                return knowledge_areas[key]
            return (f"Annex {annex_num.upper()}", "Annexes")

        # Try to match Rule number
        rule_match = AnalyticsService._RULE_PATTERN.search(reference)
        if rule_match:
            rule_num = rule_match.group(1)
            if rule_num in knowledge_areas:
                return knowledge_areas[rule_num]
            return (f"Rule {rule_num}", "Other Rules")

        # Check all knowledge-area keys for a direct hit (supports non-numeric keys)
        ref_lower = reference.lower().strip()
        for ka_key, (topic_name, area) in knowledge_areas.items():
            if ka_key in ref_lower or ref_lower in ka_key:
                return (topic_name, area)

        # Fallback
        clean_ref = reference.strip()
        if len(clean_ref) > 50:
            clean_ref = clean_ref[:47] + "..."
        return (clean_ref, "Other")

    @staticmethod
    def get_topic_performance() -> Dict[str, TopicPerformance]:
        """Get or initialize topic performance tracking (scoped to active course)."""
        key = AnalyticsService._perf_key()
        if key not in st.session_state:
            st.session_state[key] = {}
        return st.session_state[key]

    @staticmethod
    def get_knowledge_area_performance() -> Dict[str, Dict]:
        """Get performance aggregated by knowledge area."""
        performance = AnalyticsService.get_topic_performance()
        knowledge_areas = AnalyticsService._get_knowledge_areas()
        areas = {}

        for topic_perf in performance.values():
            area = "Other"
            for rule_num, (topic_name, knowledge_area) in knowledge_areas.items():
                if topic_name == topic_perf.topic:
                    area = knowledge_area
                    break

            if area not in areas:
                areas[area] = {
                    "total_questions": 0,
                    "correct_answers": 0,
                    "topics": []
                }

            areas[area]["total_questions"] += topic_perf.total_questions
            areas[area]["correct_answers"] += topic_perf.correct_answers
            areas[area]["topics"].append(topic_perf)

        return areas

    @staticmethod
    def record_answer(reference: str, is_correct: bool):
        """Record an answer for a topic extracted from reference."""
        topic_name, knowledge_area = AnalyticsService.extract_topic_from_reference(reference)
        performance = AnalyticsService.get_topic_performance()

        if topic_name not in performance:
            performance[topic_name] = TopicPerformance(topic=topic_name)

        perf = performance[topic_name]
        perf.total_questions += 1
        if is_correct:
            perf.correct_answers += 1
        else:
            perf.incorrect_answers += 1
        perf.last_attempt = datetime.now()

        st.session_state[AnalyticsService._perf_key()] = performance

    @staticmethod
    def get_weak_topics() -> List[TopicPerformance]:
        """Identify topics where user is struggling (< 70% accuracy with 3+ attempts)."""
        performance = AnalyticsService.get_topic_performance()
        weak = [p for p in performance.values() if p.is_weak]
        return sorted(weak, key=lambda x: x.accuracy)

    @staticmethod
    def get_all_topics_sorted() -> List[TopicPerformance]:
        """Get all topics sorted by performance (worst first)."""
        performance = AnalyticsService.get_topic_performance()
        topics = list(performance.values())
        return sorted(topics, key=lambda x: (x.accuracy, -x.total_questions))

    @staticmethod
    def reset_analytics():
        """Reset analytics data for the active course."""
        st.session_state[AnalyticsService._perf_key()] = {}


# --- BOOKMARK SERVICE (Question Bookmarking) ---

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

        course = get_active_course()
        story.append(Paragraph(f"{course.name} Remediation Study Guide", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated by {course.name} AI", styles['Normal']))
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

def retry_with_backoff(max_attempts: int = MAX_RETRY_ATTEMPTS, base_delay: float = RETRY_BASE_DELAY):
    """
    Decorator for retrying functions with exponential backoff.
    Handles rate limiting and transient errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limit or quota errors
                    is_retryable = any(keyword in error_str for keyword in [
                        'rate limit', 'quota', 'resource exhausted',
                        '429', '503', 'overloaded', 'temporarily unavailable'
                    ])

                    if not is_retryable or attempt == max_attempts - 1:
                        raise e

                    last_exception = e
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited. Retrying in {delay}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


class GenAIProvider:
    """Encapsulates Google GenAI interactions with caching and streaming support."""

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
        self._response_cache: Dict[str, Tuple[str, float]] = {}

    def _get_cache_key(self, prompt: str, system_instr: str, use_rag: bool) -> str:
        """Generate a cache key for the request."""
        cache_string = f"{prompt}|{system_instr}|{use_rag}|{self.model}|{self.store_id}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if valid."""
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp < RESPONSE_CACHE_TTL:
                logger.info(f"Cache hit for key: {cache_key[:8]}...")
                return response
            del self._response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache a response with timestamp."""
        self._response_cache[cache_key] = (response, time.time())
        # Limit cache size to prevent memory issues
        if len(self._response_cache) > 100:
            oldest_key = min(self._response_cache.keys(),
                           key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]

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

    @retry_with_backoff()
    def generate_content(
        self,
        prompt: str,
        system_instr: str,
        use_rag: bool = True,
        response_schema: Any = None,
        timeout: int = API_TIMEOUT_SECONDS,
        use_cache: bool = True,
        stream: bool = False,
        model_override: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> Union[str, Iterator[str]]:
        """
        Generate content using Google GenAI with optional RAG, caching, and streaming.

        Args:
            prompt: User prompt text
            system_instr: System instruction text
            use_rag: Whether to use RAG file search
            response_schema: Optional response schema (not currently used)
            timeout: Request timeout in seconds (for future implementation)
            use_cache: Whether to use response caching (default True)
            stream: Whether to stream the response (default False)
            model_override: Optional model ID to use instead of default

        Returns:
            Generated text response or iterator of text chunks if streaming

        Raises:
            ValueError: If RECITATION_ERROR occurs or response is empty
            RuntimeError: If API call fails
        """
        # Determine which model to use
        active_model = model_override or self.model

        # Check cache first (only for non-streaming requests)
        cache_key = None
        if use_cache and not stream:
            cache_key = self._get_cache_key(prompt, system_instr, use_rag)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached

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

            config = types.GenerateContentConfig(**config_args)

            # Build multi-turn contents from history if provided
            if history:
                contents = []
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
                contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            else:
                contents = prompt

            # Streaming response
            if stream and ENABLE_STREAMING:
                return self._stream_response(contents, config, cache_key, active_model)

            # Non-streaming response
            response = self.client.models.generate_content(
                model=active_model,
                contents=contents,
                config=config
            )

            result_text = None
            if response.text:
                result_text = response.text
            elif response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    parts_text = [p.text for p in candidate.content.parts if p.text]
                    if parts_text:
                        result_text = "".join(parts_text)

                if not result_text and candidate.finish_reason:
                    reason = str(candidate.finish_reason)
                    if "RECITATION" in reason:
                        raise ValueError("RECITATION_ERROR")
                    raise ValueError(f"AI Generation stopped. Reason: {reason}")

            if not result_text:
                raise ValueError("The AI returned an empty response (No candidates or text found).")

            # Cache the result
            if use_cache and cache_key:
                self._cache_response(cache_key, result_text)

            return result_text

        except Exception as e:
            if "RECITATION_ERROR" in str(e):
                raise e
            logger.error(f"GenAI generation failed: {e}")
            raise RuntimeError(str(e))

    def _stream_response(
        self,
        contents,
        config: types.GenerateContentConfig,
        cache_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream response chunks from the API.

        Args:
            contents: The prompt string or multi-turn contents list to send
            config: Generation config
            cache_key: Optional cache key to store final result
            model: Model ID to use (defaults to self.model)

        Yields:
            Text chunks as they arrive
        """
        active_model = model or self.model
        full_text_parts = []
        try:
            response_stream = self.client.models.generate_content_stream(
                model=active_model,
                contents=contents,
                config=config
            )

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
                    if hasattr(cand, 'finish_reason') and cand.finish_reason:
                        if "RECITATION" in str(cand.finish_reason):
                            raise ValueError("RECITATION_ERROR")

            # Cache the complete response
            if cache_key and full_text_parts:
                self._cache_response(cache_key, "".join(full_text_parts))

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

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

    def generate_content_multimodal(
        self,
        prompt: str,
        image_bytes_list: List[bytes],
        system_instr: str,
        mime_type: str = "image/png",
        model_override: Optional[str] = None
    ) -> str:
        """
        Generate content using text prompt and multiple images (e.g., for visual grading).

        Args:
            prompt: Text prompt
            image_bytes_list: List of image bytes
            system_instr: System instruction
            mime_type: Image MIME type
            model_override: Optional model ID to use instead of default
        """
        active_model = model_override or self.model
        try:
            parts = [prompt]
            for img_bytes in image_bytes_list:
                if img_bytes:
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                system_instruction=system_instr,
                safety_settings=SAFETY_SETTINGS
            )
            response = self.client.models.generate_content(
                model=active_model,
                contents=parts,
                config=config
            )
            if response.text:
                return response.text
            if response.candidates and response.candidates[0].content.parts:
                parts_text = [p.text for p in response.candidates[0].content.parts if p.text]
                if parts_text:
                    return "".join(parts_text)
            raise ValueError("No text in multimodal response.")
        except Exception as e:
            logger.error(f"Multimodal generation failed: {e}")
            raise RuntimeError(str(e))

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
            logger.warning(f"CSV file not found: {filepath}")
            return pd.DataFrame()
        try:
            # Try UTF-8 first, then fallback encodings
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM
                except:
                    df = pd.read_csv(filepath, encoding='latin-1')  # Fallback
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            logger.error(f"CSV loading error for {filepath}: {e}", exc_info=True)
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
        
        # Use random_state=None for variety in quiz questions
        sample = filtered_df.sample(n=min(len(filtered_df), sample_size), random_state=None)
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
        """Load visual identification questions from CSV. Columns: ID, Question_Text, Image_Path, Rubric_Path."""
        if not os.path.isabs(filepath):
            full_path = os.path.join(os.getcwd(), filepath)
        else:
            full_path = filepath
        if not os.path.exists(full_path):
            logger.warning(f"Visual bank file not found: {full_path}")
            return []
        df = DataManager._load_full_csv(full_path)
        if df.empty:
            return []
        # Column stripping already done in _load_full_csv
        column_map = {col.lower(): col for col in df.columns}
        required = ["id", "question_text", "image_path", "rubric_path"]
        if not all(k in column_map for k in required):
            logger.warning(f"Visual bank missing columns. Need: {required}, got: {list(df.columns)}")
            return []
        questions = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            q_id = str(row_dict.get(column_map["id"], "")).strip() if pd.notna(row_dict.get(column_map["id"])) else ""
            q_text = str(row_dict.get(column_map["question_text"], "")).strip() if pd.notna(row_dict.get(column_map["question_text"])) else ""
            img_path = str(row_dict.get(column_map["image_path"], "")).strip() if pd.notna(row_dict.get(column_map["image_path"])) else ""
            rubric_path = str(row_dict.get(column_map["rubric_path"], "")).strip() if pd.notna(row_dict.get(column_map["rubric_path"])) else ""
            if not q_id or not q_text or not img_path or not rubric_path:
                continue
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.getcwd(), img_path)
            if not os.path.isabs(rubric_path):
                rubric_path = os.path.join(os.getcwd(), rubric_path)
            if not os.path.exists(img_path) or not os.path.exists(rubric_path):
                logger.warning(f"Skipping visual question {q_id}: image or rubric file not found.")
                continue
            try:
                questions.append(VisualQuestion(id=q_id, question_text=q_text, image_path=img_path, rubric_path=rubric_path))
            except Exception as e:
                logger.error(f"Validation error for visual question {q_id}: {e}")
        return questions

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

    @staticmethod
    @st.cache_data(ttl=BACKGROUND_IMAGE_CACHE_TTL)
    def preload_image_bytes(filepath: str) -> Optional[bytes]:
        """
        Preload and cache image bytes for faster visual quiz grading.

        Args:
            filepath: Path to image file

        Returns:
            Image bytes or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Image preload error for {filepath}: {e}")
            return None

    @staticmethod
    def preload_visual_question_images(questions: List['VisualQuestion']) -> Dict[str, bytes]:
        """
        Preload all rubric images for visual questions.

        Args:
            questions: List of visual questions

        Returns:
            Dictionary mapping question ID to rubric image bytes
        """
        preloaded = {}
        for q in questions:
            rubric_bytes = DataManager.preload_image_bytes(q.rubric_path)
            if rubric_bytes:
                preloaded[q.id] = rubric_bytes
        return preloaded

# --- SERVICE LAYER (Business Logic) ---

def generate_with_retry(
    ai_provider: GenAIProvider,
    prompt: str,
    safe_prompt: str,
    error_context: str = "generation",
    model_override: Optional[str] = None
) -> str:
    """
    Shared retry logic for handling RECITATION_ERROR across all services.

    Args:
        ai_provider: GenAI provider instance
        prompt: Initial prompt
        safe_prompt: Paraphrased prompt for retry
        error_context: Context string for logging
        model_override: Optional model ID to use

    Returns:
        Generated text response

    Raises:
        ValueError: If RECITATION_ERROR occurs on both attempts
        RuntimeError: If other errors occur
    """
    rag_on = is_rag_enabled()
    system_instr = get_active_system_instruction()
    try:
        logger.info(f"[{error_context}] Processing request: {len(prompt)} chars (RAG: {rag_on})")
        return ai_provider.generate_content(
            prompt, system_instr, use_rag=rag_on, model_override=model_override
        )
    except Exception as e:
        if "RECITATION_ERROR" in str(e):
            logger.warning(f"[{error_context}] Hit Recitation error. Retrying with safe prompt.")
            try:
                return ai_provider.generate_content(
                    safe_prompt, system_instr, use_rag=rag_on, model_override=model_override
                )
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
        course = get_active_course()
        if not st.secrets.get(course.store_id_secret):
            st.warning(f"Missing Store ID for {course.name}. RAG features will be disabled.")
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
        except json.JSONDecodeError:
            # Sanitize control characters inside JSON string values and retry
            sanitized = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() in ('\n', '\r', '\t') else '', json_str)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError as e2:
                raise ValueError(f"Invalid JSON format: {e2}. Response preview: {json_str[:200]}")

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
        # Use runtime-configurable model for quiz generation
        quiz_model = st.session_state.get("model_for_quiz", MODEL_FOR_QUIZ)
        return generate_with_retry(self.ai, prompt, safe_prompt, error_context, quiz_model)

    def generate_from_ai(self, request: QuizRequest) -> List[QuizQuestion]:
        """
        Generate quiz questions using AI with RAG.

        Args:
            request: QuizRequest containing number of questions, topic distribution, and difficulty

        Returns:
            List of QuizQuestion objects

        Raises:
            ValueError: If AI generation fails or returns invalid JSON
            RuntimeError: If API call fails
        """
        dist = self.calculate_topic_distribution(request)
        dist_str = "\n".join([f"- {c} questions about: {t}" for t, c in dist.items()])

        # Course-aware labels
        course = get_active_course()
        subject = course.subject_label

        # Difficulty-specific instructions
        difficulty_instructions = {
            DIFFICULTY_EASY: f"""
        DIFFICULTY: EASY
        - Ask straightforward, direct questions about basic rules and definitions.
        - Use simple, clear language in questions and options.
        - Avoid tricky wording or edge cases.
        - Focus on fundamental concepts that every student of {subject} should know.
        - Wrong answers should be clearly distinguishable from the correct answer.""",
            DIFFICULTY_MEDIUM: f"""
        DIFFICULTY: MEDIUM
        - Ask questions that require understanding and application of {subject} concepts.
        - Include some situational context in questions.
        - Wrong answers should be plausible but distinguishable with proper knowledge.
        - Test both knowledge and basic application of rules.""",
            DIFFICULTY_HARD: f"""
        DIFFICULTY: HARD
        - Ask complex questions involving multiple rules or edge cases.
        - Create scenarios requiring analysis and judgment.
        - Include nuanced situations where concepts interact or overlap.
        - Wrong answers should be highly plausible, testing deep understanding.
        - Focus on exceptions, special cases, and complex scenarios within {subject}."""
        }

        difficulty_text = difficulty_instructions.get(request.difficulty, difficulty_instructions[DIFFICULTY_MEDIUM])

        rag_on = is_rag_enabled()
        source_instruction = (
            "based ONLY on the provided File Store."
            if rag_on
            else f"using your full knowledge of {subject}. You are NOT limited to any specific document set."
        )

        prompt = f"""
        Create a {request.num_questions} question multiple-choice quiz on {subject} {source_instruction}
        DISTRIBUTION: {dist_str}
        {difficulty_text}
        CRITICAL: Output ONLY a valid JSON array. No commentary before or after.
        OUTPUT SCHEMA (JSON Array):
        [
            {{
                "question": "Question text...",
                "options": ["A) ...", "B) ..."],
                "correct_answer": "Full text of option",
                "reference": "Source chapter, section, or rule reference",
                "explanation": "Concise explanation."
            }}
        ]
        """

        safe_source_instruction = (
            "based on the provided File Store."
            if rag_on
            else f"using your full knowledge of {subject}."
        )

        prompt_safe = f"""
        Create a {request.num_questions} question multiple-choice quiz on {subject} {safe_source_instruction}
        DISTRIBUTION: {dist_str}
        {difficulty_text}

        CRITICAL INSTRUCTION: The previous attempt was blocked for quoting the text too closely.
        1. Rephrase the questions and options so they are distinct from the source text (Paraphrase).
        2. Test the concept/rule, do not just ask for the definition.
        3. Keep explanations concise and synthesized.

        CRITICAL: Output ONLY a valid JSON array. No commentary before or after.
        OUTPUT SCHEMA (JSON Array):
        [
            {{
                "question": "Question text (Paraphrased)...",
                "options": ["A) ...", "B) ..."],
                "correct_answer": "Full text of option",
                "reference": "Source chapter, section, or rule reference",
                "explanation": "Synthesized explanation."
            }}
        ]
        """

        raw = self._generate_with_retry(prompt, prompt_safe, "Quiz generation")
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

    def _generate_with_retry(self, prompt: str, safe_prompt: str,
                             system_instr: str, rag_on: bool, quiz_model: str) -> str:
        """Helper method to retry generation with a safe prompt if RECITATION_ERROR occurs.

        All parameters are resolved in the main thread and passed in so that
        worker threads (which lack access to st.session_state) use the correct
        course-specific values.
        """
        try:
            logger.info(f"[Remediation batch generation] Processing request: {len(prompt)} chars (RAG: {rag_on})")
            return self.ai.generate_content(
                prompt, system_instr, use_rag=rag_on, model_override=quiz_model
            )
        except Exception as e:
            if "RECITATION_ERROR" in str(e):
                logger.warning("[Remediation batch generation] Hit Recitation error. Retrying with safe prompt.")
                try:
                    return self.ai.generate_content(
                        safe_prompt, system_instr, use_rag=rag_on, model_override=quiz_model
                    )
                except Exception as e2:
                    if "RECITATION_ERROR" in str(e2):
                        logger.error("[Remediation batch generation] Safe prompt also hit RECITATION_ERROR.")
                        raise ValueError("RECITATION_ERROR")
                    raise e2
            raise e

    def _generate_batch_content(self, batch_refs: List[str],
                                course: 'CourseConfig', rag_on: bool,
                                system_instr: str, quiz_model: str) -> str:
        """
        Generate remediation content for a batch of rule references.

        Args:
            batch_refs: List of rule references to process
            course: Active course config (resolved in main thread)
            rag_on: Whether RAG is enabled (resolved in main thread)
            system_instr: System instruction (resolved in main thread)
            quiz_model: Model to use (resolved in main thread)

        Returns:
            Markdown-formatted remediation content
        """
        topics_context = ", ".join(batch_refs)

        remediation_source = (
            "Based ONLY on the provided File Store documents."
            if rag_on
            else f"Using your full knowledge of {course.subject_label}."
        )

        prompt = f"""
        Create a "Deep Dive" remediation section for these specific topics: {topics_context}.
        Subject: {course.subject_label}.
        {remediation_source}

        For each topic:
        1. **Header**: Topic/Rule Number/Name.
        2. **Official Text**: Provide the exact verbatim text of the key sentence (1-3 sentences max).
        3. **Explanation**: Plain English summary.
        4. **Common Pitfalls**: Why students miss this.
        5. **Memory Aid**: Mnemonic or trick.

        OUTPUT FORMAT: Markdown. Do not include intro or outro.
        """

        prompt_safe = f"""
        Create a "Deep Dive" remediation section for these specific topics: {topics_context}.
        Subject: {course.subject_label}.

        CRITICAL: Paraphrase all text. Do not quote source text directly. Create original summaries.

        For each topic:
        1. **Header**: Topic/Rule Number/Name.
        2. **Summary**: Summarize concisely. Paraphrase - do not quote verbatim.
        3. **Explanation**: Plain English summary.
        4. **Common Pitfalls**: Why students miss this.
        5. **Memory Aid**: Mnemonic or trick.

        OUTPUT FORMAT: Markdown. Do not include intro or outro.
        """

        try:
            return self._generate_with_retry(prompt, prompt_safe,
                                             system_instr, rag_on, quiz_model)
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

        # Resolve course, RAG, and model settings in the MAIN thread so that
        # worker threads (which lack st.session_state) get the correct values.
        course = get_active_course()
        rag_on = is_rag_enabled()
        # Always use the strict RAG instruction for remediation so lesson content
        # is grounded exclusively in uploaded documents.  If File Search returns
        # no results for a topic the model will say so rather than hallucinate.
        system_instr = course.system_instruction if rag_on else course.system_instruction_open
        quiz_model = st.session_state.get("model_for_quiz", MODEL_FOR_QUIZ)

        # Parallel processing with progress tracking
        progress_bar = st.progress(0) if IN_STREAMLIT else None
        status_text = st.empty() if IN_STREAMLIT else None

        try:
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(self._generate_batch_content, batch,
                                    course, rag_on, system_instr, quiz_model): batch
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
        # Use runtime-configurable model for oral board
        oral_model = st.session_state.get("model_for_oral", MODEL_FOR_ORAL)
        return generate_with_retry(self.ai, prompt, safe_prompt, "Oral Board scenario generation", oral_model)

    def generate_scenario(self) -> str:
        """Generates a complex oral exam scenario."""
        course = get_active_course()
        prompt = f"""
        Generate a specific, complex 'Oral Board' scenario for a {course.subject_label} candidate.
        Subject: {course.oral_subject}.

        Format the output clearly:
        **Scenario:** [Describe the situation with relevant details]
        **Question:** [What is the specific situation and what is your required action?]
        """

        prompt_safe = f"""
        Generate a specific, complex 'Oral Board' scenario for a {course.subject_label} candidate.
        Subject: {course.oral_subject}.

        CRITICAL: Paraphrase all content. Do not quote source text directly. Create original scenario descriptions.

        Format the output clearly:
        **Scenario:** [Describe the situation with relevant details]
        **Question:** [What is the specific situation and what is your required action?]
        """

        return self._generate_with_retry(prompt, prompt_safe)

    def grade_response(self, scenario: str, audio_bytes: bytes) -> str:
        """Grades the audio response against the scenario."""
        course = get_active_course()
        prompt = f"""
        You are a {course.examiner_persona}.
        
        **SCENARIO GIVEN TO CANDIDATE:**
        {scenario}
        
        **TASK:**
        Listen to the candidate's audio response provided in the attachment.
        
        **OUTPUT FORMAT:**
        1. **Transcript:** Write down exactly what the candidate said.
        2. **Grade:** PASS or FAIL.
        3. **Critique:**
           - Did they identify the correct {course.oral_grading_concept}?
           - Did they state the correct answer?
           - Was the answer delivered confidently?
        4. **Correct Answer:** What was the expected response?
        """
        
        response = self.ai.generate_content_with_audio(
            prompt=prompt,
            audio_bytes=audio_bytes,
            mime_type="audio/wav", # Streamlit audio recorder output
            system_instr=get_active_system_instruction(),
            use_rag=is_rag_enabled()  # Respects RAG Master Switch
        )
        
        # Filter out citation metadata but preserve all formatting and content
        cleaned = filter_citation_metadata(response)
        
        # Ensure we have valid content - if filtering removed everything meaningful, return original
        if not cleaned or (len(cleaned.strip()) < 50 and len(response.strip()) > 100):
            logger.warning("Citation filter may have been too aggressive, using original response")
            return response
        
        return cleaned


class VoiceChatService:
    """Handles voice-based chat interactions for conversational Q&A."""

    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def transcribe_and_respond(self, audio_bytes: bytes) -> Tuple[str, str]:
        """
        Process audio input: transcribe and generate response.
        Returns: (transcription, response)
        """
        voice_model = st.session_state.get("model_for_voice", MODEL_FOR_VOICE)
        rag_on = is_rag_enabled()
        voice_instr = get_active_voice_instruction()

        course = get_active_course()
        prompt = f"""
        Listen to the user's question about {course.subject_label}.

        1. First, transcribe exactly what the user said.
        2. Then, provide a helpful, conversational response.

        Format your response as:
        TRANSCRIPTION: [what the user said]
        RESPONSE: [your helpful answer]

        Keep responses concise and suitable for spoken delivery.
        """

        try:
            response = self.ai.generate_content_with_audio(
                prompt=prompt,
                audio_bytes=audio_bytes,
                mime_type="audio/wav",
                system_instr=voice_instr,
                use_rag=rag_on
            )

            # Parse response
            transcription = ""
            answer = response

            if "TRANSCRIPTION:" in response and "RESPONSE:" in response:
                parts = response.split("RESPONSE:")
                trans_part = parts[0].replace("TRANSCRIPTION:", "").strip()
                transcription = trans_part
                answer = parts[1].strip() if len(parts) > 1 else ""

            return transcription, answer

        except Exception as e:
            logger.error(f"Voice chat error: {e}")
            raise RuntimeError(f"Voice processing failed: {str(e)}")

    def text_to_voice_response(self, text_input: str) -> str:
        """Generate a voice-friendly response from text input."""
        voice_model = st.session_state.get("model_for_voice", MODEL_FOR_VOICE)
        rag_on = is_rag_enabled()
        voice_instr = get_active_voice_instruction()

        course = get_active_course()
        prompt = f"""
        User question: {text_input}

        Provide a helpful, conversational response about {course.subject_label}.
        Keep it concise and suitable for spoken delivery.
        """

        try:
            return self.ai.generate_content(
                prompt=prompt,
                system_instr=voice_instr,
                use_rag=rag_on,
                model_override=voice_model
            )
        except Exception as e:
            logger.error(f"Voice text response error: {e}")
            raise


class GradingService:
    """Handles auto-grading of Free Response Questions against curriculum standards."""
    
    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider

    def grade_submission(self, question: FreeResponseQuestion, student_answer: str) -> GradingResult:
        """
        Compares student answer to the ideal answer using AI with context-aware intelligence.
        """
        # UPDATED PROMPT: Intelligent Concept Extraction (v18.0)
        course = get_active_course()
        prompt = f"""
        You are a strict automated grading system for a {course.subject_label} exam.
        
        **QUESTION:** {question.question_text}
        **IDEAL ANSWER (CURRICULUM STANDARD):** {question.ideal_answer}
        **STUDENT SUBMISSION:** {student_answer}
        
        **INSTRUCTIONS FOR INTELLIGENT GRADING:**
        1. **Extract Key Concepts:** Identify the core technical facts/actions in the Ideal Answer.
        2. **Analyze Student Answer:** Check if these core facts are present in the Student Submission.
        3. **Apply Context:** 
           - {course.grading_synonym_hint}
           - Do NOT simply match words. Match the *meaning* and *intent*.
           - Be strict about specific numbers (lights, shapes, degrees).
        
        **SCORING:**
        - 0-100 score based on how many Key Concepts from the Ideal Answer are present in the Student Submission.
        - 100 = All concepts present and correct.
        - 0 = No concepts present or answer is factually dangerous.
        
        **OUTPUT FORMAT (JSON ONLY):**
        {{
            "score": <int 0-100>,
            "reasoning": "<Concise justification: Which concepts were hit/missed?>"
        }}
        """
        
        try:
            # use_rag=False to strictly adhere to the CSV provided answer
            # Use runtime-configurable model for grading
            grading_model = st.session_state.get("model_for_grading", MODEL_FOR_GRADING)
            response_text = self.ai.generate_content(
                prompt=prompt,
                system_instr="You are a strict, intelligent grading assistant.",
                use_rag=False,
                model_override=grading_model
            )
            
            # Parse JSON response
            clean_text = re.sub(r'```json\s*', '', response_text)
            clean_text = re.sub(r'```', '', clean_text).strip()
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(clean_text[start:end])
                score = int(data.get("score", 0))
                reasoning = data.get("reasoning", "No reasoning provided.")
                
                if score >= 80:
                    status = "Correct"
                elif score >= 50:
                    status = "Flagged for Review"
                else:
                    status = "Incorrect"
                    
                return GradingResult(score=score, status=status, reasoning=reasoning)
                
            raise ValueError("Invalid JSON from grader")
            
        except Exception as e:
            logger.error(f"Grading error: {e}")
            return GradingResult(score=0, status="Error", reasoning=f"System error: {str(e)}")


class VisualGradingService:
    """Grades student drawings against a rubric image using multimodal AI."""

    # Max dimension for images sent to API (smaller = faster)
    MAX_IMAGE_DIMENSION = 800
    JPEG_QUALITY = 85

    def __init__(self, ai_provider: GenAIProvider):
        self.ai = ai_provider
        self._rubric_cache: Dict[str, bytes] = {}
        self._base_image_cache: Dict[str, Image.Image] = {}

    def preload_rubrics(self, questions: List[VisualQuestion]):
        """Preload and optimize rubric images for faster grading."""
        for q in questions:
            # Preload and optimize rubric
            rubric_bytes = self._load_and_optimize_image(q.rubric_path)
            if rubric_bytes:
                self._rubric_cache[q.id] = rubric_bytes
            # Preload base images too
            try:
                base_img = Image.open(q.image_path).convert("RGBA")
                self._base_image_cache[q.id] = base_img
            except Exception as e:
                logger.warning(f"Could not preload base image for {q.id}: {e}")
        logger.info(f"Preloaded {len(self._rubric_cache)} rubric images and {len(self._base_image_cache)} base images")

    def _load_and_optimize_image(self, filepath: str) -> Optional[bytes]:
        """Load image, resize if needed, and convert to optimized JPEG."""
        try:
            img = Image.open(filepath)
            img = self._resize_image(img)
            # Convert to RGB for JPEG (removes alpha channel)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.JPEG_QUALITY, optimize=True)
            return buffer.getvalue()
        except Exception as e:
            logger.warning(f"Image optimization failed for {filepath}: {e}")
            return None

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if larger than MAX_IMAGE_DIMENSION while maintaining aspect ratio."""
        width, height = img.size
        if width <= self.MAX_IMAGE_DIMENSION and height <= self.MAX_IMAGE_DIMENSION:
            return img
        if width > height:
            new_width = self.MAX_IMAGE_DIMENSION
            new_height = int(height * (self.MAX_IMAGE_DIMENSION / width))
        else:
            new_height = self.MAX_IMAGE_DIMENSION
            new_width = int(width * (self.MAX_IMAGE_DIMENSION / height))
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def grade_drawing(self, question: VisualQuestion, drawing_data: np.ndarray) -> GradingResult:
        try:
            # Use cached base image if available
            if question.id in self._base_image_cache:
                base_image = self._base_image_cache[question.id].copy()
            else:
                base_image = Image.open(question.image_path).convert("RGBA")

            # Process drawing
            drawing_image = Image.fromarray(drawing_data.astype("uint8"), "RGBA")
            if drawing_image.size != base_image.size:
                drawing_image = drawing_image.resize(base_image.size, Image.Resampling.NEAREST)

            # Composite and optimize
            student_composite = Image.alpha_composite(base_image, drawing_image)
            student_composite = self._resize_image(student_composite)

            # Convert to optimized JPEG (much faster than PNG)
            if student_composite.mode in ('RGBA', 'P'):
                student_composite = student_composite.convert('RGB')
            student_bytes_io = BytesIO()
            student_composite.save(student_bytes_io, format="JPEG", quality=self.JPEG_QUALITY, optimize=True)
            student_bytes = student_bytes_io.getvalue()

            # Use preloaded rubric if available
            rubric_bytes = self._rubric_cache.get(question.id)
            if not rubric_bytes:
                rubric_bytes = self._load_and_optimize_image(question.rubric_path)
            if not rubric_bytes:
                # Fallback to direct file read
                with open(question.rubric_path, "rb") as f:
                    rubric_bytes = f.read()

            prompt = (
                f"Compare images. Image 1: Student answer (drawing on base). Image 2: Answer key. "
                f"Question: {question.question_text}. "
                "Respond with JSON only: {\"score\": <0-100>, \"reasoning\": \"<string>\"}"
            )
            # Use runtime-configurable model for visual grading
            visual_model = st.session_state.get("model_for_visual", MODEL_FOR_VISUAL)
            response_text = self.ai.generate_content_multimodal(
                prompt, [student_bytes, rubric_bytes], "Strict visual grader.",
                model_override=visual_model,
                mime_type="image/jpeg"
            )

            clean_text = re.sub(r"```json\s*|```", "", response_text).strip()
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            if start != -1 and end != 0:
                data = json.loads(clean_text[start:end])
                score = int(data.get("score", 0))
                status = "Correct" if score >= 80 else "Incorrect"
                return GradingResult(
                    score=score,
                    status=status,
                    reasoning=data.get("reasoning", ""),
                )
            raise ValueError("No valid JSON in response")
        except Exception as e:
            logger.error(f"Visual grading error: {e}")
            return GradingResult(score=0, status="Error", reasoning=f"System Error: {str(e)}")


# --- PRESENTATION LAYER ---

def _get_welcome_message() -> str:
    """Return a welcome message for the active course."""
    course = get_active_course()
    return f"{course.icon} {course.name} Expert Ready."


def init_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        "messages": [{"role": "assistant", "content": _get_welcome_message()}],
        "quiz_data": None,
        "current_quiz_source": None,
        "incorrect_questions": [],
        "remediation_text": None,
        "oral_scenario": None,  # For Oral Board
        "oral_grade": None,      # For Oral Board
        "frq_bank": [],          # Store loaded FRQ questions
        "frq_results": {},       # Store user results {question_id: GradingResult}
        "visual_bank": [],      # Store loaded visual questions
        "visual_results": {},   # Store visual quiz results {question_id: GradingResult}
        "quiz_answers": {},      # Persist quiz answers across reruns
        "quiz_submitted": False, # Track if quiz has been submitted
        "voice_messages": [],         # Voice chat history
        # RAG Master Switch (True = RAG/File Store mode, False = Open LLM mode)
        "rag_enabled": True,
        # Course selection
        "active_course": DEFAULT_COURSE_KEY,
        # Model configuration (runtime overrides)
        "model_for_chat": MODEL_FOR_CHAT,
        "model_for_quiz": MODEL_FOR_QUIZ,
        "model_for_grading": MODEL_FOR_GRADING,
        "model_for_oral": MODEL_FOR_ORAL,
        "model_for_visual": MODEL_FOR_VISUAL,
        "model_for_voice": MODEL_FOR_VOICE,
        # Admin / query logging
        "query_log": load_query_log(),
        "admin_authenticated": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def switch_course(new_key: str):
    """Switch to a different course, resetting course-specific state while preserving settings."""
    if new_key == st.session_state.get("active_course"):
        return
    # Preserve global settings
    saved_models = {
        "model_for_chat": st.session_state.get("model_for_chat"),
        "model_for_quiz": st.session_state.get("model_for_quiz"),
        "model_for_grading": st.session_state.get("model_for_grading"),
        "model_for_oral": st.session_state.get("model_for_oral"),
        "model_for_visual": st.session_state.get("model_for_visual"),
        "model_for_voice": st.session_state.get("model_for_voice"),
    }
    saved_rag = st.session_state.get("rag_enabled", True)
    saved_mode = st.session_state.get("selected_mode", "Chat 🤖")

    # Clear course-specific state
    st.session_state.messages = []
    st.session_state.quiz_data = None
    st.session_state.incorrect_questions = []
    st.session_state.remediation_text = None
    st.session_state.oral_scenario = None
    st.session_state.oral_grade = None
    st.session_state.frq_bank = []
    st.session_state.frq_results = {}
    st.session_state.visual_bank = []
    st.session_state.visual_results = {}
    st.session_state.voice_messages = []
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False

    # Set new course
    st.session_state.active_course = new_key
    course = COURSES[new_key]
    st.session_state.messages = [{"role": "assistant", "content": f"{course.icon} {course.name} Expert Ready."}]

    # Restore settings
    for k, v in saved_models.items():
        st.session_state[k] = v
    st.session_state.rag_enabled = saved_rag
    st.session_state.selected_mode = saved_mode


def get_active_model(task: str) -> Optional[str]:
    """Get the active model for a specific task from session state."""
    key = f"model_for_{task}"
    return st.session_state.get(key)


def is_rag_enabled() -> bool:
    """Check if RAG mode is enabled (Master Switch)."""
    return st.session_state.get("rag_enabled", True)


def get_active_system_instruction() -> str:
    """Get the system instruction based on RAG toggle state and active course."""
    course = get_active_course()
    if is_rag_enabled():
        return course.system_instruction
    return course.system_instruction_open


def get_active_voice_instruction() -> str:
    """Get the voice chat instruction based on RAG toggle state and active course."""
    course = get_active_course()
    if is_rag_enabled():
        return course.voice_instruction
    return course.voice_instruction_open


# --- QUERY LOGGING ---

def _parse_sources_from_response(response: str) -> List[str]:
    """Extract source citations from a response string."""
    pattern = re.compile(r'\*\*Source:\s*([^,\n*]+),\s*Page\s*([^*\n]+)\*\*')
    return [
        f"{m.group(1).strip()}, p.{m.group(2).strip()}"
        for m in pattern.finditer(response)
    ]


def _is_found_in_docs(response: str) -> bool:
    """Return False if the response indicates the RAG search found nothing."""
    not_found_phrases = [
        "could not find information",
        "not found in the current manuals",
        "no relevant results",
        "i'm sorry, i could not",
        "i am sorry, i could not",
    ]
    lower = response.lower()
    return not any(phrase in lower for phrase in not_found_phrases)


def make_query_entry(question: str, response: str, rag_mode: bool) -> dict:
    """Build a query log entry dict from a completed chat exchange."""
    return {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response_snippet": response[:300] + ("..." if len(response) > 300 else ""),
        "sources": _parse_sources_from_response(response),
        "found_in_docs": _is_found_in_docs(response),
        "rag_mode": rag_mode,
    }


def load_query_log() -> List[dict]:
    """Load persisted query log from disk; returns empty list on any error."""
    if not os.path.exists(QUERY_LOG_FILE):
        return []
    entries = []
    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return entries


def persist_query_entry(entry: dict):
    """Append one entry to the on-disk log (silent fail on cloud/read-only FS)."""
    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def log_query(entry: dict):
    """Record a query entry in session state and on disk."""
    if "query_log" not in st.session_state:
        st.session_state.query_log = []
    st.session_state.query_log.append(entry)
    persist_query_entry(entry)


def render_model_settings():
    """Render model configuration UI in sidebar."""
    with st.expander("⚙️ Model Settings", expanded=False):
        st.caption("Switch between Flash (fast) and Pro (accurate)")

        model_options = {
            "Flash (Fast)": MODEL_FLASH,
            "Pro (Accurate)": MODEL_PRO,
        }
        model_labels = {v: k for k, v in model_options.items()}

        # Chat Model
        current_chat = st.session_state.model_for_chat or MODEL_ID
        chat_label = model_labels.get(current_chat, "Flash (Fast)")
        new_chat = st.selectbox(
            "Chat",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(chat_label),
            key="select_chat_model"
        )
        st.session_state.model_for_chat = model_options[new_chat]

        # Quiz Model
        current_quiz = st.session_state.model_for_quiz or MODEL_ID
        quiz_label = model_labels.get(current_quiz, "Flash (Fast)")
        new_quiz = st.selectbox(
            "Quiz Generation",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(quiz_label),
            key="select_quiz_model"
        )
        st.session_state.model_for_quiz = model_options[new_quiz]

        # Grading Model
        current_grading = st.session_state.model_for_grading or MODEL_ID
        grading_label = model_labels.get(current_grading, "Flash (Fast)")
        new_grading = st.selectbox(
            "FRQ Grading",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(grading_label),
            key="select_grading_model"
        )
        st.session_state.model_for_grading = model_options[new_grading]

        # Oral Model
        current_oral = st.session_state.model_for_oral or MODEL_ID
        oral_label = model_labels.get(current_oral, "Flash (Fast)")
        new_oral = st.selectbox(
            "Oral Board",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(oral_label),
            key="select_oral_model"
        )
        st.session_state.model_for_oral = model_options[new_oral]

        # Visual Model
        current_visual = st.session_state.model_for_visual or MODEL_ID
        visual_label = model_labels.get(current_visual, "Flash (Fast)")
        new_visual = st.selectbox(
            "Visual Grading",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(visual_label),
            key="select_visual_model"
        )
        st.session_state.model_for_visual = model_options[new_visual]

        # Voice Model
        current_voice = st.session_state.get("model_for_voice") or MODEL_ID
        voice_label = model_labels.get(current_voice, "Flash (Fast)")
        new_voice = st.selectbox(
            "Voice Chat",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(voice_label),
            key="select_voice_model"
        )
        st.session_state.model_for_voice = model_options[new_voice]

        # Quick presets
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("All Flash", use_container_width=True):
                st.session_state.model_for_chat = MODEL_FLASH
                st.session_state.model_for_quiz = MODEL_FLASH
                st.session_state.model_for_grading = MODEL_FLASH
                st.session_state.model_for_oral = MODEL_FLASH
                st.session_state.model_for_visual = MODEL_FLASH
                st.session_state.model_for_voice = MODEL_FLASH
                st.rerun()
        with col2:
            if st.button("All Pro", use_container_width=True):
                st.session_state.model_for_chat = MODEL_PRO
                st.session_state.model_for_quiz = MODEL_PRO
                st.session_state.model_for_grading = MODEL_PRO
                st.session_state.model_for_oral = MODEL_PRO
                st.session_state.model_for_visual = MODEL_PRO
                st.session_state.model_for_voice = MODEL_PRO
                st.rerun()


@st.cache_resource
def _build_ai_provider(store_id_secret: str):
    """
    Create and cache an AI provider for a given store-ID secret key.
    Cached per unique store_id_secret so switching courses reuses providers.
    """
    api_key = st.secrets.get(API_KEY_NAME)
    store_id = st.secrets.get(store_id_secret, "")
    if not api_key:
        return None
    return GenAIProvider(api_key, store_id or "")


def get_ai_provider():
    """Return the AI provider for the active course."""
    course = get_active_course()
    return _build_ai_provider(course.store_id_secret)


def set_background():
    """Ethos design system: solid warm off-white background (no image texture)."""
    # Background is handled by .streamlit/config.toml and inject_ethos_css().
    # Background images are removed per Ethos "no textures" rule.
    pass


def inject_ethos_css():
    """Inject Ethos Design System CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Ethos Design System variables ─────────────────────────────── */
    :root {
        --bg:          #FBF9F7;
        --card:        #ffffff;
        --border:      #e8e5e1;
        --row-alt:     #f0eeeb;
        --row-hover:   #faf8f6;
        --text:        #1a1a1a;
        --muted:       #6b6b6b;
        --accent:      #000000;
        --green:       #6FC07A;
        --green-dark:  #4a9c55;
        --yellow:      #d4940a;
        --red:         #cf1322;
        --shadow:      0 1px 3px rgba(0,0,0,0.04);
        --radius:      12px;
        --font:        -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    }

    /* ── Surface ────────────────────────────────────────────────────── */
    .stApp {
        background-color: var(--bg) !important;
        background-image: none !important;
        font-family: var(--font) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 24px;
    }

    /* ── Typography ─────────────────────────────────────────────────── */
    h1 {
        color: var(--accent) !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        font-family: var(--font) !important;
        margin-bottom: 4px !important;
    }
    h2 {
        color: var(--text) !important;
        font-family: var(--font) !important;
    }
    h3 {
        color: var(--muted) !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-family: var(--font) !important;
    }
    p, li {
        font-family: var(--font) !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        background: var(--card) !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
        font-size: 13px !important;
        box-shadow: none !important;
        transition: border-color 0.15s, color 0.15s;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--card) !important;
    }
    /* Primary (active mode) buttons → black fill */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        border-color: var(--accent) !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #333333 !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }

    /* ── Cards / containers ─────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
        background: var(--card) !important;
    }
    [data-testid="stChatMessage"] {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
    }

    /* ── Metrics (score cards) ──────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        color: var(--accent) !important;
        font-family: var(--font) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: var(--muted) !important;
        font-family: var(--font) !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        color: var(--muted) !important;
    }

    /* ── Tables ─────────────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: var(--font); }
    th {
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        color: var(--muted) !important;
        padding: 8px 10px !important;
        border-bottom: 2px solid var(--border) !important;
    }
    td { padding: 9px 10px !important; border-bottom: 1px solid var(--row-alt) !important; }
    tr:last-child td { border-bottom: none !important; }
    tr:hover td { background: var(--row-hover) !important; }

    /* ── Inputs ─────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-color: var(--border) !important;
        background-color: var(--card) !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* ── Selectbox / dropdowns ──────────────────────────────────────── */
    [data-baseweb="select"] > div {
        border-color: var(--border) !important;
        background-color: var(--card) !important;
        border-radius: 6px !important;
    }

    /* ── Dividers ───────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; border-width: 1px 0 0 !important; }

    /* ── Caption / muted text ───────────────────────────────────────── */
    .stCaption, small, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 12px !important;
        font-family: var(--font) !important;
    }

    /* ── Info / warning / success boxes ────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow) !important;
    }

    /* ── Progress bar ───────────────────────────────────────────────── */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--green-dark) !important;
    }

    /* ── Spinner ────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] { color: var(--muted) !important; }

    /* ── Radio buttons ──────────────────────────────────────────────── */
    [data-testid="stRadio"] label { font-family: var(--font) !important; font-size: 13px !important; }

    /* ── Slider ─────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] { background-color: var(--accent) !important; }

    /* ── Multiselect tags ───────────────────────────────────────────── */
    [data-baseweb="tag"] {
        background-color: var(--row-alt) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 4px !important;
    }

    /* ── Preserve Streamlit Material Symbols icon font ──────────────── */
    /* Streamlit renders icons as text inside spans using Material Symbols.
       Any global font-family override (including config.toml font setting)
       will break these, showing raw strings like _arrow_right, art_,
       keyboard_double_arrow_left. Restore the icon font explicitly. */
    [class*="material-symbols"],
    [class*="MaterialSymbol"],
    span[data-testid*="Icon"],
    .stIconMaterial,
    button[data-testid*="baseButton"] span,
    [data-testid="stSidebar"] button span,
    summary span {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', serif !important;
    }

    /* ── Apply Ethos system font to text elements only ──────────────── */
    body, input, textarea, select, button {
        font-family: var(--font) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_oral_board(oral_service: OralBoardService):
    st.title("🎙️ Oral Board Simulator")
    st.caption(get_active_course().oral_board_caption)
    
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


def render_weak_topics():
    """Render weak topics analysis panel."""
    st.title("📊 Performance Analytics")
    st.caption("Track your progress and identify areas needing improvement.")

    all_topics = AnalyticsService.get_all_topics_sorted()
    weak_topics = AnalyticsService.get_weak_topics()
    knowledge_areas = AnalyticsService.get_knowledge_area_performance()

    if not all_topics:
        st.info("No quiz data yet. Complete some quizzes to see your performance analysis.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_questions = sum(t.total_questions for t in all_topics)
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Rules Studied", len(all_topics))
    with col3:
        st.metric("Weak Rules", len(weak_topics))
    with col4:
        if all_topics:
            total_correct = sum(t.correct_answers for t in all_topics)
            overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
            st.metric("Overall Accuracy", f"{overall_accuracy:.0f}%")

    # Knowledge Area Performance
    if knowledge_areas:
        st.subheader("📚 Performance by Knowledge Area")
        for area_name, area_data in sorted(knowledge_areas.items()):
            if area_data["total_questions"] > 0:
                area_accuracy = (area_data["correct_answers"] / area_data["total_questions"]) * 100
                status_icon = "🔴" if area_accuracy < 70 else "🟡" if area_accuracy < 85 else "🟢"

                with st.expander(f"{status_icon} {area_name} - {area_accuracy:.0f}% ({area_data['total_questions']} questions)"):
                    st.progress(area_accuracy / 100)

                    # Show individual rules in this area
                    st.write("**Rules in this area:**")
                    for topic_perf in sorted(area_data["topics"], key=lambda x: x.accuracy):
                        rule_status = "🔴" if topic_perf.is_weak else "⚠️" if topic_perf.needs_more_data else "💪" if topic_perf.is_strong else "🟢"
                        st.write(f"  {rule_status} {topic_perf.topic}: {topic_perf.accuracy:.0f}% ({topic_perf.correct_answers}/{topic_perf.total_questions})")

    st.divider()

    # Weak topics alert
    if weak_topics:
        st.warning(f"⚠️ You have {len(weak_topics)} weak rule(s) that need attention!")

        st.subheader("🔴 Weak Rules (< 70% accuracy with 3+ attempts)")
        for topic in weak_topics:
            # Find the knowledge area for this topic
            knowledge_area = "Other"
            for rule_num, (topic_name, area) in get_active_course().rule_knowledge_areas.items():
                if topic_name == topic.topic:
                    knowledge_area = area
                    break

            with st.expander(f"🔴 {topic.topic} - {topic.accuracy:.0f}% accuracy"):
                st.caption(f"Knowledge Area: {knowledge_area}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Questions:** {topic.total_questions}")
                with col2:
                    st.write(f"**Correct:** {topic.correct_answers}")
                with col3:
                    st.write(f"**Incorrect:** {topic.incorrect_answers}")
                st.progress(topic.accuracy / 100)

                # Create a cleaner topic name for focus practice
                focus_name = topic.topic
                if st.button(f"🎯 Focus Practice", key=f"practice_{topic.topic}"):
                    st.session_state["focus_topic"] = focus_name
                    st.success(f"Focus mode set! Go to Quiz mode to practice {focus_name}.")

    # All topics table
    st.subheader("📋 All Rules Performance")
    if all_topics:
        data = []
        for t in all_topics:
            # Find knowledge area
            knowledge_area = "Other"
            for rule_num, (topic_name, area) in get_active_course().rule_knowledge_areas.items():
                if topic_name == t.topic:
                    knowledge_area = area
                    break

            data.append({
                "Rule": t.topic,
                "Knowledge Area": knowledge_area,
                "Questions": t.total_questions,
                "Correct": t.correct_answers,
                "Accuracy": f"{t.accuracy:.0f}%",
                "Status": "🔴 Weak" if t.is_weak else "⚠️ Needs More Data" if t.needs_more_data else "💪 Strong" if t.is_strong else "🟢 OK"
            })
        st.dataframe(data, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("🗑️ Reset All Analytics"):
        AnalyticsService.reset_analytics()
        st.success("Analytics reset!")
        st.rerun()


def render_voice_chat(voice_service: VoiceChatService):
    """Render voice chat interface."""
    st.title("🎙️ Voice Chat")
    st.caption("Ask questions using your voice or type below for conversational responses.")

    if not HAS_AUDIO_RECORDER:
        st.warning("⚠️ Audio recorder not available. Install with: `pip install audio-recorder-streamlit`")
        st.info("You can still use text input below for voice-style responses.")

    # Voice messages history
    for msg in st.session_state.voice_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("transcription"):
                st.caption(f"🎤 You said: {msg['transcription']}")

    # Audio input section
    if HAS_AUDIO_RECORDER:
        st.subheader("🎤 Record Your Question")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            key="voice_recorder"
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            if st.button("🚀 Process Voice Input"):
                with st.spinner("Processing your question..."):
                    try:
                        transcription, response = voice_service.transcribe_and_respond(audio_bytes)

                        st.session_state.voice_messages.append({
                            "role": "user",
                            "content": transcription or "[Voice Input]",
                            "transcription": transcription
                        })
                        st.session_state.voice_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing voice: {e}")

    # Text input fallback
    st.divider()
    st.subheader("⌨️ Or Type Your Question")
    text_input = st.text_input("Your question:", key="voice_text_input", placeholder=get_active_course().voice_placeholder)

    if st.button("Send") and text_input:
        with st.spinner("Generating response..."):
            try:
                response = voice_service.text_to_voice_response(text_input)
                st.session_state.voice_messages.append({
                    "role": "user",
                    "content": text_input
                })
                st.session_state.voice_messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("Clear Voice History"):
        st.session_state.voice_messages = []
        st.rerun()


def render_frq_section(grading_service: GradingService):
    st.title("✍️ Free Response Practice")
    st.caption("Type your answers. The system will compare them to the official curriculum standard.")

    # Load Questions
    if not st.session_state.frq_bank:
        course = get_active_course()
        filepath = course.frq_csv
        # Check if file exists
        if not os.path.exists(filepath):
            st.error(f"❌ File not found: `{filepath}`")
            st.info(f"Current working directory: {os.getcwd()}")
            st.info(f"Please ensure `{filepath}` is in the same directory as the app.")
            return
        
        frq_questions = DataManager.load_frq_bank(filepath)
        if not frq_questions:
            st.warning(f"⚠️ No questions found. Please ensure `{filepath}` exists and is populated.")
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                df = DataManager._load_full_csv(filepath)
                st.write(f"**File Path:** `{filepath}`")
                st.write(f"**File exists:** {os.path.exists(filepath)}")
                st.write(f"**Current directory:** `{os.getcwd()}`")
                st.write(f"**DataFrame empty:** {df.empty}")
                if not df.empty:
                    st.write(f"**DataFrame shape:** {df.shape}")
                    st.write(f"**Columns found:** {list(df.columns)}")
                    st.write(f"**Expected columns:** ID, Topic, Question_Text, Ideal_Answer")
                    
                    # Check for required columns (case-insensitive)
                    column_map = {col.lower().strip(): col for col in df.columns}
                    required = {'id': 'ID', 'question_text': 'Question_Text'}
                    missing = []
                    for req_key, req_name in required.items():
                        if req_key not in column_map:
                            missing.append(req_name)
                    
                    if missing:
                        st.error(f"**Missing required columns:** {missing}")
                    else:
                        st.success("✅ All required columns found!")
                    
                    st.write("**First few rows:**")
                    st.dataframe(df.head())
                    
                    # Show sample data from first row
                    if len(df) > 0:
                        st.write("**Sample data from first row:**")
                        first_row = df.iloc[0].to_dict()
                        for key, value in first_row.items():
                            st.write(f"- **{key}:** {str(value)[:100] if len(str(value)) > 100 else str(value)}")
                    
                    # Add button to clear cache and retry
                    if st.button("🔄 Clear Cache and Retry"):
                        st.cache_data.clear()
                        st.rerun()
            
            # Create dummy CSV instructions for user
            with st.expander("How to set up Free Response Questions"):
                st.markdown("""
                Create a file named **{filepath}** in your project folder with these columns:
                `ID`, `Topic`, `Question_Text`, `Ideal_Answer`
                
                **Example Content:**
                ```csv
                ID,Topic,Question_Text,Ideal_Answer
                1,Rule 5,Define a proper look-out.,"Maintain proper look-out by sight and hearing..."
                ```
                """)
            return
        st.session_state.frq_bank = frq_questions

    # Render Questions
    for q in st.session_state.frq_bank:
        with st.container():
            st.markdown(f"**Q{q.id}: {q.question_text}**")
            
            # Input key unique to question ID
            user_input = st.text_area(f"Your Answer ({q.topic}):", key=f"frq_input_{q.id}")
            
            if st.button(f"Submit Answer {q.id}", key=f"btn_{q.id}"):
                stripped_input = user_input.strip()
                if not stripped_input:
                    st.warning("Please type an answer first.")
                elif len(stripped_input) < MIN_ANSWER_LENGTH:
                    st.warning(f"Please provide a more detailed answer (at least {MIN_ANSWER_LENGTH} characters).")
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


def render_visual_quiz(visual_service: VisualGradingService):
    st.subheader("🎨 Visual Identification Quiz")

    course = get_active_course()
    st.caption(course.visual_caption)

    v_questions = DataManager.load_visual_bank(course.visual_csv)
    if not v_questions:
        st.warning(f"⚠️ No visual questions found. Ensure `{course.visual_csv}` exists and is populated.")
        return
    st.session_state.visual_bank = v_questions
    # Preload rubric images for faster grading
    visual_service.preload_rubrics(v_questions)

    q_options = {f"Q{q.id}: {q.question_text}": q for q in st.session_state.visual_bank}
    selected_label = st.selectbox("Select Question:", list(q_options.keys()))
    question = q_options[selected_label]

    st.markdown(f"**Task:** {question.question_text}")

    try:
        # Load and resize base image
        base_img = Image.open(question.image_path)
        max_width = 600
        if base_img.width > max_width:
            ratio = max_width / base_img.width
            base_img = base_img.resize((max_width, int(base_img.height * ratio)))

        # Encode base image as base64 for the custom canvas component
        if base_img.mode in ('RGBA', 'P'):
            base_img = base_img.convert('RGB')
        buf = BytesIO()
        base_img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Render custom drawing canvas (returns base64 PNG of drawing overlay on submit)
        drawing_result = drawing_canvas(
            image_base64=img_b64,
            width=base_img.width,
            height=base_img.height,
            key=f"canvas_{question.id}"
        )

        # Process submission when drawing data is returned
        if drawing_result is not None and question.id not in st.session_state.visual_results:
            # Decode base64 PNG to numpy RGBA array (same format as old st_canvas)
            drawing_bytes = base64.b64decode(drawing_result)
            drawing_image = Image.open(BytesIO(drawing_bytes)).convert("RGBA")
            drawing_array = np.array(drawing_image)

            with st.spinner("Analyzing..."):
                result = visual_service.grade_drawing(question, drawing_array)
                st.session_state.visual_results[question.id] = result
            st.rerun()

        # Display results
        if question.id in st.session_state.visual_results:
            res = st.session_state.visual_results[question.id]
            if res.status == "Correct":
                st.success(f"✅ Correct! (Score: {res.score})")
            else:
                st.error(f"❌ Incorrect. (Score: {res.score})")
            st.info(f"**Feedback:** {res.reasoning}")

            with st.expander("View Answer Key"):
                rubric_img = Image.open(question.rubric_path)
                st.image(rubric_img, caption="Correct Answer")

    except Exception as e:
        st.error(f"Error loading image: {e}")


def render_admin_panel():
    """Password-gated admin dashboard showing query logs and topic flags."""
    from collections import Counter

    st.title("🔐 Admin Dashboard")

    # ── Password gate ────────────────────────────────────────────────────────
    if not st.session_state.get("admin_authenticated", False):
        st.subheader("Authentication Required")
        pwd = st.text_input("Enter admin password:", type="password", key="admin_pwd_input")
        if st.button("Login", key="admin_login_btn"):
            expected = st.secrets.get(ADMIN_PASSWORD_NAME, "")
            if expected and pwd == expected:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    if st.button("Logout", key="admin_logout_btn"):
        st.session_state.admin_authenticated = False
        st.rerun()

    # ── Load log ─────────────────────────────────────────────────────────────
    log = st.session_state.get("query_log", [])
    if not log:
        st.info("No queries logged yet. Chat with the assistant to populate this dashboard.")
        return

    df = pd.DataFrame(log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total = len(df)
    found = int(df["found_in_docs"].sum()) if "found_in_docs" in df.columns else 0
    not_found_count = total - found
    rag_queries = int(df["rag_mode"].sum()) if "rag_mode" in df.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Queries", total)
    m2.metric("Found in Docs", found)
    m3.metric("Not Found", not_found_count)
    m4.metric("RAG Mode Queries", rag_queries)

    st.divider()

    # ── Flagged topics ────────────────────────────────────────────────────────
    st.subheader("🚩 Flagged Topics")

    question_counts = Counter(df["question"].str.lower().str.strip().tolist())
    high_interest = [(q, c) for q, c in question_counts.items() if c >= HIGH_INTEREST_THRESHOLD]

    not_found_mask = df["found_in_docs"] == False  # noqa: E712
    not_found_df = df[not_found_mask] if not_found_mask.any() else pd.DataFrame()
    if len(not_found_df) > 0:
        gap_counts = Counter(not_found_df["question"].str.lower().str.strip().tolist())
        knowledge_gaps = [(q, c) for q, c in gap_counts.items() if c >= KNOWLEDGE_GAP_THRESHOLD]
    else:
        knowledge_gaps = []

    flag_col1, flag_col2 = st.columns(2)
    with flag_col1:
        st.markdown(f"**🔥 High Interest** *(asked {HIGH_INTEREST_THRESHOLD}+ times)*")
        if high_interest:
            for q, c in sorted(high_interest, key=lambda x: -x[1]):
                label = q[:90] + ("..." if len(q) > 90 else "")
                st.markdown(f"- `{label}` — **{c}x**")
        else:
            st.caption("No high-interest topics yet.")

    with flag_col2:
        st.markdown(f"**⚠️ Knowledge Gaps** *(not found {KNOWLEDGE_GAP_THRESHOLD}+ times)*")
        if knowledge_gaps:
            for q, c in sorted(knowledge_gaps, key=lambda x: -x[1]):
                label = q[:90] + ("..." if len(q) > 90 else "")
                st.markdown(f"- `{label}` — **{c}x not found**")
        else:
            st.caption("No knowledge gaps detected.")

    st.divider()

    # ── Query log table ───────────────────────────────────────────────────────
    st.subheader("📋 Query Log")

    fc1, fc2 = st.columns(2)
    with fc1:
        filter_found = st.selectbox(
            "Filter by result:", ["All", "Found in Docs", "Not Found"],
            key="admin_filter_found"
        )
    with fc2:
        search_term = st.text_input(
            "Search questions:", key="admin_search", placeholder="Type to filter..."
        )

    display_df = df.copy()
    if filter_found == "Found in Docs":
        display_df = display_df[display_df["found_in_docs"] == True]  # noqa: E712
    elif filter_found == "Not Found":
        display_df = display_df[display_df["found_in_docs"] == False]  # noqa: E712
    if search_term:
        display_df = display_df[
            display_df["question"].str.contains(search_term, case=False, na=False)
        ]

    display_df["Result"] = display_df["found_in_docs"].map({True: "✅ Found", False: "❌ Not Found"})
    display_df["Mode"] = display_df["rag_mode"].map({True: "RAG", False: "Open LLM"})
    display_df["Time"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["Sources"] = display_df["sources"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) and x else "—"
    )

    show_df = display_df[["Time", "question", "Result", "Mode", "Sources", "response_snippet"]].rename(
        columns={"question": "Question", "response_snippet": "Response Preview"}
    )

    st.dataframe(show_df, use_container_width=True, height=420)

    csv_data = show_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name=f"query_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="admin_csv_download",
    )

    st.divider()
    if st.button("🗑️ Clear Query Log", type="secondary", key="admin_clear_log"):
        st.session_state.query_log = []
        try:
            if os.path.exists(QUERY_LOG_FILE):
                os.remove(QUERY_LOG_FILE)
        except Exception:
            pass
        st.success("Query log cleared.")
        st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Ethos Learning Platform", page_icon="⬛", layout="wide")
    init_session_state()
    set_background()
    inject_ethos_css()

    course = get_active_course()

    # Validate configuration
    if not validate_config():
        st.stop()

    # Use cached AI provider for performance (course-aware)
    ai_provider = get_ai_provider()
    if not ai_provider:
        st.error("Missing required API key. Please check your configuration.")
        st.stop()

    # Initialize services (these are lightweight wrappers)
    quiz_service = QuizService(ai_provider)
    remediation_service = RemediationService(ai_provider)

    with st.sidebar:
        st.title(f"{course.icon} {course.name}")

        # --- RAG Master Switch ---
        st.markdown("---")
        rag_col1, rag_col2 = st.columns([1, 1])
        with rag_col1:
            rag_toggle = st.toggle(
                "RAG Mode",
                value=st.session_state.get("rag_enabled", True),
                key="rag_toggle_widget",
                help="ON = Answers from File Store / RAG documents only. OFF = Full LLM knowledge + web grounding."
            )
        with rag_col2:
            if rag_toggle:
                st.markdown('<span style="color:#4a9c55;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:0.5px">FILE STORE</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#d4940a;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:0.5px">OPEN LLM</span>', unsafe_allow_html=True)

        # Update session state if changed
        if rag_toggle != st.session_state.get("rag_enabled", True):
            st.session_state.rag_enabled = rag_toggle
            st.rerun()
        st.session_state.rag_enabled = rag_toggle
        st.markdown("---")

        # Initialize selected mode in session state
        if "selected_mode" not in st.session_state:
            st.session_state.selected_mode = "Chat 🤖"

        main_tools = ["Chat 🤖", "Quiz 📝", "Admin 🔐"]

        for tool in main_tools:
            if st.button(tool, key=f"main_{tool}", use_container_width=True,
                         type="primary" if st.session_state.selected_mode == tool else "secondary"):
                st.session_state.selected_mode = tool
                st.rerun()

        mode = st.session_state.selected_mode

        # Model settings UI
        render_model_settings()

        # Quick stats
        st.divider()
        weak_count = len(AnalyticsService.get_weak_topics())
        st.metric("Weak Topics", weak_count)

        if st.button("Reset Session"):
            switch_course(st.session_state.get("active_course", DEFAULT_COURSE_KEY))
            st.rerun()
        st.divider()
        st.caption(f"Ethos Learning Platform · v22.0 · {course.name}")

    # --- Mode indicator banner ---
    if is_rag_enabled():
        st.caption('<span style="color:#4a9c55;font-weight:600">RAG Mode: File Store</span> — Answers sourced from uploaded documents only.', unsafe_allow_html=True)
    else:
        st.caption('<span style="color:#d4940a;font-weight:600">Open LLM Mode</span> — Full model knowledge + web grounding enabled.', unsafe_allow_html=True)

    if mode == "Chat 🤖":
        st.title(f"{course.icon} {course.name} Chat")
        for m in st.session_state.messages:
            st.chat_message(m["role"]).write(m["content"])
        if p := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": p})
            st.chat_message("user").write(p)
            try:
                # Get model from session state (runtime configurable)
                chat_model = get_active_model("chat")
                rag_on = is_rag_enabled()
                system_instr = get_active_system_instruction()
                # Build history: all messages except the just-appended user message
                chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None

                if ENABLE_STREAMING:
                    # Streaming response for better UX
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_response = ""
                        for chunk in ai_provider.generate_content(
                            p, system_instr, use_rag=rag_on, stream=True,
                            model_override=chat_model, history=chat_history
                        ):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    log_query(make_query_entry(p, full_response, rag_on))
                else:
                    # Non-streaming fallback
                    with st.spinner("Thinking..."):
                        r = ai_provider.generate_content(
                            p, system_instr, use_rag=rag_on, model_override=chat_model,
                            history=chat_history
                        )
                        st.session_state.messages.append({"role": "assistant", "content": r})
                        st.chat_message("assistant").write(r)
                        log_query(make_query_entry(p, r, rag_on))
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
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.last_quiz_id = None

        source = "AI Gen"
        if source == "AI Gen":
            # Difficulty Level Selection
            st.subheader("Difficulty Level")
            difficulty = st.select_slider(
                "Select Difficulty:",
                options=DIFFICULTY_LEVELS,
                value=DIFFICULTY_MEDIUM
            )

            # Focus topic from weak topics
            focus_topic = st.session_state.get("focus_topic")
            if focus_topic:
                st.success(f"🎯 Focus Mode: Practicing **{focus_topic}**")
                if st.button("Clear Focus"):
                    st.session_state.pop("focus_topic", None)
                    st.rerun()

            st.divider()

            topics = []
            for i in range(1, 5):
                c1, c2 = st.columns([3, 1])
                # Pre-fill focus topic if set
                default_topic = focus_topic if i == 1 and focus_topic else ""
                default_pct = 100 if i == 1 and focus_topic else 0
                t = c1.text_input(f"Topic {i}", value=default_topic, key=f"topic_{i}")
                p = c2.number_input(f"%", 0, 100, default_pct, key=f"p{i}")
                if t:
                    topics.append(TopicDistribution(topic_name=t, percentage=p))

            n = st.number_input("Count", MIN_QUIZ_QUESTIONS, MAX_QUIZ_QUESTIONS, DEFAULT_QUIZ_COUNT)

            current_total = sum(t.percentage for t in topics)
            st.progress(min(current_total, 100) / 100)
            if current_total == 100:
                st.success(f"Total: {current_total}% (Valid)")
            else:
                st.warning(f"Total: {current_total}% (Target: 100%)")

            if st.button(f"Generate Quiz ({difficulty})"):
                st.info(f"Generating {difficulty} Quiz... Please wait.")
                try:
                    req = QuizRequest(num_questions=n, topics=topics, difficulty=difficulty)
                    with st.spinner("Processing Documents..."):
                        st.session_state.quiz_data = quiz_service.generate_from_ai(req)
                        st.session_state.current_quiz_source = f"AI (Standard - {difficulty})"
                    st.rerun()
                except ValueError as e:
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

            # Reset quiz answers if this is a new quiz
            if "last_quiz_id" not in st.session_state:
                st.session_state.last_quiz_id = None

            current_quiz_id = id(st.session_state.quiz_data)
            if st.session_state.last_quiz_id != current_quiz_id:
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.last_quiz_id = current_quiz_id

            with st.form("quiz"):
                score = 0
                answers = {}
                for i, q in enumerate(questions):
                    st.markdown(f"**{i+1}. {q.question}**")
                    # Restore previous answer if available
                    prev_answer = st.session_state.quiz_answers.get(i)
                    default_index = None
                    if prev_answer and prev_answer in q.options:
                        default_index = q.options.index(prev_answer)
                    answers[i] = st.radio("Opt", q.options, key=f"q{i}", label_visibility="collapsed", index=default_index)
                    st.write("---")

                if st.form_submit_button("Submit"):
                    # Persist answers
                    st.session_state.quiz_answers = answers
                    st.session_state.quiz_submitted = True

                    current_incorrect = []
                    for i, q in enumerate(questions):
                        correct = q.correct_answer
                        user = answers.get(i)
                        is_correct = user == correct

                        # Record for analytics (Weak Topic Identification)
                        # Pass the full reference - AnalyticsService will extract the proper topic
                        AnalyticsService.record_answer(q.reference, is_correct)

                        if is_correct:
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
                    
                    # Remediation difficulty - default to Easy since user struggled
                    rem_difficulty = st.select_slider(
                        "Remediation Difficulty:",
                        options=DIFFICULTY_LEVELS,
                        value=DIFFICULTY_EASY,
                        key="rem_difficulty",
                        help="Remediation defaults to Easy to reinforce fundamentals"
                    )

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
                                topics.append(TopicDistribution(topic_name=f"General {get_active_course().subject_label}", percentage=100))

                            req = QuizRequest(num_questions=rem_n, topics=topics, difficulty=rem_difficulty)

                            with st.spinner("Generating targeted practice quiz..."):
                                new_quiz = quiz_service.generate_from_ai(req)

                                st.session_state.quiz_data = new_quiz
                                st.session_state.current_quiz_source = f"Remediation (Standard - {rem_difficulty})"
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

    elif mode == "Admin 🔐":
        render_admin_panel()


if __name__ == "__main__":
    main()
