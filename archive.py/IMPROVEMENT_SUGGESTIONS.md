# Code Improvement Suggestions for voice_2.py

These suggestions improve code quality, maintainability, and user experience **without breaking any existing functionality**.

## 1. Extract Duplicate Retry Logic (Code Reusability)

**Current Issue**: `_generate_with_retry` is duplicated in `QuizService`, `RemediationService`, and `OralBoardService`.

**Suggestion**: Create a shared utility function or base class:

```python
# Add to SERVICE LAYER section
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
    """
    try:
        return ai_provider.generate_content(prompt, SYSTEM_INSTRUCTION, use_rag=True)
    except Exception as e:
        if "RECITATION_ERROR" in str(e):
            logger.warning(f"{error_context} hit Recitation error. Retrying with safe prompt.")
            try:
                return ai_provider.generate_content(safe_prompt, SYSTEM_INSTRUCTION, use_rag=True)
            except Exception as e2:
                if "RECITATION_ERROR" in str(e2):
                    logger.error(f"{error_context}: Safe prompt also hit RECITATION_ERROR.")
                    raise ValueError("RECITATION_ERROR")
                raise e2
        raise e
```

Then update each service to use this shared function instead of their own `_generate_with_retry`.

## 2. Fix Session State Reset (Bug Fix)

**Current Issue**: Reset Session button doesn't reset oral board state.

**Location**: Line ~1041-1046

**Fix**:
```python
if st.button("Reset Session"):
    st.session_state.messages = []
    st.session_state.quiz_data = None
    st.session_state.incorrect_questions = []
    st.session_state.remediation_text = None
    st.session_state.oral_scenario = None  # ADD THIS
    st.session_state.oral_grade = None    # ADD THIS
    st.rerun()
```

## 3. Update Version Number

**Current**: Line 1048 shows "v15.0 - Optimized Performance"

**Suggestion**: Update to reflect Oral Board feature:
```python
st.caption("✅ v16.0 - Oral Board Active")
```

## 4. Extract Safety Settings to Constant

**Current Issue**: Safety settings are repeated in multiple places.

**Suggestion**: Add to CONSTANTS section:
```python
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
```

Then use `SAFETY_SETTINGS` in `_get_config` and `generate_content` methods.

## 5. Extract Audio Reading Logic

**Current Issue**: Audio reading logic is duplicated in `render_oral_board`.

**Suggestion**: Create a helper function:
```python
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
```

## 6. Improve Type Hints

**Suggestions**:
- Add return type hints where missing
- Use `Optional[bytes]` instead of just `bytes` where None is possible
- Add type hints to function parameters that are missing them

## 7. Add Input Validation Helper

**Suggestion**: Create a validation helper for audio:
```python
def validate_audio_bytes(audio_bytes: Optional[bytes], min_size: int = 100) -> tuple[bool, Optional[str]]:
    """
    Validate audio bytes.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not audio_bytes:
        return False, "No audio data received. Please record your answer again."
    
    if len(audio_bytes) < min_size:
        return False, f"Audio file is very small ({len(audio_bytes)} bytes). Please record a longer response."
    
    # Validate WAV header if needed
    if len(audio_bytes) >= 12:
        if not (audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'):
            logger.warning("Audio file may not be a valid WAV format")
    
    return True, None
```

## 8. Improve Error Messages

**Suggestion**: Make error messages more user-friendly and actionable:
- Instead of "System Error: {e}", use "We encountered an issue. Please try again or contact support if the problem persists."
- Add specific guidance for common errors

## 9. Add Configuration Validation

**Suggestion**: Validate configuration at startup:
```python
def validate_config() -> bool:
    """Validate that all required configuration is present."""
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
```

## 10. Add Request Rate Limiting Awareness

**Suggestion**: Add comments/documentation about API rate limits and consider adding delays if needed:
```python
# Note: Google GenAI has rate limits. For high-volume usage, consider:
# - Adding delays between requests
# - Implementing exponential backoff
# - Caching responses where appropriate
```

## 11. Improve Progress Bar Check

**Current**: Line 822 uses `'st' in globals()` which is fragile.

**Suggestion**: Use a more reliable check:
```python
# Check if we're in Streamlit context
try:
    import streamlit as st
    IN_STREAMLIT = True
except ImportError:
    IN_STREAMLIT = False

# Then use:
progress_bar = st.progress(0) if IN_STREAMLIT else None
```

## 12. Add Constants for Magic Numbers

**Suggestions**:
```python
MIN_AUDIO_SIZE_BYTES = 100
WAV_HEADER_SIZE = 12
AUDIO_TEMPERATURE = 0.2  # For audio grading
```

## 13. Improve Logging Context

**Suggestion**: Add more context to log messages:
```python
logger.info(f"[{error_context}] Processing request: {len(prompt)} chars")
```

## 14. Add Docstring Improvements

**Suggestion**: Ensure all public methods have comprehensive docstrings with:
- Clear description
- Args section
- Returns section
- Raises section (where applicable)
- Examples (for complex methods)

## 15. Consider Adding Request Timeout Handling

**Suggestion**: Add timeout handling for long-running requests:
```python
# In generate_content, consider adding timeout parameter usage
# Currently defined but not used
```

## Priority Recommendations

**High Priority** (Bug fixes, important improvements):
1. Fix session state reset (#2)
2. Update version number (#3)
3. Extract duplicate retry logic (#1)

**Medium Priority** (Code quality, maintainability):
4. Extract safety settings constant (#4)
5. Extract audio reading logic (#5)
6. Add input validation helper (#7)

**Low Priority** (Nice to have):
7. Improve type hints (#6)
8. Improve error messages (#8)
9. Add configuration validation (#9)

## Implementation Notes

- All suggestions maintain backward compatibility
- No breaking changes to existing functionality
- All improvements are additive or refactoring
- Test thoroughly after implementing any changes
