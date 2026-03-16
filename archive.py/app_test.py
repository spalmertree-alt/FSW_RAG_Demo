import streamlit as st
from google import genai
from google.genai import types
import os
import base64

# --- CONFIGURATION ---
# Replace with your actual API Key and Store ID
API_KEY = "AIzaSyCPGJYKe9UkxtVL6W1YkQhuausYzeG5_lU"
STORE_ID = "fileSearchStores/maritime-rules-of-the-road--gfijzkkpff4o" 

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Class Tutor", layout="centered")

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
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')

# --- GEMINI CLIENT SETUP ---
@st.cache_resource
def get_client():
    if not API_KEY or "PASTE_YOUR" in API_KEY:
        st.error("CRITICAL: Check your API Key line in the code.")
        return None

    try:
        # Use the newer google.genai client which supports file search
        client = genai.Client(api_key=API_KEY)
        return client

    except Exception as e:
        st.error(f"Client Initialization Failed: {str(e)}")
        return None

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    client = get_client()
    if client:
        st.session_state.client = client
        st.success("✅ AI Tutor Connected!")
    else:
        st.session_state.client = None

# --- UI LAYOUT ---
st.title("📚 Course AI Tutor")

# Debug toggle (optional, can be removed in production)
with st.sidebar:
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode, help="Show detailed response information")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.client:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # --- UPDATED: Build History for Context ---
                    # We map Streamlit's "assistant" role to Gemini's "model" role
                    # and pass the entire list to the API.
                    history_for_api = []
                    for msg in st.session_state.messages:
                        role = "model" if msg["role"] == "assistant" else "user"
                        history_for_api.append({"role": role, "parts": [msg["content"]]})

                    # Define system instruction separately (best practice for history)
                    sys_instruction = (
                        "You are a Maritime Rules of the Road expert. "
                        "CRITICAL: You MUST use the File Search tool to find information in the provided documents. "
                        "Do NOT answer from general knowledge. "
                        "You MUST search the documents for every question. "
                        "Always cite specific Rule numbers, Annex sections, or document references. "
                        "If you cannot find the answer in the documents, state clearly that the information is not available in the provided documents."
                    )
                    
                    # Generate response using history
                    response = st.session_state.client.models.generate_content(
                        model="gemini-1.5-flash", # Note: 2.5 doesn't exist yet, using 1.5 to prevent 404
                        contents=history_for_api, # Passing the full history list here
                        config=types.GenerateContentConfig(
                            system_instruction=sys_instruction, # System prompt goes here now
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
                    
                    response_text = response.text
                    st.markdown(response_text)
                    
                    # Debug mode: Show response structure
                    if 'debug_mode' in st.session_state and st.session_state.debug_mode:
                        with st.expander("🔍 Debug: Response Structure", expanded=False):
                            st.json({
                                "has_candidates": bool(response.candidates),
                                "candidate_count": len(response.candidates) if response.candidates else 0,
                                "candidate_attrs": dir(response.candidates[0]) if response.candidates and len(response.candidates) > 0 else [],
                                "citation_metadata_exists": hasattr(response.candidates[0], 'citation_metadata') if response.candidates and len(response.candidates) > 0 else False,
                                "grounding_metadata_exists": hasattr(response.candidates[0], 'grounding_metadata') if response.candidates and len(response.candidates) > 0 else False,
                            })
                    
                    # Extract and display citation metadata
                    citations = []
                    try:
                        if response.candidates and len(response.candidates) > 0:
                            candidate = response.candidates[0]
                            
                            # Check citation_metadata first
                            if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                                metadata = candidate.citation_metadata
                                if hasattr(metadata, 'citation_sources') and metadata.citation_sources:
                                    citations = list(metadata.citation_sources)
                            
                            # Also check for grounding_metadata (alternative field)
                            if not citations and hasattr(candidate, 'grounding_metadata'):
                                grounding = candidate.grounding_metadata
                                if grounding and hasattr(grounding, 'grounding_chunks'):
                                    # Convert grounding chunks to citation format
                                    for chunk in grounding.grounding_chunks:
                                        if hasattr(chunk, 'file') and hasattr(chunk.file, 'uri'):
                                            citations.append(chunk.file)
                            
                            # Check if content parts have grounding metadata
                            if not citations and hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'grounding_metadata'):
                                        grounding = part.grounding_metadata
                                        if grounding and hasattr(grounding, 'grounding_chunks'):
                                            for chunk in grounding.grounding_chunks:
                                                if hasattr(chunk, 'file') and hasattr(chunk.file, 'uri'):
                                                    citations.append(chunk.file)
                    except Exception as e:
                        # Debug: uncomment to see what's available
                        # st.write(f"Debug citation error: {e}")
                        pass
                    
                    # Display citations if available
                    if citations:
                        st.markdown("---")
                        with st.expander(f"📚 Sources & References ({len(citations)})", expanded=True):
                            for idx, citation in enumerate(citations, start=1):
                                # Extract filename from URI if possible
                                uri = citation.uri if hasattr(citation, 'uri') else str(citation)
                                # Try to get a clean filename
                                filename = uri.split('/')[-1] if '/' in uri else uri
                                # Remove file extension for cleaner display
                                display_name = filename.replace('.pdf', '').replace('_', ' ').title()
                                st.markdown(f"**{idx}.** {display_name}")
                                if hasattr(citation, 'start_index') and hasattr(citation, 'end_index'):
                                    if citation.start_index is not None and citation.end_index is not None:
                                        st.caption(f"Referenced in answer (positions {citation.start_index}-{citation.end_index})")
                    else:
                        # Check if file search was actually used by inspecting the response
                        file_search_used = False
                        try:
                            if response.candidates and len(response.candidates) > 0:
                                candidate = response.candidates[0]
                                # Check if content parts indicate tool usage
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        # Check for function call or tool usage indicators
                                        if hasattr(part, 'function_call') or hasattr(part, 'file_data'):
                                            file_search_used = True
                                            break
                        except:
                            pass
                        
                        if file_search_used:
                            st.caption("ℹ️ *File search was used, but citation metadata format may differ*")
                        else:
                            st.caption("⚠️ *Note: File search may not have been triggered. The model may be using general knowledge.*")
                    
                    # Add assistant response to conversation history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")