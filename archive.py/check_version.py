import streamlit as st
import sys
import google.generativeai as genai

st.title("🕵️ Environment Detective")

# 1. Check the Library Version
try:
    version = genai.__version__
    st.metric(label="Google GenAI Version", value=version)
    if version < "0.5.2":
        st.error(f"❌ Version is too old! You need at least 0.5.2. You have {version}")
    else:
        st.success("✅ Version looks good.")
except AttributeError:
    st.error("❌ Could not determine version (Library is likely very old or broken).")

# 2. Check WHERE Streamlit is running from
st.write("---")
st.subheader("Where am I running?")
st.code(f"Python Executable: {sys.executable}")

st.info("""
**How to fix:**
Copy the path above (Python Executable). 
Run the install command using that specific python path.
""")