import streamlit as st

st.set_page_config(page_title="Debug Mode")

st.title("✅ Interface is Working")
st.write("If you can see this, Streamlit is functioning correctly.")

# Check for background file existence only (don't load it yet)
import os
if os.path.exists("background.png"):
    st.success("Found 'background.png' in the folder.")
else:
    st.error("Could not find 'background.png'.")

# Placeholder for where the chat would be
st.info("Chat module is currently disabled for testing.")