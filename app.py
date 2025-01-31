import streamlit as st

st.set_page_config(
    page_title="Biomarker Analysis Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Biomarker Analysis Platform")
st.markdown("""
    Welcome to the Biomarker Analysis Platform. This application provides two main functionalities:
    
    1. **Classification**: Analyze biomarkers and clinical data for diagnosis classification
    2. **Progression Detection**: Predict disease progression based on biomarker patterns
    
    Please select a page from the sidebar to get started.
""")