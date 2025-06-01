import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")

# Load small LLM
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm = load_model()

st.title("📄 Smart Resume & LinkedIn Analyzer (No API Key)")

tab1, tab2 = st.tabs(["📄 Resume (PDF)", "🔗 LinkedIn Profile (Text)"])

# --- Resume Analysis Tab ---
with tab1:
    st.header("📄 Resume Analyzer")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_title = st.text_input("Target Job Title (e.g., Software Engineer)")

    if uploaded_file and job_title:
        with st.spinner("Reading your resume..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = "\n".join(page.get_text() for page in doc)

        st.success("Resume parsed!")

        if st.button("🔍 Analyze Resume"):
            with st.spinner("Analyzing..."):
                prompt = f"""Analyze this resume for the job role: {job_title}.
Highlight strengths, weaknesses, and give improvement suggestions.

Resume:
{resume_text}
"""
                try:
                    response = llm(prompt, max_new_tokens=256)[0]['generated_text']
                    st.subheader("📊 Analysis Result")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload a resume and enter a job title to begin.")

# --- LinkedIn Profile Analyzer Tab ---
with tab2:
    st.header("🔗 LinkedIn Profile Analyzer")
    profile_text = st.text_area("Paste your LinkedIn 'About' or Summary section:")

    linkedin_job = st.text_input("Job Role You're Targeting (e.g., Project Manager)", key="linkedin_job")

    if profile_text and linkedin_job:
        if st.button("🧠 Analyze LinkedIn Profile"):
            with st.spinner("Analyzing LinkedIn summary..."):
                prompt = f"""Evaluate this LinkedIn profile summary for the job role: {linkedin_job}.
Assess clarity, tone, strengths, weaknesses, and give suggestions for improvement.

LinkedIn Summary:
{profile_text}
"""
                try:
                    response = llm(prompt, max_new_tokens=256)[0]['generated_text']
                    st.subheader("🔍 LinkedIn Analysis Result")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Paste your LinkedIn summary and enter a job role.")

st.markdown("---")
st.caption("No API key needed. Powered by Hugging Face and Streamlit.")
