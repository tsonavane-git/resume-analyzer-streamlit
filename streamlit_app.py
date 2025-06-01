import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load small LLM
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm = load_model()

st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")
st.title("📄 Smart Resume Analyzer (Free LLM Version)")

st.write("Upload a resume and get instant insights, no API key required!")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_title = st.text_input("Target Job Title (e.g., UI/UX Designer)")

if uploaded_file and job_title:
    with st.spinner("Reading your resume..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        resume_text = "\n".join(page.get_text() for page in doc)

    st.success("Resume extracted!")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing..."):
            prompt = f"Analyze this resume for the job role: {job_title}. Highlight strengths, weaknesses, and give suggestions.\n\nResume:\n{resume_text}"

            try:
                response = llm(prompt, max_new_tokens=256)[0]['generated_text']
                st.subheader("📊 Analysis Result")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a resume and enter a job title to begin.")

st.markdown("---")
st.caption("Lightweight LLM via Hugging Face – No API key needed.")
