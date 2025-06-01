import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# === Load Hugging Face model once ===
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=512)

llm = load_model()

st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")

st.title("📄 Smart Resume Analyzer (Offline LLM Version)")
st.write("Upload your resume and get instant feedback using a free local LLM.")

# === Upload PDF Resume ===
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_title = st.text_input("Target Job Title (e.g., Software Engineer)")

if uploaded_file and job_title:
    with st.spinner("Extracting resume content..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        resume_text = "\n".join(page.get_text() for page in doc)

    st.success("Resume uploaded and parsed!")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Generating analysis..."):
            prompt = f"""### Instruction:
You are a resume review expert.

Analyze the following resume for the job title "{job_title}". Provide:
1. Summary
2. Strengths
3. Weaknesses
4. Suggestions for improvement

### Resume:
{resume_text}

### Response:
"""
            try:
                response = llm(prompt)[0]['generated_text'].split("### Response:")[-1].strip()
                st.subheader("📊 Analysis Result")
                st.write(response)
            except Exception as e:
                st.error(f"LLM Error: {str(e)}")
else:
    st.info("Please upload a resume and enter a job title.")

st.markdown("---")
st.caption("No API key needed. Powered by Hugging Face + Streamlit.")
