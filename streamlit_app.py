import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="Smart Resume & LinkedIn Analyzer", layout="centered")

# Load Falcon model (chat-style)
@st.cache_resource
def load_model():
    model_id = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

llm = load_model()

st.title("📄 Smart Resume & LinkedIn Analyzer (Free LLM)")
tab1, tab2 = st.tabs(["📄 Resume (PDF)", "🔗 LinkedIn Profile (Text)"])

# === Resume Analyzer ===
with tab1:
    st.header("📄 Resume Analyzer")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_title = st.text_input("Target Job Title", placeholder="e.g., Software Engineer")

    if uploaded_file and job_title:
        with st.spinner("Extracting resume text..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = "\n".join(page.get_text() for page in doc)

        st.success("Resume parsed!")

        if st.button("🔍 Analyze Resume"):
            with st.spinner("Analyzing..."):
                prompt = f"""You are a career advisor. Analyze the following resume for the job title "{job_title}".

Provide:
1. Summary
2. Strengths
3. Weaknesses
4. Suggestions for improvement

Resume:
{resume_text}
"""
                try:
                    response = llm(prompt)[0]['generated_text']
                    st.subheader("📊 Analysis Result")
                    st.write(response.split("Resume:")[-1].strip())
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload a resume and enter a target job title.")

# === LinkedIn Profile Analyzer ===
with tab2:
    st.header("🔗 LinkedIn Profile Analyzer")
    profile_text = st.text_area("Paste your LinkedIn Summary/About section:")
    linkedin_job = st.text_input("Target Job Title", placeholder="e.g., Marketing Manager", key="linkedin_job")

    if profile_text and linkedin_job:
        if st.button("🧠 Analyze LinkedIn"):
            with st.spinner("Analyzing LinkedIn profile..."):
                prompt = f"""You are a LinkedIn profile expert.

Analyze the following summary for the job title "{linkedin_job}". Evaluate:
- Clarity and professionalism
- Alignment with the role
- Strengths and weaknesses
- Suggestions for rewriting

LinkedIn Summary:
{profile_text}
"""
                try:
                    response = llm(prompt)[0]['generated_text']
                    st.subheader("🔍 LinkedIn Analysis Result")
                    st.write(response.split("LinkedIn Summary:")[-1].strip())
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Paste LinkedIn summary and target job title.")

st.markdown("---")
st.caption("🚀 Powered by Hugging Face Falcon 7B – No API key required.")
