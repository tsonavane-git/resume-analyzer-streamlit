import streamlit as st
import fitz  # PyMuPDF
import openai
import os

# === OpenAI API Key ===
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")

st.title("📄 Smart Resume Analyzer")
st.write("Upload your resume and get instant feedback based on the job title you specify.")

# === Upload Resume ===
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_title = st.text_input("Target Job Title (e.g., Data Scientist)")

if uploaded_file and job_title:
    with st.spinner("Reading your resume..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        resume_text = "\n".join(page.get_text() for page in doc)

    st.success("Resume uploaded and parsed successfully!")

    # === Prompt the LLM ===
    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing with LLM..."):
            prompt = f"""
You are a resume review assistant.

Evaluate this resume for the job title: {job_title}.
Provide:
1. Overall summary
2. Strengths
3. Weaknesses
4. Specific improvement suggestions

Resume Content:
{resume_text}
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=800,
                )
                st.subheader("📊 Analysis Result")
                st.write(response['choices'][0]['message']['content'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("Please upload a resume and enter a job title.")

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and OpenAI")
