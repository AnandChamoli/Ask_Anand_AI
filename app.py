import streamlit as st
from rag_engine import answer_question
from jd_match import analyze_job_description

st.set_page_config(
    page_title="Ask Anand AI",
    page_icon="🤖",
    layout="wide"
)


col1, col2 = st.columns([1, 4])

with col1:
    st.image("profile.jpeg", width=220)

with col2:
    st.title("Ask Anand AI - AI-Powered Career & Portfolio Assistant")

    st.markdown("""
**Anand Chamoli**

Business Analytics Professional with 14 years of experience across Sales, Marketing, Customer Service, After-Sales Operations, Spare Parts, and Business Transformation.

🎓 B.S. Data Science (Business Analytics) – Arizona State University
🎓 Diploma in Automobile Engineering (CSMT, B.S.F)

💻 Python | SQL | Power BI | Tableau | Machine Learning

🚜 Sales | Marketing | After-Sales | Spare Parts | Customer Experience | Business Transformation""")

st.markdown(
    "[🔗 LinkedIn](https://www.linkedin.com/in/anand-chamoli-3702bb8b) | "
    "[💻 GitHub](https://github.com/AnandChamoli)"
)

st.markdown(
    "Explore Anand's professional experience, analytics projects, leadership accomplishments, and suitability for various business and analytics roles."
)

# -----------------------------
# Ask Anand Q&A Section
# -----------------------------

# -----------------------------
# Ask Anand Q&A Section
# -----------------------------

sample_questions = [
    "Why should we hire Anand?",
    "Tell me about Anand's tractor industry experience?",
    "Tell me about CARE Direct 24x7.",
    "What was Anand's role in CARE Direct 24x7?",
    "How does Anand's tractor industry experience help in analytics roles?",
    "What business problems can Anand solve for our company?",
    "What analytics projects has Anand completed?",
    "Is Anand suitable for Business Analyst roles?",
    "What makes Anand different from a fresh data analyst?",
    "What roles is Anand best suited for?"
]

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


with st.form("ask_anand_form"):
    question = st.text_input(
        "Ask a question:",
        placeholder="Example: Why should we hire Anand?"
    )

    selected_question = st.selectbox(
        "Or choose a sample question:",
        [""] + sample_questions
    )

    submitted = st.form_submit_button("Ask Anand")

if submitted:
    final_question = question.strip() if question.strip() else selected_question.strip()

    if final_question:
        with st.spinner("Searching Anand's knowledge base..."):
            answer = answer_question(final_question)

        st.session_state.last_question = final_question
        st.session_state.last_answer = answer
    else:
        st.warning("Please enter or select a question.")

if st.session_state.last_answer:
    st.markdown("### Question")
    st.write(st.session_state.last_question)

    st.markdown("### Answer")
    st.write(st.session_state.last_answer)
# -----------------------------
# Job Description Match Section
# -----------------------------

st.markdown("---")

st.subheader("Job Fit Analysis")

job_description = st.text_area(
    "Paste a Job Description",
    height=200
)

if st.button("Analyze Match"):
    if job_description.strip():
        score, matched_skills, missing_skills = analyze_job_description(job_description)

        st.metric("Match Score", f"{score}%")

        st.markdown("### Matched Skills")
        for skill in matched_skills:
            st.write(f"✅ {skill}")

        st.markdown("### Possible Gaps")
        for skill in missing_skills[:10]:
            st.write(f"⚠️ {skill}")
    else:
        st.warning("Please paste a job description first.")

st.markdown("---")
st.caption(
    "Built by Anand Chamoli using | Python, Streamlit, ChromaDB, SentenceTransformers, Retrieval-Augmented Generation (RAG)"
)
