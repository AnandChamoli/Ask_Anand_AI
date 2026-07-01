# 🤖 Ask Anand AI

### AI-Powered Career & Portfolio Assistant

The application enables recruiters and hiring managers to explore professional experience, analytics projects, leadership achievements, and job-role fit through natural language conversations and Job Description matching.

---

## 🚀 Live Demo

🔗 Live Application: https://askanandai-qkfzbvtwkmld2ko7b38nrb.streamlit.app/

💡 Best viewed on desktop. Try asking questions about experience, analytics projects, leadership, or role fit.

🔗 GitHub Repository: https://github.com/AnandChamoli/Ask_Anand_AI

## Run Locally

```bash
git clone ...
cd ...
pip install ...
streamlit run app.py
```
---

## Key Highlights

- Built using Retrieval-Augmented Generation (RAG)
- Semantic search powered by Sentence Transformers
- Recruiter-focused portfolio assistant
- Job Description matching and skill-gap analysis
- Deployed on Streamlit Cloud

---

## Screenshots

### Home Page

<img width="1426" height="754" alt="homepage" src="https://github.com/user-attachments/assets/118d2648-91b1-4d51-abbe-606b073e4f9c" />

### Recruiter Q&A

<img width="895" height="644" alt="recruiter_question" src="https://github.com/user-attachments/assets/30c4fb68-03e4-4514-9345-0956c2f75a26" />

### Job Fit Analysis

<img width="1414" height="788" alt="job_fit_analysis" src="https://github.com/user-attachments/assets/c9e0f6c3-8a44-4878-9eae-d0e52f9d6055" />

---

## 🎯 Project Objective

Traditional resumes provide static information.

Ask Anand AI transforms my professional profile into an interactive AI assistant that can answer questions about:

- Professional experience
- Leadership accomplishments
- Analytics projects
- Industry expertise
- Job-role suitability
- Technical and business skills

---

## ✨ Features

### 🔍 AI-Powered Question Answering

Interact with the assistant using natural language to explore professional experience, analytics projects, leadership achievements, and domain expertise.

---

### 📊 Job Description Match Analysis

Paste any Job Description and receive:

- Match Score
- Matched Skills
- Potential Skill Gaps

This helps evaluate role fit and identify improvement opportunities.

---


## 🏗️ Architecture

```text
User Question
      │
      ▼
Sentence Transformer
(all-MiniLM-L6-v2)
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Knowledge Chunk
      │
      ▼
Generated Response
```

---

## 🛠️ Technology Stack

### AI & Machine Learning

- Sentence Transformers
- Embeddings
- Retrieval-Augmented Generation (RAG)
- Semantic Search

### Data & Storage

- ChromaDB
- Vector Database

### Application

- Python
- Streamlit

### Deployment

- GitHub
- Streamlit Community Cloud

---

## 📁 Project Structure

```text
Ask_Anand_AI/
│
├── app.py
├── rag_engine.py
├── ingest.py
├── jd_match.py
├── requirements.txt
│
├── KnowledgeBase/
└── vector_db/
```

---

## 📈 Sample Questions

### Career & Experience

- Why should we hire Anand?
- Tell me about Anand's tractor industry experience.
- What roles is Anand best suited for?

### Analytics

- What analytics projects has Anand completed?
- Does Anand have machine learning experience?

### Leadership

- Tell me about CARE Direct 24x7.
- What leadership experience does Anand have?

---

## 🔮 Future Enhancements

Planned improvements include:

- LLM-powered grounded response generation
- Voice assistant integration
- Interview scheduling capability
- Enhanced job-match analytics

---

## 👤 Author

### Anand Chamoli

Business Analytics Professional | Data Science Graduate | AI Enthusiast

🔗 LinkedIn:
https://www.linkedin.com/in/anand-chamoli-3702bb8b

🔗 GitHub:
https://github.com/AnandChamoli

---

## ⭐ If you found this project interesting

Please feel free to connect with me on LinkedIn or explore my other analytics and machine learning projects.
