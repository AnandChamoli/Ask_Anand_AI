import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "ask_anand"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_context(query, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


def clean_text(text):
    return " ".join(text.replace("\n", " ").split())


def clean_answer(text):
    text = clean_text(text)

    headings_to_remove = [
        "Recruiter Q&A",
        "Recruiter Questions",
        "Recruiter Q&A - Part 2 (Technical & Analytics)",
        "Recruiter Q&A - Part 3 (Industry Expertise, Career Transition & Role Fit)",
        "Expanded Recruiter Q&A",
        "CARE Direct 24x7 Q&A",
        "Project Questions"
    ]

    for heading in headings_to_remove:
        text = text.replace(heading, "")

    if "A:" in text:
        text = text.split("A:", 1)[1].strip()

    if "---" in text:
        text = text.split("---", 1)[0].strip()

    if "## Q:" in text:
        text = text.split("## Q:", 1)[0].strip()

    return text.strip()


def answer_question(query, n_results=5):
    results = retrieve_context(query, n_results)

    documents = results["documents"][0]

    if not documents:
        return "No relevant information found."

    # Retrieval-only version:
    # Use only the best matching chunk to avoid mixing unrelated answers.
    best_answer = clean_answer(documents[0])

    if best_answer:
        return best_answer

    return "No relevant information found."


if __name__ == "__main__":
    test_questions = [
        "Who is Anand Chamoli?",
        "Does Anand have GCP experience?",
        "Why should we hire Anand?",
        "Tell me about CARE Direct 24x7.",
        "Tell me about Anand's Customer Churn Prediction project.",
        "Tell us about Anand's career transition journey?",
        "What analytics projects has Anand completed?"
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print("Question:", question)
        print("Answer:", answer_question(question))
