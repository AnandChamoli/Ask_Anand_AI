import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "ask_anand"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_context(query, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


def answer_question(query, n_results=5):
    results = retrieve_context(query, n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    answer = f"""
Question:
{query}

Answer:
{documents[0].strip()}

Supporting Evidence:
"""

    for i, doc in enumerate(documents[:3], 1):
        source = metadatas[i - 1].get("source", "Unknown source")
        answer += f"\n--- Evidence {i} ---\n"
        answer += doc.strip()[:600] + "\n"
        answer += f"Source: {source}\n"

    return answer


if __name__ == "__main__":
    print(answer_question("Does Anand have GCP experience?"))

