import os
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


KNOWLEDGE_BASE_PATH = "KnowledgeBase"
VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "ask_anand"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def read_pdf_file(file_path):
    if PdfReader is None:
        return ""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx_file(file_path):
    if docx is None:
        return ""
    document = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])


def load_documents():
    documents = []

    for root, _, files in os.walk(KNOWLEDGE_BASE_PATH):
        for file_name in files:
            if file_name.startswith("."):
                continue

            file_path = Path(root) / file_name
            extension = file_path.suffix.lower()

            if extension in [".txt", ".md"]:
                text = read_text_file(file_path)
            elif extension == ".pdf":
                text = read_pdf_file(file_path)
            elif extension == ".docx":
                text = read_docx_file(file_path)
            else:
                continue

            if text.strip():
                documents.append({
                    "text": text,
                    "source": str(file_path)
                })

    return documents


def create_chunks(documents):

    chunks = []

    for document in documents:

        # Special handling for recruiter Q&A files
        if "---" in document["text"]:

            qa_chunks = text.split("---")

            for index, qa in enumerate(qa_chunks):

                qa = qa.strip()

                if qa:
                    chunks.append({
                        "text": qa,
                        "source": document["source"],
                        "chunk_id": index
                    })

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=120
            )

            split_texts = splitter.split_text(document["text"])

            for index, chunk_text in enumerate(split_texts):

                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "chunk_id": index
                })

    return chunks


def build_vector_db():
    print("Loading documents...")
    documents = load_documents()
    print(f"Documents loaded: {len(documents)}")

    print("Creating chunks...")
    chunks = create_chunks(documents)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created. Check your KnowledgeBase files.")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Rebuilding vector database...")
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Ask Anand AI Resume RAG Knowledge Base"}
    )

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"chunk_{index}")
        texts.append(chunk["text"])
        embeddings.append(embedding_model.encode(chunk["text"]).tolist())
        metadatas.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        })

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Vector database created successfully.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_db()
