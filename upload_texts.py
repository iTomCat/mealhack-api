import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ładowanie plików tekstowych

load_dotenv()


PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base_dev")


def upload_knowledge():
    print(f"🚀 Starting upload text files to: {COLLECTION_NAME}...")

    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004", project=PROJECT_ID)

    # Ładujemy dwa konkretne pliki tekstowe
    loaders = [
        TextLoader("dane/zalecenia_ogolne_2020.txt", encoding='utf-8'),
        TextLoader("dane/protokol_io_2024.txt", encoding='utf-8')
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Dzielimy na fragmenty
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Wysyłamy do Firestore
    # Uwaga: To DODAJE dokumenty do istniejącej kolekcji.
    # Jeśli chcesz mieć czysty test, możesz zmienić nazwę kolekcji w .env na np. "test_konfliktu"
    FirestoreVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=COLLECTION_NAME
    )

    print(f"✅ Success! Uploaded {len(chunks)} chunks to {COLLECTION_NAME}")


if __name__ == "__main__":
    upload_knowledge()
