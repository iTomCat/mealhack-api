import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# UWAGA
# zeby wysłać do _prod trzeba w termianlu wpisać:
# COLLECTION_NAME=knowledge_base_prod python upload.py

# Konfiguracja z .env
PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # knowledge_base_dev


def upload_knowledge():
    print(f"🚀 Starting upload to collection: {COLLECTION_NAME}...")

    # 1. Inicjalizacja Embeddings (Model 004 jest najnowszy i stabilny)
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004", project=PROJECT_ID)

    # 2. Ładowanie Twojego PDF-a o odżywianiu
    # Upewnij się, że plik o tej nazwie jest w Twoim folderze mealhack-app
    loader = PyPDFLoader("dane/nutrition_data.pdf")
    docs = loader.load()

    # 3. Dzielenie tekstu na mniejsze fragmenty
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 4. Wysyłka do Firestore (Wektoryzacja)
    # Wykorzystujemy uprawnienia IAM, które nadaliśmy w konsoli
    _ = FirestoreVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=COLLECTION_NAME
    )

    print(f"✅ Success! Knowledge uploaded to: {COLLECTION_NAME}")


if __name__ == "__main__":
    upload_knowledge()
