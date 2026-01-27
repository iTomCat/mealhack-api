import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.cloud import firestore

# ------------------------------------------------------
# Łdowanie do RAG z kompletu plików PDF
# UWAGA: Ten skrypt najpierw CZYŚCI kolekcję knowledge_base_dev, jezli jest i wczytuje wszystko od nowa
# potrzebnych do opisu tabel indeksu glikemicznego, klasyfikacji NOVA itp.
# ------------------------------------------------------
# Uruchamiamy: python upload_rag_pdfs.py

# Ładowanie zmiennych środowiskowych
load_dotenv()

# getenv - pobieramy zmienne z .env
PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base_dev")
DATABASE_NAME = os.getenv("DATABASE_NAME", "meal-base")
DATA_FOLDER = "dane_rag"

FILES_TO_UPLOAD = [
    "Tabela1_IG.pdf",
    "Tabela2_IG.pdf",
    "NOVA-Classification-Reference-Sheet.pdf",
    "nutrients-12-02502.pdf"
]


def delete_all_documents(client):
    print(f"🗑️  CZYSZCZENIE kolekcji '{COLLECTION_NAME}'...")
    coll_ref = client.collection(COLLECTION_NAME)

    deleted_count = 0
    while True:
        docs = list(coll_ref.limit(100).stream())
        if not docs:
            break
        batch = client.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted_count += len(docs)
        print(f"   ...usunięto {deleted_count}...")
    print("✨ Kolekcja wyczyszczona.\n")


def process_and_upload_file(filename, client, embeddings, text_splitter):
    file_path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ Brak pliku: {filename}")
        return

    print(f"🔄 Przetwarzanie: {filename}...")

    # 1. Load
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"❌ Błąd ładowania {filename}: {e}")
        return

    # 2. Metadata Fix
    for doc in docs:
        doc.metadata['source_file'] = filename
        # Usuwamy wszystko inne, żeby było czysto
        to_remove = [k for k in doc.metadata.keys() if k != 'source_file']
        for k in to_remove:
            del doc.metadata[k]

    # 3. Split
    chunks = text_splitter.split_documents(docs)
    print(f"   -> Pocięto na {len(chunks)} fragmentów.")

    # 4. Upload (OD RAZU!)
    print(f"   ☁️  Wysyłanie {filename} do Firestore...")
    FirestoreVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=COLLECTION_NAME,
        client=client
    )
    print(f"   ✅ Zakończono dla: {filename}\n")


def run_reset_and_upload():
    print(f"🚀 TRYB BEZPIECZNY: Upload plik po pliku do {DATABASE_NAME}")

    client = firestore.Client(project=PROJECT_ID, database=DATABASE_NAME)
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004", project=PROJECT_ID)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)

    # 1. Najpierw czyścimy całość
    delete_all_documents(client)

    # 2. Potem wgrywamy jeden po drugim
    for filename in FILES_TO_UPLOAD:
        process_and_upload_file(filename, client, embeddings, text_splitter)

    print("🏁 CAŁOŚĆ ZAKOŃCZONA SUKCESEM!")


if __name__ == "__main__":
    run_reset_and_upload()
