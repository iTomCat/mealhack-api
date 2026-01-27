import os
from dotenv import load_dotenv
from google.cloud import firestore
from collections import Counter

# ------------------------------------------------------
# Testowanie plików w RAG - statystyki bazy danych
# Jak zostały pocięte i jaki CHunk naley do jakiego pliku
# ------------------------------------------------------
# Uruchamiamy: python check_rag_stats.py

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base_dev")
DATABASE_NAME = os.getenv("DATABASE_NAME", "meal-base")


def check_database_stats():
    print(
        f"📊 Audyt bazy danych: {DATABASE_NAME} (Kolekcja: {COLLECTION_NAME})...")

    # 1. Łączymy się bezpośrednio z Firestore (bez LangChaina, czysty klient)
    db = firestore.Client(project=PROJECT_ID, database=DATABASE_NAME)
    collection_ref = db.collection(COLLECTION_NAME)

    # 2. Pobieramy tylko pole 'metadata' ze wszystkich dokumentów (dla szybkości)
    # stream() pobiera dokumenty jeden po drugim
    docs = collection_ref.stream()

    stats = Counter()
    total_count = 0

    print("⏳ Liczenie dokumentów (to może chwilę potrwać)...")

    for doc in docs:
        data = doc.to_dict()
        # Wyciągamy nazwę pliku z metadanych
        # Czasami metadata może być puste, więc używamy .get()
        meta = data.get("metadata", {})
        source = meta.get("source_file", "Nieznane źródło")

        stats[source] += 1
        total_count += 1

    # 3. Raport
    print("\n" + "="*40)
    print(f"✅ RAZEM DOKUMENTÓW (CHUNKS): {total_count}")
    print("="*40)
    print("Rozkład według plików źródłowych:")
    for source, count in stats.items():
        print(f"📄 {source}: {count} fragmentów")
    print("="*40)


if __name__ == "__main__":
    check_database_stats()
