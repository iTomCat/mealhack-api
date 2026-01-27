import json
import vertexai
from google.cloud import firestore
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore

# Konfiguracja (można też trzymać w .env, ale dla spójności zostawiam tu)
PROJECT_ID = "test-wellness-rag"
COLLECTION_NAME = "knowledge_base_dev"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
DATABASE_NAME = "meal-base"


class RagBatchAgent:
    """
    🕵️ RAG BATCH AGENT (Wersja: AI Studio Clone)
    Symuluje działanie AI Studio poprzez szeroki kontekst i precyzyjny prompt systemowy.
    """

    def __init__(self):
        try:
            print(
                f"🔌 [rag_agent.py] Inicjalizacja RAG (Model: {MODEL_NAME})...")

            vertexai.init(project=PROJECT_ID, location=LOCATION)

            self.client = firestore.Client(
                project=PROJECT_ID, database=DATABASE_NAME)
            self.embeddings = VertexAIEmbeddings(
                model_name="text-embedding-004", project=PROJECT_ID)

            self.vector_store = FirestoreVectorStore(
                collection=COLLECTION_NAME,
                embedding_service=self.embeddings,
                client=self.client
            )

            self.model = GenerativeModel(MODEL_NAME)

        except Exception as e:
            print(f"❌ RAG Init Error: {e}")
            self.vector_store = None

    def resolve_batch(self, missing_products_data):
        """
        Input: Lista słowników [{'name': '...', 'description': '...'}]
        """
        if not missing_products_data or not self.vector_store:
            return []

        # Lista nazw do logowania
        names_only = [p['name'] for p in missing_products_data]
        print(f"🔄 RAG (Deep Search + Vision Context): {names_only}...")

        aggregated_context = ""

        # 1. ZBIERANIE KONTEKSTU (PDF + VISION)
        for item in missing_products_data:
            p_name = item['name']
            p_desc = item.get('description', '')

            try:
                # Szukamy w PDF (k=15 dla pewności trafienia w tabelę)
                docs = self.vector_store.similarity_search(p_name, k=15)
                pdf_content = "\n".join(
                    [re.sub(r'\s+', ' ', d.page_content) for d in docs])
                sources = list(
                    set([d.metadata.get('source_file', '?') for d in docs]))

                # Budujemy kontekst: Doklejamy opis z Vision do treści z PDF
                aggregated_context += f"""
                \n--- PRODUKT: '{p_name}' ---
                Źródła PDF: {sources}
                OPIS WIZUALNY (Stan faktyczny): "{p_desc}"
                TREŚĆ DOKUMENTACJI (PDF):
                {pdf_content}
                """
            except Exception:
                aggregated_context += f"\n--- BRAK PDF DLA: '{p_name}' (Opis Vision: {p_desc}) ---\n"

        # 2. PROMPT "ARCHITEKT" (Twój sprawdzony + 1 poprawka w Zasadzie nr 1)
        prompt = f"""
        SYSTEM INSTRUCTION:
        Jesteś Architektem Danych Żywieniowych. Twoim celem jest stworzenie bazy JSON na podstawie załączonego KONTEKSTU (PDF + Opis Wizualny).

        ŹRÓDŁA:
        1. ATKINSON: Priorytet dla liczb (metrics).
        2. NOVA: Priorytet dla klasyfikacji jakości.
        3. KUBOTA: Priorytet dla porad.
        4. OPIS WIZUALNY: Kluczowy dla doprecyzowania składu (np. rodzaj panierki) i oceny przetworzenia.

        ZASADY EKSTRAKCJI (ŚCIŚLE PRZESTRZEGAJ):
        1. Węglowodany (Carbs): 
           - Szukaj w tekście PDF wartości "Avail Carb" i "Test Portion". Przelicz na 100g: (Avail Carb / Test Portion) * 100.
           - WAŻNE: Jeśli produktu nie ma w tabeli Atkinsona (np. mięso, ryby), a OPIS WIZUALNY wskazuje na dodatek węglowodanowy (np. "panierka", "bułka tarta", "sos słodki"), oszacuj węglowodany na podstawie tego opisu i wiedzy ogólnej. Nie wpisuj 0, jeśli jest panierka!
        2. Grupa NOVA: Sprawdź nazwę i OPIS WIZUALNY. Jeśli opis zawiera "smażone", "panierowane", "instant" -> zazwyczaj NOVA 4.
        3. Fix Tip: Porada ma być krótka i operacyjna.

        KONTEKST:
        {aggregated_context}

        ZADANIE:
        Wygeneruj listę JSON dla produktów: {', '.join(names_only)}.

        WYMAGANY FORMAT (Czysta lista JSON):
        [
          {{
            "id": "RAG_NAZWA",
            "name": "Nazwa",
            "english_name": "Znajdź w tekście",
            "category": "Kategoria",
            "metrics": {{
              "gi_value": 0,
              "gi_category": "LOW/MEDIUM/HIGH",
              "carbs_per_100g": 0.0,
              "gl_per_serving": 0
            }},
            "quality_score": {{
              "nova_group": 1, 
              "explanation": "Uzasadnienie (uwzględnij opis wizualny)"
            }},
            "metabolic_intelligence": {{
              "fix_tip": "Hack metaboliczny."
            }}
          }}
        ]
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ Błąd generowania AI: {e}")
            return []
