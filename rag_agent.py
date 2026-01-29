import json
import vertexai
from google.cloud import firestore
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
import re
import os

# Konfiguracja (można też trzymać w .env, ale dla spójności zostawiam tu)
PROJECT_ID = "mealhack-app"
COLLECTION_NAME = "knowledge_base_dev"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
DATABASE_NAME = "meal-base"


class RagBatchAgent:
    """
    🕵️ RAG BATCH AGENT
    Klasa do obsługi RAG dla partii brakujących produktów.
    Używa Vertex AI Generative Models oraz Firestore jako Vector Store.
    SPrawdza czy produkt jest ju w bazie food_database_master.json
    jeśli nie to tłumaczy naazwę na angielski i szuka w RAG.
    Na podstawie znalezionych dokumentów generuje pełne dane JSON.
    w new_products_rag.json
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

    def _clean_json_response(self, text):
        clean = re.sub(r'```json\s*', '', text)
        clean = re.sub(r'```\s*$', '', clean)
        return clean.strip()

    def _translate_names(self, pl_names):
        if not pl_names:
            return {}
        fallback = {name: name for name in pl_names}

        prompt = f"""
        Zadanie: Przetłumacz polskie nazwy na ANGIELSKI (terminologia USDA/Atkinson).
        Input: {json.dumps(pl_names)}
        Output JSON: {{"Nazwa Polska": "English Name"}}
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return json.loads(self._clean_json_response(response.text))
        except Exception:
            return fallback

    def _get_source_name(self, doc):
        """
        Wyciąga nazwę pliku. 
        Wiemy już z debugowania, że klucz to 'source_file'.
        """
        if not doc.metadata:
            return "Nieznane źródło"

        meta = doc.metadata
        # Priorytet dla 'source_file', bo to widzieliśmy w bazie
        src = meta.get('source_file') or meta.get(
            'source') or meta.get('file_path') or meta.get('filename')

        if not src:
            return "Nieznane źródło"

        return os.path.basename(str(src))

    def resolve_batch(self, missing_products_data):
        if not missing_products_data or not self.vector_store:
            return []

        pl_names = [p['name'] for p in missing_products_data]
        print(f"🔄 RAG Start: {pl_names}")

        translations_map = self._translate_names(pl_names)
        print(f"🌍 Tłumaczenie: {list(translations_map.values())}")

        aggregated_context = ""

        for item in missing_products_data:
            p_name_pl = item['name']
            p_desc = item.get('description', '')
            search_query_eng = translations_map.get(p_name_pl, p_name_pl)

            try:
                # Szukamy w bazie
                docs = self.vector_store.similarity_search(
                    search_query_eng, k=25)

                # Wyciągamy źródła
                found_sources = sorted(
                    list(set([self._get_source_name(d) for d in docs])))

                pdf_content = "\n".join(
                    [re.sub(r'\s+', ' ', d.page_content) for d in docs])

                aggregated_context += f"""
                \n--- PRODUKT: '{p_name_pl}' ---
                SEARCH QUERY (ENG): '{search_query_eng}'
                OPIS WIZUALNY: "{p_desc}"
                ZNALEZIONE PLIKI (FILES): {found_sources}
                TREŚĆ Z DOKUMENTACJI (PDF):
                {pdf_content}
                """
            except Exception as e:
                print(f"⚠️ Błąd RAG dla {p_name_pl}: {e}")
                aggregated_context += f"\n--- BRAK DANYCH DLA: '{p_name_pl}' ---\n"

        # 4. PROMPT "ARCHITEKT" (Pełna logika biznesowa)
        prompt = f"""
        SYSTEM INSTRUCTION:
        Jesteś Architektem Danych Żywieniowych. Twoim celem jest stworzenie bazy JSON, łącząc dane z dokumentacji (PDF) oraz analizy wizualnej (Vision).

        ŹRÓDŁA WIEDZY (Do wykorzystania):
        1. ATKINSON (Tabela1_IG, Tabela2_IG): Priorytet dla liczb (GI, Carbs).
        2. NOVA (Classification Sheet): Priorytet dla oceny przetworzenia.
        3. KUBOTA: Priorytet dla porad metabolicznych (kolejność jedzenia).
        4. VISION (Opis wizualny): Kluczowy dla produktów złożonych (np. panierka), których nie ma w tabelach prostych składników.

        ZASADY EKSTRAKCJI I OBLICZEŃ (Kluczowe!):
        
        1. WĘGLOWODANY (Carbs per 100g):
           - Szukaj w tabelach (pod angielską nazwą) kolumn "Avail Carb" i "Test Portion".
           - Wzór: (Avail Carb / Test Portion) * 100.
           - REGUŁA "PANIERKA" (Crucial): Jeśli produkt to mięso/ryba (0 węgli w naturze), ALE opis wizualny zawiera "panierka", "panko", "bułka tarta", "smażone" -> MUSISZ oszacować węglowodany z panierki (ok. 15-25g/100g). Nie wpisuj 0!
        
        2. GRUPA NOVA:
           - "White rice" / "Sticky rice" -> NOVA 1 (Minimally processed).
           - "Breaded" (panierowane) / "Fried" (smażone) / "Instant" -> NOVA 4 (Ultra-processed).
        
        3. ŹRÓDŁA (Source Origin):
           - Wymień WSZYSTKIE pliki PDF, z których skorzystałeś przy ocenie produktu (zarówno do liczb, jak i klasyfikacji NOVA).
           - Oddziel nazwy przecinkami.
           - Jeśli dane są szacowane na podstawie opisu wizualnego (np. panierka), wpisz: "Estimation based on Vision & General Knowledge".

        4. PORADA (Fix Tip):
           - Jeśli wysoki IG lub NOVA 4 -> Porada z pliku Kubota (np. "Zjedz najpierw warzywa").

        DANE WEJŚCIOWE:
        {aggregated_context}

        ZADANIE:
        Wygeneruj JSON dla listy: {', '.join(pl_names)}.

        WYMAGANY FORMAT (JSON):
        [
          {{
            "id": "RAG_NAZWA",
            "name": "Polska nazwa",
            "english_name": "Angielska nazwa",
            "category": "Kategoria",
            "metrics": {{
              "gi_value": 70, (int)
              "gi_category": "HIGH",
              "carbs_per_100g": 28.0, (float)
              "gl_per_serving": 0
            }},
            "quality_score": {{
              "nova_group": 1, 
              "explanation": "..."
            }},
            "metabolic_intelligence": {{
              "fix_tip": "..."
            }},
            "source_origin": "Nazwa pliku PDF"
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
            return json.loads(self._clean_json_response(response.text))
        except Exception as e:
            print(f"❌ Błąd generowania AI: {e}")
            return []
