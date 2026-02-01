import json
import os
from rag_agent import RagBatchAgent
import datetime
from google.cloud import firestore


# --------------------------------------------------------------
"""
PObieranie danych JSON z RAG
🧠 ROLA: Główny Silnik Metaboliczny (Production Logic / RAG Lite)

OPIS:
Ten plik integruje trzy warstwy:
1. WARSTWA DANYCH (Fakty): Pobiera twarde dane z pliku 'food_database_master.json' (IG, węglowodany, fix_tips).
2. WARSTWA MATEMATYCZNA (Logika): Oblicza Ładunek Glikemiczny (GL) na podstawie wagi porcji.
3. WARSTWA AI (Osobowość): Buduje bogaty kontekst (Prompt) na podstawie faktów i wysyła go do Vertex AI.

KLUCZOWA CECHA:
AI tutaj NIE zgaduje faktów. AI otrzymuje fakty (np. "To jest produkt przetworzony") i ubiera je w formę "Mądrego Kumpla".

UŻYCIE:
Główny plik do uruchamiania analizy posiłków z wykorzystaniem pełnej wiedzy metabolicznej.
"""
# --------------------------------------------------------------

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, 'food_database_master.json')
REVIEW_FILE = os.path.join(BASE_DIR, 'new_products_review.json')


class MetabolicEngine:
    def __init__(self, db_file):
        self.db_path = db_file
        self.db = self._load_json(self.db_path)

        # Inicjalizacja Agenta RAG
        self.rag_agent = RagBatchAgent()

        # Inicjalizacja bazy danych (do zapisu review)
        self.firestore_db = firestore.Client(
            project="mealhack-app", database="meal-base")

        if len(self.db) > 0:
            print(f"✅ BAZA LOKALNA: {len(self.db)} produktów.")
        else:
            print("⚠️ BAZA LOKALNA PUSTA.")
        print("-" * 50)

    def _load_json(self, path):
        """Uniwersalna funkcja do ładowania JSON (Odporna na puste pliki)"""
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Jeśli plik jest pusty -> zwróć pustą listę
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Plik {path} uszkodzony/pusty. Rozpoczynam od nowa.")
            return []
        except Exception as e:
            print(f"❌ Błąd ładowania {path}: {e}")
            return []

    def _save_to_review_file_to_json_locally(self, new_items, source_id="UNKNOWN"):
        """
        Zapisuje produkty z RAG do pliku 'new_products_review.json'.
        STRUKTURA: Grupuje produkty pod konkretnym ID posiłku (Wrapper).
        DEDUPLIKACJA: Sprawdza, czy paczka o danym ID już istnieje.
        """
        if not new_items:
            return

        current_review_list = self._load_json(REVIEW_FILE)

        # --- KROK 1: Sprawdzenie duplikatów po ID posiłku ---
        # Zbieramy ID wszystkich paczek, które już są w pliku
        existing_meal_ids = {entry.get('meal_unique_id')
                             for entry in current_review_list}

        if source_id in existing_meal_ids:
            print(
                f"INFO: Paczka '{source_id}' już istnieje w pliku review, pominięto zapis.")
            return
        # ----------------------------------------------------

        print(
            f"💾 Zapisywanie {len(new_items)} produktów do nowej paczki '{source_id}'...")

        # 2. Przygotowujemy listę produktów dla TEGO KONKRETNEGO posiłku
        meal_products_list = []

        for item in new_items:
            metrics = item.get('metrics', {})
            gi_val = metrics.get('gi_value', 0)
            gi_cat = metrics.get('gi_category')
            if not gi_cat:
                if gi_val < 55:
                    gi_cat = "LOW"
                elif gi_val < 70:
                    gi_cat = "MEDIUM"
                else:
                    gi_cat = "HIGH"

            clean_product_entry = {
                "id": item.get('id'),
                "name": item.get('name'),
                "english_name": item.get('english_name', item.get('name')),
                "category": item.get('category', 'Gen'),
                "metrics": {
                    "gi_value": gi_val,
                    "gi_category": gi_cat,
                    "carbs_per_100g": metrics.get('carbs_per_100g', 0),
                },
                "quality_score": item.get('quality_score', {}),
                "metabolic_intelligence": item.get('metabolic_intelligence', {}),
                "source_origin": item.get('source_origin', 'RAG_AUTO_GENERATED')
            }

            meal_products_list.append(clean_product_entry)

        # 3. Tworzymy WRAPPER
        meal_wrapper = {
            "meal_unique_id": source_id,
            "source_origin": "RAG_BATCH_PROCESS",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": meal_products_list
        }

        # 4. Dodajemy i zapisujemy
        current_review_list.append(meal_wrapper)

        try:
            with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
                json.dump(current_review_list, f, indent=4, ensure_ascii=False)
            print(
                f"✅ Dodano paczkę {source_id} z {len(meal_products_list)} produktami do review.")
        except Exception as e:
            print(f"❌ Błąd zapisu review: {e}")

    def _save_to_review_file(self, rag_results, source_id):
        """
        Wysyła produkty wygenerowane przez AI do kolekcji 'new_product_reviews' w Firestore.
        Ustawia flagę is_verified = False.
        """
        if not rag_results:
            return

        print(
            f"💾 Zapisuję {len(rag_results)} produktów do poczekalni (Firestore)...")

        # Uchwyt do kolekcji w bazie
        collection_ref = self.firestore_db.collection('new_product_reviews')

        for item in rag_results:
            doc_id = item.get('id', 'UNKNOWN_ID')

            # Struktura dokumentu w bazie
            review_doc = {
                "id": doc_id,                       # ID (np. RAG_RYZ_BIALY)
                # Nazwa dla łatwego podglądu
                "name": item.get('name'),

                # --- KLUCZOWE FLAGI ---
                "is_verified": False,               # <--- Domyślnie NIEZATWIERDZONE

                # --- METADANE ---
                "created_at": firestore.SERVER_TIMESTAMP,  # Data dodania
                "source_meal_id": source_id,        # Z jakiego posiłku to pochodzi

                # --- DANE PRODUKTU ---
                "product_data": item                # Pełny JSON wygenerowany przez RAG
            }

            try:
                # .set() nadpisze dokument jeśli już istnieje (aktualizacja danych),
                # lub stworzy nowy. To bezpieczne.
                collection_ref.document(doc_id).set(review_doc)
                print(f"   - ⏳ [PENDING] Wysłano do weryfikacji: {doc_id}")
            except Exception as e:
                print(f"   ❌ Błąd zapisu do Firestore: {e}")

    def _clean_product_name(self, raw_name):
        """
        Zmienia: 'Kurczak (Pierś) w płatkach' -> 'Kurczak Pierś w płatkach'
        Usuwa nawiasy, które mylą wyszukiwarkę, ale ZACHOWUJE treść (np. rodzaj panierki).
        """
        # Zamień nawiasy na spacje
        clean = raw_name.replace('(', ' ').replace(')', ' ').replace('/', ' ')
        # Usuń podwójne spacje i białe znaki
        clean = ' '.join(clean.split())
        return clean

    def find_product_local(self, query_name):
        """Inteligentne wyszukiwanie lokalne (Strict Match)."""
        query_lower = query_name.lower()
        # Tutaj query_name jest już oczyszczone przez _clean_product_name
        clean_query = query_lower.replace(',', '')
        query_tokens = set(clean_query.split())

        SAFE_EXTRAS = {'gotowany', 'surowy', 'tradycyjny', 'zwykły',
                       'na', 'parze', 'w', 'wodzie', 'smażony', 'pieczony'}

        for db_product in self.db:
            db_name_lower = db_product['name'].lower()

            # Level 1: Exact
            if query_lower == db_name_lower:
                return db_product

            # Level 2: Smart Subset
            clean_db = db_name_lower.replace(
                '(', '').replace(')', '').replace(',', '')
            db_tokens = set(clean_db.split())

            if query_tokens.issubset(db_tokens):
                extra_words = db_tokens - query_tokens
                dangerous_extras = extra_words - SAFE_EXTRAS
                if not dangerous_extras:
                    return db_product
        return None

    def process_meal(self, meal_json):

        meta = meal_json.get('meta', {})
        meal_unique_id = meta.get('meal_unique_id', 'MANUAL_TEST')

        ingredients = meal_json.get('skladniki', [])
        print(f"\n📸 ANALIZA DANYCH: {len(ingredients)} składników...")

        products_to_process = []
        missing_items_map = {}

        # 1. Wery# Pętla po składnikach w przesłanym do analizy posiłku
        for item in ingredients:
            raw_name = item.get('nazwa', 'Unknown')
            weight = item.get('waga_g', 0)
            raw_state = item.get('stan', '')

            # --- NAPRAWA 1: CZYSZCZENIE NAZWY ---
            clean_name = self._clean_product_name(raw_name)

            # Budujemy bogaty opis dla RAG (Stan + Oryginalna nazwa z nawiasami)
            full_description = f"{raw_state}. Oryginalna nazwa: {raw_name}"

            local_product = self.find_product_local(clean_name)

            if local_product:
                print(f"   ✅ Znaleziono lokalnie: {clean_name}")
                p_copy = local_product.copy()
                p_copy['serving_weight_g'] = weight
                p_copy['original_description'] = full_description
                p_copy['source_type'] = 'LOCAL_DB'
                products_to_process.append(p_copy)
            else:
                print(
                    f"   🔸 Brak ścisłego dopasowania: '{clean_name}' (Oryginał: {raw_name})")
                missing_items_map[clean_name] = {
                    'weight': weight,
                    'desc': full_description
                }
        # 🛑 --- MIEJSCE NA TWOJĄ BLOKADĘ --- 🛑
        # print("\n🛑 [TEST MODE] Zatrzymuję przed wysłaniem do RAG.")
        # print(
        #     f"✅ Znalezione w bazie lokalnej: {[p['name'] for p in products_to_process]}")
        # print(
        #     f"🔸 Nieznane (trafiłyby do RAG): {list(missing_items_map.keys())}")

        # return []

        # 2. RAG Batch
        if missing_items_map:

            # --- NAPRAWA 2: Budujemy listę SŁOWNIKÓW dla rag_agent ---
            rag_input_data = []
            for c_name, data in missing_items_map.items():
                rag_input_data.append({
                    'name': c_name,       # Czysta nazwa (do szukania w PDF)
                    'description': data['desc']  # Bogaty opis (do analizy)
                })

            # Tłumaczenie nazw i pobranie danych z RAG
            rag_results = self.rag_agent.resolve_batch(rag_input_data)

            # Zapis do new_products_review.json
            self._save_to_review_file(rag_results, source_id=meal_unique_id)

            for rag_item in rag_results:
                name = rag_item.get('name')

                # Próbujemy odzyskać dane po nazwie zwróconej przez RAG
                input_data = missing_items_map.get(name)

                # Fallback - jeśli RAG lekko zmienił nazwę
                if not input_data:
                    found_key = next(
                        (k for k in missing_items_map if k in name or name in k), None)
                    input_data = missing_items_map[found_key] if found_key else list(
                        missing_items_map.values())[0]

                rag_item['id'] = f"RAG_{name.replace(' ', '_')}"
                rag_item['serving_weight_g'] = input_data['weight']
                rag_item['original_description'] = input_data['desc']
                rag_item['source_type'] = 'RAG_GENERATED'

                products_to_process.append(rag_item)

        # 3. Finalne Formatowanie
        final_json_output = []

        print(
            f"\n🧮 GENEROWANIE JSON DLA {len(products_to_process)} ELEMENTÓW...")

        for product in products_to_process:
            weight = product.get('serving_weight_g', 0)
            metrics = product.get('metrics', {})
            carbs = metrics.get('carbs_per_100g', 0)
            gi = metrics.get('gi_value', 0)

            gl = (carbs * weight / 100) * gi / 100

            gi_cat = metrics.get('gi_category')
            if not gi_cat:
                if gi < 55:
                    gi_cat = "LOW"
                elif gi < 70:
                    gi_cat = "MEDIUM"
                else:
                    gi_cat = "HIGH"
            # TO jest przekazywane do następnego zapytania AI
            # SPrawdzić obliczanie GL czy waga w obliczeniu jest OK
            meal_item_json = {
                "id": product.get("id"),
                "name": product["name"],
                "category": product.get('category', 'Gen'),
                "metrics": {
                    "gi_value": int(gi),
                    "gi_category": gi_cat,
                    "carbs_per_100g": float(carbs),
                    "gl_per_serving": round(gl, 1),
                    "serving_size_g": int(weight)
                },
                "quality_score": {
                    "nova_group": product.get("quality_score", {}).get("nova_group", 1),
                    "explanation": product.get("quality_score", {}).get("explanation", "")
                },
                "vision_context": {
                    "detailed_description": product.get('original_description', ''),
                    "user_input_name": product["name"]
                },
                "metabolic_intelligence": {
                    "fix_tip": product.get("metabolic_intelligence", {}).get("fix_tip", "")
                },
                "source": product.get("source_type", "UNKNOWN")
            }
            final_json_output.append(meal_item_json)

        return final_json_output


if __name__ == "__main__":
    engine = MetabolicEngine(DATABASE_FILE)

    print("\n🧪 TEST: Traceability ID (Nested Meta)")

    # Symulacja danych z Vision (Nowa struktura)
    input_json_data = {
        "meta": {
            "meal_unique_id": "DupaBiskupa666",
            "talerz_srednica_mm": 240,
            "calkowita_waga_g": 481
        },
        "skladniki": [
            {
                "nazwa": "Biały ryż",
                "waga_g": 247,
                "stan": "Gotowany na sypko, biały"
            },
            {
                "nazwa": "Kurczak w panierce (Pierś z kurczaka) (Płatki kukurydziane / Panko) (Smażone na głębokim tłuszczu)",
                "waga_g": 234,
                "stan": "Smażone kawałki fileta w złocistej, chrupiącej panierce"
            }
        ]
    }

    result = engine.process_meal(input_json_data)

    print("\n✅ Koniec testu.")
