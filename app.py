import json
import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- KONFIGURACJA CHMURY ---
PROJECT_ID = "test-wellness-rag"  # <--- TWOJE ID PROJEKTU
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

# --- KONFIGURACJA BAZY ---
DATABASE_FILE = 'food_database_master.json'

DEFAULT_PORTIONS = {
    "rice": 150, "pasta": 150, "potatoes": 150, "cereal grains": 150,
    "breads": 70, "bakery products": 80,
    "meat": 150, "fish": 150, "eggs": 120,
    "vegetables": 150, "fruit": 120,
    "sauces": 30, "fats": 15, "nuts": 30,
    "snacks": 50, "beverages": 250, "soft drinks": 330
}
SIZE_MULTIPLIERS = {"S": 0.5, "M": 1.0, "L": 1.5, "XL": 2.0}


class AICoach:
    def __init__(self):
        print(f"🔌 Łączenie z Mózgiem AI ({MODEL_NAME})...")
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self.model = GenerativeModel(MODEL_NAME)
            print("✅ AI Podłączone.")
        except Exception as e:
            print(f"❌ Błąd AI: {e}")
            self.model = None

    def get_advice(self, meal_items, gl_score, gl_level, missing, database_tips):
        if not self.model:
            return "Brak połączenia z AI. (Tryb offline)"

        # Formatowanie tipów dla AI
        tips_context = "\n".join([f"- {item}: {tip}" for item, tip in database_tips]
                                 ) if database_tips else "Brak specyficznych uwag w bazie."

        # --- OSOBOWOŚĆ COACHA (WINGMAN) ---

        #  CEL: Budowanie więzi i hackowanie biologii (Glucose Goddess).
        prompt = f"""
        JESTEŚ: AI Health Coachem - "Mądrym Kumplem" (Wingman).
        CEL: Budowanie zdrowych nawyków poprzez wiedzę, a nie zakazy.
        TON: Ciepły, luźny, wspierający. Traktuj użytkownika jak inteligentnego przyjaciela.
        
        DANE O POSIŁKU:
        - Co widzę: {', '.join(meal_items)}
        - Ładunek Glikemiczny (GL): {gl_score:.1f} ({gl_level})
        - Czego brakuje (Bufor): {', '.join(missing) if missing else "Nic"}

        BAZA WIEDZY O PRODUKTACH (Twoja ściąga):
        {tips_context}
        (Użyj tych informacji, żeby Twoja porada była merytoryczna i dopasowana do konkretnych składników!)

        TWOJE ZADANIE (Wybierz odpowiedni scenariusz i napisz wiadomość - max 3-4 zdania):
        
        SCENARIUSZ 1: GL WYSOKI/KRYTYCZNY (>20) (np. Pizza, Frytki):
           - Doceń jedzenie ("Brzmi pysznie!", "Ale uczta!").
           - Nie strasz "zjazdem", ale zaproponuj "Fix" (Bufor).
           - Jeśli masz TIP w bazie (np. o łączeniu składników), użyj go.
           - Jeśli nie masz tipa, a brakuje bufora -> "Zjedz najpierw [to co w Buforze] z lodówki".

        SCENARIUSZ 2: GL ŚREDNI (10-20) (Solidny posiłek):
           - Pochwal za dobry wybór i balans.
           - Jeśli w bazie jest ciekawostka (np. o błonniku/tłuszczu), wspomnij o niej edukacyjnie.
           - Daj opcjonalny tip (np. "Delektuj się każdym kęsem").

        SCENARIUSZ 3: GL NISKI (<10) (np. Jajka, Sałatka, Keto):
           - PEŁEN ENTUZJAZM! ("Jesteś mistrzynią!", "Paliwo rakietowe!").
           - Wyjaśnij DLACZEGO to jest super (np. "Dzięki [Składnik z bazy] Twój cukier będzie stabilny jak skała").
           - Buduj poczucie sprawczości ("Widzisz? Małe zmiany dają wielki efekt!").
        

        ZASADY JĘZYKOWE:
        1. RÓŻNE OTWARCIA: Nie zaczynaj ciągle od wykrzyknień ("Uuu", "Ooo", "Wow"). To brzmi sztucznie.
           - Raz zacznij od nazwy dania ("Frytki z batatów? Klasyk!").
           - Innym razem od pytania ("Głodna? Wygląda to konkretnie").
        2. SŁOWNIK ZAKAZANY: "dieta", "kalorie", "grzech", "nie wolno", "źle", "otyli".
        3. NIGDY nie oceniaj negatywnie. Jeśli jedzenie jest "niezdrowe", znajdź sposób, by zminimalizować szkody (fix).
        4. FORMAT: Krótka wiadomość na czacie (max 3-4 zdania), konkretnie, z empatią.
        5. WAŻNE: Zawsze kończ wypowiedź pełnym zdaniem. Nie urywaj myśli.

        Twoja wiadomość:
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                    top_p=0.95,
                    top_k=40  # Dodane dla stabilności przy wyższej temperaturze
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Błąd generowania: {e}"


class MetabolicEngine:
    def __init__(self, db_file):
        self.db_path = db_file
        self.db = self._load_database()
        self.coach = AICoach()  # Inicjalizacja Vertex AI wewnątrz silnika
        print(f"✅ BAZA DANYCH: {len(self.db)} produktów.")
        print("-" * 50)

    def _load_database(self):
        if not os.path.exists(self.db_path):
            return []
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def find_product(self, query):
        query = query.lower()
        # Mały słownik mapowania dla Vision AI (ułatwienie)
        if "frytki z batatów" in query:
            query = "batat"

        results = [p for p in self.db if query in p['name'].lower()
                   or query in p['english_name'].lower()]
        return results[0] if results else None

    def get_weight(self, category, size_code):
        base = 100
        for key, val in DEFAULT_PORTIONS.items():
            if key in category.lower():
                base = val
                break
        return base * SIZE_MULTIPLIERS.get(size_code, 1.0)

    #
    #
    #
    #
    #
    #
    # ------------------------------------------------------------------------------------

    def process_meal(self, vision_items):
        """Główna pętla przetwarzania"""
        print(f"\n📸 FOTO-ANALIZA: Wykryto {len(vision_items)} składników...")

        total_gl = 0
        meal_composition = {"Carbs": 0, "Fiber": 0, "Fat": 0}
        detected_names = []
        expert_tips = []

        # 1. Matematyka (Obliczanie GL)
        for item_name, size in vision_items:
            product = self.find_product(item_name)
            if not product:
                print(f"   ⚠️ Nieznany produkt: {item_name}")
                continue

            detected_names.append(product['name'])
            cat = product.get('category', 'Gen')
            weight = self.get_weight(cat, size)

            # --- A. Obliczanie GL
            gl = (product['metrics']['carbs_per_100g'] *
                  weight / 100) * product['metrics']['gi_value'] / 100
            total_gl += gl

            # --- B. ZBIERANIE INTELIGENCJI  ---
            #
            # Pobieramy dane z JSON
            tip = product.get('metabolic_intelligence', {}).get('fix_tip')
            explanation = product.get('quality_score', {}).get('explanation')
            nova = product.get('quality_score', {}).get('nova_group', 1)

            # Budujemy "paczkę wiedzy" dla AI
            context_string = ""

            # 1. Ostrzeżenie o przetworzeniu (tylko jeśli NOVA >= 4)
            if nova >= 4:
                context_string += f"[UWAGA: Ultra-przetworzone (NOVA 4)!] "

            # 2. Wyjaśnienie dlaczego to jest złe/dobre
            if explanation:
                context_string += f"{explanation} "

            # 3. Konkretna porada
            if tip:
                context_string += f"Rada: {tip}"

            # Jeśli zebraliśmy jakiekolwiek info, dodajemy do listy dla Coacha
            if context_string:
                expert_tips.append((product['name'], context_string))

            # Analiza składu
            if gl > 10:
                meal_composition["Carbs"] += 1
            if "vegetable" in cat.lower() or "salad" in cat.lower():
                meal_composition["Fiber"] += 1
            if "fat" in cat.lower() or "sauce" in cat.lower() or "nuts" in cat.lower() or "egg" in cat.lower() or "meat" in cat.lower() or "cheese" in cat.lower():
                meal_composition["Fat"] += 1

        # 2. Diagnoza poziomu GL
        gl_level = "Niski"
        if total_gl > 35:
            gl_level = "Krytyczny"
        elif total_gl > 20:
            gl_level = "Wysoki"
        elif total_gl > 10:
            gl_level = "Średni"

        # 3. Sprawdzanie czego brakuje
        missing = []
        if meal_composition["Carbs"] > 0:
            if meal_composition["Fiber"] == 0:
                missing.append("Błonnik")
            if meal_composition["Fat"] == 0:
                missing.append("Białko")

        # 4. Wyświetlanie wyników w konsoli
        print(f"DETECTED NAMES: {detected_names}")
        print(f"MEAL COMPOSITION: {meal_composition}")

        print(f"📊 WYNIK: Ładunek Glikemiczny {total_gl:.1f} ({gl_level})")

        if expert_tips:
            print("🧠 WIEDZA Z BAZY (Kontekst dla AI) >>>>>>>>>>>>>>>>>>>>>>>>>")
            for item, tip in expert_tips:
                print(f"   - {item}: {tip}")

        print("-" * 50)

        # 5. Wywołanie AI Coacha

        # advice = self.coach.get_advice(
        #     detected_names, total_gl, gl_level, missing, expert_tips)

        # print("💬 WIADOMOŚĆ NA CZACIE:")
        # print(advice)
        # print("=" * 50)


# ------------------------------------------------------------
#
#
#
#
# --- URUCHOMIENIE TESTOWE ---
if __name__ == "__main__":
    app = MetabolicEngine(DATABASE_FILE)

    print("\n🧪 TESTUJEMY TYLKO SCENARIUSZ 1: Loaded Frytki")

    # Scenariusz 1: Loaded Frytki (Wysoki GL, ale z tłuszczem)
    # Sprawdzamy, czy AI zauważy brak warzyw (błonnika) i zaproponuje "Fix"
    app.process_meal([
        ("Batat gotowany", "L"),  # Vision widzi "Frytki z batatów"
        ("Awokado", "M"),
        ("Miśki zelki", "M"),
        ("Dupa Biskupa", "S"),
        ("Sos czosnkowy", "M")
    ])
