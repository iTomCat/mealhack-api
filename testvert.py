import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
# import os

"""
📄 PLIK: testvert.py
🧪 ROLA: Tester Połączenia i Osobowości (Connectivity / Personality Sandbox)

OPIS:
To jest prosty skrypt diagnostyczny. Służy TYLKO do dwóch celów:
1. Sprawdzenie, czy klucz API Google Cloud i biblioteka 'vertexai' działają poprawnie.
2. Szybkie testowanie "tonu głosu" Coacha (promptu) bez uruchamiania całej matematyki.

KLUCZOWA CECHA:
Ten plik NIE korzysta z bazy danych ('food_database_master.json') ani nie liczy GL.
Wszystkie dane o posiłku (np. "GL jest WYSOKI") są wpisane ręcznie (hardcoded/mocked) jako symulacja.

UŻYCIE:
Uruchom ten plik, gdy zmieniasz projekt w Google Cloud, testujesz nowy model Gemini lub sprawdzasz uprawnienia.
"""


# --- KONFIGURACJA TWOJEGO PROJEKTU ---
# Wpisz tu ID swojego projektu z Google Cloud
MY_PROJECT_ID = "test-wellness-rag"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"


class VertexAIHealthCoach:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        # Szybki i tani model, idealny do chatowania
        self.model_name = MODEL_NAME

        print(f"🔌 Łączenie z Vertex AI ({self.model_name})...")
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            print("✅ Połączenie nawiązane.")
        except Exception as e:
            print(f"❌ Błąd połączenia: {e}")
            self.model = None

    def get_coach_message(self, meal_summary, gl_level, missing_elements):
        """
        Wysyła dane posiłku do Gemini i odbiera wiadomość w stylu 'Mądrego Kumpla'.
        """
        if not self.model:
            return "Brak połączenia z AI."

        # 1. Definicja Osobowości (System Prompt)
        # To jest klucz do "Skrzydłowego", a nie "Dietetyka"
        system_instruction = f"""
        JESTEŚ: AI Health Coachem - "Mądrym Kumplem" (Wingman).
        CEL: Budowanie więzi, wsparcie emocjonalne i sprytne hackowanie biologii (Glucose Goddess style).
        TON: Ciepły, luzacki, krótki (SMS style). Nigdy nie oceniasz.
        
        ZASADY KRYTYCZNE:
        1. NIGDY nie używaj słów: "dieta", "kalorie", "grzech", "nie wolno", "źle", "otyli".
        2. Jeśli skok cukru jest WYSOKI -> Nie zabraniaj. Zaproponuj "Bufor" (np. zjedz najpierw to, co brakuje).
        3. Jeśli skok cukru jest NISKI -> Pochwal za "paliwo rakietowe" dla mózgu.
        
        DANE O POSIŁKU UŻYTKOWNICZKI:
        - Co je: {meal_summary}
        - Przewidywany skok cukru (GL): {gl_level}
        - Czego brakuje w posiłku (Bufor): {missing_elements}
        
        ZADANIE:
        Napisz krótką wiadomość (max 2-3 zdania). Bądź jak przyjaciel, który dba o jej energię.
        """

        # 2. Konfiguracja generowania (żeby nie było za długie)
        config = GenerationConfig(
            temperature=0.7,  # 0.7 daje trochę kreatywności i "luzu"
            max_output_tokens=150,
        )

        # 3. Strzał do AI
        try:
            response = self.model.generate_content(
                system_instruction,
                generation_config=config
            )
            return response.text.strip()
        except Exception as e:
            return f"❌ Błąd generowania odpowiedzi: {e}"


# --- TEST URUCHOMIENIOWY ---
if __name__ == "__main__":
    # 1. Inicjalizacja
    coach = VertexAIHealthCoach(MY_PROJECT_ID, LOCATION)

    # 2. Symulacja: "Loaded Frytki z Batatów" (Wysoki cukier, brak białka)
    print("\n--- TEST 1: Loaded Frytki (High GL) ---")
    msg1 = coach.get_coach_message(
        meal_summary="Duża porcja frytek z batatów, sos czosnkowy",
        gl_level="WYSOKI",
        missing_elements="Białko, Zielone Warzywa"
    )
    print(f"🤖 Coach mówi:\n{msg1}")

    # 3. Symulacja: "Jajecznica z awokado" (Idealnie)
    print("\n--- TEST 2: Jajecznica (Low GL) ---")
    msg2 = coach.get_coach_message(
        meal_summary="Jajecznica na maśle, awokado, pomidor",
        gl_level="NISKI",
        missing_elements="Nic, jest idealnie"
    )
    print(f"🤖 Coach mówi:\n{msg2}")
