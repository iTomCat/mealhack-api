import json
import os

FILE_NAME = 'food_database_master.json'

# ----------------------------------------
# Usuwanie pola 'gl_per_serving' z każdego produktu w pliku JSON food_database_master.json
# ----------------------------------------


def clean_and_overwrite():
    if not os.path.exists(FILE_NAME):
        print(f"❌ Błąd: Nie widzę pliku {FILE_NAME}")
        return

    print(f"📂 Otwieram {FILE_NAME}...")

    # 1. Wczytanie danych
    try:
        with open(FILE_NAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Błąd odczytu JSON: {e}")
        return

    # 2. Usuwanie pola
    removed_count = 0
    for product in data:
        metrics = product.get('metrics')
        # Sprawdzamy czy metrics istnieje i czy ma klucz gl_per_serving
        if metrics and 'gl_per_serving' in metrics:
            del metrics['gl_per_serving']
            removed_count += 1

    # 3. Zapisywanie (Nadpisywanie tego samego pliku)
    try:
        with open(FILE_NAME, 'w', encoding='utf-8') as f:
            # ensure_ascii=False jest kluczowe dla polskich znaków (żeby nie było krzaków)
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("-" * 30)
        print(f"✅ SUKCES! Plik został nadpisany.")
        print(f"🗑️  Usunięto 'gl_per_serving' z {removed_count} produktów.")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Błąd podczas zapisywania: {e}")


if __name__ == "__main__":
    clean_and_overwrite()
