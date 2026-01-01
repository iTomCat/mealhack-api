# Używamy stabilnego obrazu Pythona 3.11 (nasza sprawdzona baza)
FROM python:3.11-slim

# Ustawiamy folder roboczy wewnątrz kontenera
WORKDIR /app

# Kopiujemy requirements.txt, aby zainstalować biblioteki
COPY requirements.txt .

# Instalujemy zależności (bez cache, żeby obraz był lekki)
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę Twojego kodu (main.py, dane_test.pdf itp.)
COPY . .

# Cloud Run domyślnie nasłuchuje na porcie 8080
ENV PORT 8080

# Uruchamiamy serwer FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]