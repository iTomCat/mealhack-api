import os
import time
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from google.cloud import firestore
import firebase_admin
from firebase_admin import auth, credentials
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_firestore import FirestoreVectorStore
# from langchain.chains import ConversationalRetrievalChain (tu wstawisz swoje importy LangChain)

# Ładowanie zmiennych środowiskowych (dla testów lokalnych z pliku .env)
from dotenv import load_dotenv
load_dotenv()

# --- 1. KONFIGURACJA ---
# Pobieramy konfigurację ze zmiennych środowiskowych (Cloud Run to podstawi)
PROJECT_ID = os.environ.get("PROJECT_ID", "mealhack-app")
COLLECTION_NAME = os.environ.get(
    "COLLECTION_NAME", "WiedzaTestowa")  # Domyślnie testowa

# Inicjalizacja Firebase (tylko raz)
if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})

# Kolekcja w Firestore, gdzie trzymamy informacje o użytkownikach.
USER_KNOWLEADGE_COLLECTION = "Profile"

app = FastAPI()

# Model danych przychodzących z Fluttera


class ChatRequest(BaseModel):
    message: str

# --- 2. FUNKCJA OBSERWATORA (To dzieje się w tle) ---


def aktualizuj_profil_background(user_id: str, message: str, reply: str):
    """
    Ta funkcja uruchomi się PO wysłaniu odpowiedzi do użytkownika.
    Nie blokuje czatu!
    """
    print(f"Log: Zaczynam aktualizację profilu dla {user_id}...")

    # Tu Twoja logika zapisu do Firestore
    try:
        db = firestore.Client(project=PROJECT_ID)
        doc_ref = db.collection("Profile").document(user_id)

        # Przykład prostego zapisu historii
        doc_ref.set({
            "ostatnia_aktywnosc": firestore.SERVER_TIMESTAMP,
            "ostatnia_wiadomosc": message,
            # Tutaj w przyszłości Twoja logika wyciągania faktów żywieniowych
        }, merge=True)

        print(f"Log: Profil {user_id} zaktualizowany.")
    except Exception as e:
        print(f"Błąd zapisu profilu: {e}")

# --- 3. ENDPOINT CHATU ---


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,  # <--- Magiczny parametr FastAPI
    authorization: str = Header(None)
):
    # A. Weryfikacja Tokena Firebase (Bezpieczeństwo)
    env_type = os.environ.get("ENV_TYPE", "PROD")

    # --- ZMIANA: Uproszczona logika dla trybu DEV ---
    if env_type == "DEV":
        # W trybie DEV wpuszczamy zawsze, ignorując poprawność tokena
        print(f"🔓 Tryb DEV aktywny. Bypass dla tokena: {authorization}")
        user_id = "tester_lokalny"
    else:
        # W trybie PROD (domyślnym) sprawdzamy rygorystycznie
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Brak tokena autoryzacyjnego")

        token = authorization.split(" ")[1]
        try:
            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token['uid']
        except Exception as e:
            print(f"Błąd weryfikacji tokena PROD: {e}")
            raise HTTPException(status_code=401, detail="Nieprawidłowy token")

    # ------------------------------------------------

    # B. Logika RAG (Tu wstawiasz kod LangChain)
    # Skrótowo dla przykładu:

    # embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    # vector_store = FirestoreVectorStore(collection=COLLECTION_NAME, embedding_service=embeddings)
    # retriever = vector_store.as_retriever()
    # ... tu Twoja sieć LangChain ...

    # Symulacja odpowiedzi AI:
    ai_response_text = f"Cześć! Korzystam z bazy: {COLLECTION_NAME}. Odpowiadam na: {request.message}"

    # C. Zlecenie zadania w tle (Background Task)
    background_tasks.add_task(
        aktualizuj_profil_background, user_id, request.message, ai_response_text)

    # D. Natychmiastowy zwrot do Fluttera/Streamlita
    # Zmieniłem klucz na "response", żeby pasował do Twojego test_ui.py
    return {
        "response": ai_response_text
    }
