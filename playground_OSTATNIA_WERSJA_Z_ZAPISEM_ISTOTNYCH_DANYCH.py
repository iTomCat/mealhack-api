import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_firestore import FirestoreVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from google.cloud import firestore

# --- 1. KONFIGURACJA PROJEKTU ---
# Tutaj definiujemy stałe, które łączą nas z chmurą Google.
PROJECT_ID = "test-wellness-rag"
COLLECTION_NAME = "WiedzaTestowa"  # Tam leżą Twoje poćwiartowane PDF-y
# Tymczasowe ID. W przyszłości tu będzie login użytkowniczki.
USER_ID = "test_user_anabella"

# Kolekcja w Firestore, gdzie trzymamy informacje o użytkownikach.
USER_KNOWLEADGE_COLLECTION = "Profile"

st.set_page_config(page_title="Playground Anabellaaa", page_icon="🥑")
st.title("🥑 Anabelllla: Twoja Osobista Dietetyczka")

# --- 2. FUNKCJA: OBSERWATOR (PAMIĘĆ DŁUGOTERMINOWA) ---
# Ta funkcja działa jak "sekretarka". Czyta rozmowę i notuje ważne fakty w osobnej teczce (Firestore).
# Nie bierze udziału w rozmowie, tylko słucha i notuje.


def aktualizuj_profil_uzytkownika(user_knowelage_collection, tekst_uzytkownika, user_id):

    # A. FILTR KOSZTÓW
    # Jeśli użytkownik napisał tylko "Hej" (mniej niż 10 znaków), szkoda uruchamiać AI.
    # Oszczędzamy pieniądze i czas.
    if len(tekst_uzytkownika) < 10:
        return False

    # B. POŁĄCZENIE Z BAZĄ PROFILI
    # Łączymy się z kolekcją "Profile" (inna niż ta z PDF-ami!).
    db = firestore.Client(project=PROJECT_ID)
    doc_ref = db.collection(user_knowelage_collection).document(user_id)

    # Pobieramy to, co już wiemy, żeby nie nadpisywać bez sensu.
    doc = doc_ref.get()
    obecny_profil = doc.to_dict().get(
        "info", "Brak danych.") if doc.exists else "Brak danych."

    # C. MODEL "OBSERWATOR"
    # Używamy szybkiego modelu. Jego zadaniem jest tylko wyciąganie danych (Data Extraction).
    llm_extractor = ChatVertexAI(
        model_name="gemini-2.5-flash",
        # Zero kreatywności. Ma być precyzyjny jak matematyk.
        temperature=0,
        project=PROJECT_ID,
        location="us-central1"
    )

    # D. PROMPT "BIUROKRATA" (Anti-Hallucination)
    # Stosujemy technikę "Few-Shot Prompting" (uczenie na przykładach),
    # żeby model wiedział, że ma ignorować pytania o pogodę, a notować tylko fakty medyczne.
    prompt_analizy = f"""
    Jesteś precyzyjnym urzędnikiem medycznym. Twoim zadaniem jest aktualizacja karty pacjenta.
    
    ZASADY KRYTYCZNE:
    1. Wyciągaj TYLKO fakty podane WPROST w "NOWA WIADOMOŚĆ".
    2. NIE ZGADUJ. Jeśli użytkownik pisze "chcę schudnąć", wpisz cel. Jeśli nie podał liczby kilogramów, NIE WYMYŚLAJ.
    3. Jeśli nowa informacja nadpisuje starą (np. nowa waga), zaktualizuj ją.
    4. Jeśli w wiadomości nie ma faktów medycznych/osobistych (wiek, imię, choroby, waga, cel), zwróć słowo: BEZ_ZMIAN.
    
    --- PRZYKŁADY (Naucz się z nich) ---
    
    Profil: Brak danych.
    Wiadomość: "Cześć, mam na imię Ania."
    Wynik: Imię: Ania
    
    Profil: Imię: Ania, Wiek: 20 lat.
    Wiadomość: "Jaka jest pogoda?"
    Wynik: BEZ_ZMIAN
    
    Profil: Imię: Ania, Waga: 60kg.
    Wiadomość: "Schudłam, teraz ważę 58 kg."
    Wynik: Imię: Ania, Waga: 58 kg
    
    Profil: Brak danych.
    Wiadomość: "Chciałabym lepiej się odżywiać."
    Wynik: Cel: Zdrowe odżywianie
    
    --- KONIEC PRZYKŁADÓW ---
    
    A TERAZ TY:
    
    OBECNY PROFIL: {obecny_profil}
    NOWA WIADOMOŚĆ: "{tekst_uzytkownika}"
    
    WYNIK (Tylko czyste dane po przecinku lub BEZ_ZMIAN):
    """

    # E. WYKONANIE ZADANIA
    wynik = llm_extractor.invoke(prompt_analizy).content.strip()

    # Czyszczenie śmieci (czasem model doda `backticks` lub słowo json)
    wynik = wynik.replace("`", "").replace("json", "")

    # F. DECYZJA O ZAPISIE
    # Zapisujemy w bazie tylko wtedy, gdy coś się faktycznie zmieniło.
    if "BEZ_ZMIAN" in wynik:
        return False

    if wynik != obecny_profil:
        doc_ref.set({"info": wynik})
        return True  # Zgłaszamy sukces (True), żeby wyświetlić powiadomienie.

    return False

# --- 3. GŁÓWNY SILNIK (RAG + OSOBOWOŚĆ) ---


@st.cache_resource
def get_qa_chain():
    # A. EMBEDDINGI (Tłumacz Tekst -> Liczby)
    # Musi być w tym samym regionie co Twoja baza Firestore (Holandia).
    embedding_model = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID,
        location="europe-west4"
    )

    # B. MÓZG (Gemini)
    # Tutaj dzieje się magia rozmowy. Ustawiamy go w USA dla najlepszej dostępności.
    llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        # Lekka kreatywność (0.3), żeby Anabella brzmiała naturalnie, a nie jak robot.
        temperature=0.3,
        project=PROJECT_ID,
        location="us-central1"
    )

    # C. POŁĄCZENIE Z BAZĄ WIEDZY (PDF-y)
    client = firestore.Client(project=PROJECT_ID)
    vector_store = FirestoreVectorStore(
        collection=COLLECTION_NAME,
        embedding_service=embedding_model,
        client=client
    )

    # D. RETRIEVER (Szperacz)
    # k=3 oznacza: "Znajdź 3 najlepsze fragmenty z PDF-ów".
    # Dzięki temu AI ma szerszy kontekst (np. fragment o imieniu I fragment o wieku).
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # E. DEFINICJA OSOBOWOŚCI (SYSTEM PROMPT)
    # To jest najważniejsza część. Tutaj mówimy AI, kim jest.
    # Zmienna {profil_uzytkownika} zostanie podmieniona dynamicznie w trakcie rozmowy.
    system_template = """Rozmawiasz z użytkowniczką o zdrowym odżywianiu.
    
    PROFIL TWOJEJ ROZMÓWCZYNI (UŻYTKOWNICZKI):
    {profil_uzytkownika}
    
    TWOJE ZADANIE:
    Odpowiedz na pytanie użytkowniczki, bazując WYŁĄCZNIE na poniższych fragmentach (CONTEXT).
    
    ZASADY LOGIKI:
    1. Skup się TYLKO na zadanym pytaniu. Nie zgłaszaj konfliktów w tematach, o które nikt nie pytał.
    2. Jeśli pytanie dotyczy konkretnej cechy (np. wieku Anabelli) i widzisz w CONTEXT różne wartości (np. 44 i 53), 
    napisz: "Dokumenty zawierają sprzeczność w tym temacie: źródło A podaje X, a źródło B podaje Y".
    3. Traktuj instrukcję, że jesteś dietetyczką, jako nadrzędną rolę, ale jeśli w CONTEXT (PDF) są informacje 
    o Twoim życiu prywatnym (np. hobby, wygląd), używaj ich.
    4. Rozróżniaj osoby: "TY" to Anabella. "UŻYTKOWNICZKA" to osoba zadająca pytania.
    
    WIEDZA Z BAZY (CONTEXT):
    {context}
    """

    # Sklejenie szablonu w całość
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # F. TWORZENIE ŁAŃCUCHA
    # ConversationalRetrievalChain potrafi pamiętać historię rozmowy ("A ile ona ma lat?").
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        # <-- Tu wstrzykujemy duszę Anabelli
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )


qa_chain = get_qa_chain()

# --- 4. INTERFEJS (PASEK BOCZNY) ---
# Tutaj wyświetlamy to, co "Obserwator" zapisał w bazie.

with st.sidebar:
    st.header("🕵️‍♀️ Co o Tobie wiem?")
    # Pobieramy dane prosto z Firestore przy każdym odświeżeniu strony
    db = firestore.Client(project=PROJECT_ID)
    profil_doc = db.collection("Profile").document(USER_ID).get()

    if profil_doc.exists:
        wiedza_o_userze = profil_doc.to_dict().get("info", "")
        st.info(wiedza_o_userze)  # Niebieska ramka z danymi
    else:
        st.write("Jeszcze nic. Napisz coś o sobie!")
        wiedza_o_userze = "Brak danych medycznych."

# --- 5. OBSŁUGA CZATU (GŁÓWNA PĘTLA) ---

# Inicjalizacja historii w pamięci przeglądarki
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie starych wiadomości na ekranie
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Czekamy, aż użytkownik coś wpisze...
if prompt := st.chat_input("Napisz coś (np. Mam IO, co jeść na śniadanie?)..."):

    # KROK 1: Wyświetlamy pytanie użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # KROK 2: Uruchamiamy "Obserwatora" w tle
    # Sprawdza, czy w pytaniu są nowe fakty o użytkowniczce.
    zaktualizowano = aktualizuj_profil_uzytkownika(
        USER_KNOWLEADGE_COLLECTION, prompt, USER_ID)

    if zaktualizowano:
        st.toast("Zaktualizowałam Twój profil zdrowotny!", icon="✅")
        # Pobieramy od razu świeże dane, żeby Anabella uwzględniła je w TEJ odpowiedzi
        profil_doc = db.collection("Profile").document(USER_ID).get()
        wiedza_o_userze = profil_doc.to_dict().get("info", "")

    # KROK 3: Budowanie historii rozmowy (dla LangChaina)
    # LangChain potrzebuje kontekstu, żeby wiedzieć, o czym rozmawialiśmy wcześniej
    # (np. gdy pytasz "A jakie ma objawy?" - AI musi wiedzieć, o jaką chorobę pytałeś wcześniej).
    chat_history = []
    # Przechodzimy przez wiadomości parami (Pytanie -> Odpowiedź)
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            pytanie = st.session_state.messages[i]["content"]
            # Pobieramy odpowiedź i czyścimy ją ze stopek "Źródło"
            # UWAGA: Odpowiedź w historii zawiera stopkę "--- (Źródło: ...)", której NIE CHCEMY
            # karmić modelu ponownie (żeby nie pętlił się na źródłach).
            # Dlatego ucinamy wszystko co znajduje się po znaczniku "\n\n---".
            odpowiedz_czysta = st.session_state.messages[i+1]["content"].split("\n\n---")[
                0]
            # Dodajemy parę do historii
            # UWAGA: LangChain używa listy par (Pytanie, Odpowiedź), a nie płaskiej listy wiadomości.
            chat_history.append((pytanie, odpowiedz_czysta))

    # KROK 4: Generowanie odpowiedzi Anabelli
    with st.chat_message("assistant"):
        with st.spinner("Anabella analizuje wiedzę i Twój profil..."):

            # STRZAŁ DO AI
            response = qa_chain.invoke({
                "question": prompt,
                "chat_history": chat_history,
                "profil_uzytkownika": wiedza_o_userze  # <-- Tu przekazujemy choroby/wiek
            })

            result_text = response['answer']

            # Wyciąganie źródeł (żeby wiedzieć, z którego pliku to wzięła)
            sources = []
            if 'source_documents' in response:
                for doc in response['source_documents']:
                    md = doc.metadata
                    # Szukamy pola 'source' (może być głęboko zagnieżdżone)
                    zrodlo = md.get('source')
                    if not zrodlo and 'metadata' in md and isinstance(md['metadata'], dict):
                        zrodlo = md['metadata'].get('source')
                    sources.append(zrodlo or 'nieznany')

            # Formatowanie stopki ze źródłami
            source_info = f"\n\n---\n*(Źródło: {', '.join(set(sources))})*" if sources else ""
            final_answer = result_text + source_info

            st.markdown(final_answer)

    # KROK 5: Zapisanie odpowiedzi w historii sesji
    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer})
