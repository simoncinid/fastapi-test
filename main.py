from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import time
import logging

# Configura il logger
logging.basicConfig(level=logging.DEBUG)

# Configura le variabili di ambiente per OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("La variabile di ambiente OPENAI_API_KEY deve essere configurata.")

openai.api_key = OPENAI_API_KEY

# Inizializza l'app FastAPI
app = FastAPI()

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia con l'URL del tuo frontend in produzione
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modello per la richiesta
class OpenAIRequest(BaseModel):
    user_id: str
    prompt: str

# Memorizza lo stato dei thread per ogni utente e assistente
active_threads = {}

@app.post("/chat/{assistant_id}")
async def chat(assistant_id: str, request: OpenAIRequest):
    """Gestisce la conversazione con l'assistente specificato."""
    user_id = request.user_id
    prompt = request.prompt

    try:
        logging.debug(f"Richiesta ricevuta: user_id={user_id}, assistant_id={assistant_id}, prompt={prompt}")

        # Chiave univoca per thread per utente e assistente
        thread_key = f"{user_id}:{assistant_id}"

        # Step 1: Verifica se esiste già un thread per l'utente e l'assistente
        if thread_key not in active_threads:
            logging.debug(f"Nessun thread trovato per user_id={user_id} e assistant_id={assistant_id}. Creo un nuovo thread.")
            # Crea un nuovo thread alla prima richiesta
            chat = openai.Client().beta.threads.create(
                messages=[{"role": "user", "content": prompt}]
            )
            active_threads[thread_key] = chat.id
        else:
            # Aggiungi un nuovo messaggio al thread esistente
            chat_id = active_threads[thread_key]
            logging.debug(f"Thread esistente trovato per user_id={user_id} e assistant_id={assistant_id}: thread_id={chat_id}")
            openai.Client().beta.threads.messages.create(
                thread_id=chat_id, role="user", content=prompt
            )

        # Step 2: Avvia un run con l'ASSISTANT_ID specificato
        logging.debug(f"Avvio del run per thread_id={active_threads[thread_key]} con assistant_id={assistant_id}")
        run = openai.Client().beta.threads.runs.create(
            thread_id=active_threads[thread_key],
            assistant_id=assistant_id
        )

        # Step 3: Polling dello stato del run
        while run.status != "completed":
            logging.debug(f"Stato del run: {run.status}")
            time.sleep(0.5)
            run = openai.Client().beta.threads.runs.retrieve(
                thread_id=active_threads[thread_key], run_id=run.id
            )

        # Step 4: Recupera i messaggi dal thread
        message_response = openai.Client().beta.threads.messages.list(
            thread_id=active_threads[thread_key]
        )
        messages = message_response.data
        logging.debug(f"Messaggi ricevuti per thread_id={active_threads[thread_key]}: {messages}")

        # Step 5: Trova l'ultimo messaggio dell'assistente
        for message in messages[::-1]:  # Itera al contrario per ottenere l'ultimo messaggio
            if message.role == "assistant" and message.run_id == run.id:
                response_text = message.content[0].text.value
                logging.debug(f"Risposta trovata per user_id={user_id} e assistant_id={assistant_id}: {response_text}")
                return {"response": response_text}

        logging.warning(f"Nessuna risposta trovata per user_id={user_id} e assistant_id={assistant_id}")
        return {"response": "Nessuna risposta trovata"}
    except Exception as e:
        logging.error(f"Errore durante la richiesta per user_id={user_id} e assistant_id={assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore durante la richiesta a OpenAI: {str(e)}")
