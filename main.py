from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import time

# Configura le variabili di ambiente per OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

if not OPENAI_API_KEY or not ASSISTANT_ID:
    raise EnvironmentError("Le variabili di ambiente OPENAI_API_KEY e ASSISTANT_ID devono essere configurate.")

openai.api_key = OPENAI_API_KEY

# Inizializza l'app FastAPI
app = FastAPI()

# Configurazione CORS per consentire richieste dal frontend
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

# Memorizza lo stato dei thread per ogni utente (in memoria, usa un DB in produzione)
active_threads = {}

@app.post("/chat")
async def chat(request: OpenAIRequest):
    """Gestisce la conversazione con l'assistente OpenAI."""
    user_id = request.user_id
    prompt = request.prompt

    try:
        # Verifica se esiste già un thread per l'utente
        if user_id not in active_threads:
            # Crea un nuovo thread
            chat = openai.Client().beta.threads.create(
                messages=[{"role": "user", "content": prompt}]
            )
            active_threads[user_id] = chat.id
        else:
            # Recupera il thread esistente
            chat_id = active_threads[user_id]
            openai.Client().beta.threads.messages.create(
                thread_id=chat_id, role="user", content=prompt
            )

        # Avvia un run con ASSISTANT_ID
        run = openai.Client().beta.threads.runs.create(
            thread_id=active_threads[user_id],
            assistant_id=ASSISTANT_ID
        )

        # Polling dello stato del run
        while run.status != "completed":
            time.sleep(0.5)
            run = openai.Client().beta.threads.runs.retrieve(
                thread_id=active_threads[user_id], run_id=run.id
            )

        # Recupera i messaggi
        message_response = openai.Client().beta.threads.messages.list(
            thread_id=active_threads[user_id]
        )
        messages = message_response.data

        # Trova il messaggio dell'assistente
        for message in messages[::-1]:  # Itera al contrario per ottenere l'ultimo messaggio
            if message.role == "assistant":
                response_text = message.content[0].text.value
                return {"response": response_text}

        return {"response": "Nessuna risposta trovata"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la richiesta a OpenAI: {str(e)}")
