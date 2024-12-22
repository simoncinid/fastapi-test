from fastapi import FastAPI, HTTPException
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

# Modello per la richiesta
class OpenAIRequest(BaseModel):
    prompt: str

@app.get("/test")
async def test_endpoint():
    """Endpoint di test per verificare se l'API funziona."""
    return {"message": "API funzionante"}

@app.post("/openai-assistant")
async def openai_assistant(request: OpenAIRequest):
    """Interagisce con l'assistente tramite ASSISTANT_ID."""
    try:
        # Crea un nuovo thread
        chat = openai.Client().beta.threads.create(
            messages=[
                {"role": "user", "content": request.prompt}
            ]
        )

        # Avvia un run con ASSISTANT_ID
        run = openai.Client().beta.threads.runs.create(
            thread_id=chat.id,
            assistant_id=ASSISTANT_ID
        )

        # Polling dello stato del run
        while run["status"] != "completed":
            time.sleep(0.5)
            run = openai.Client().beta.threads.runs.retrieve(
                thread_id=chat.id, run_id=run.id
            )

        # Recupera i messaggi
        message_response = openai.Client().beta.threads.messages.list(thread_id=chat.id)
        messages = message_response.data

        if messages:
            latest_message = messages[-1]
            if latest_message["role"] == "assistant":
                return {"response": latest_message["content"]}
        return {"response": "Nessuna risposta trovata"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la richiesta a OpenAI: {str(e)}")
