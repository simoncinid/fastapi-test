from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import asyncio
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estrai le variabili necessarie dall'ambiente
api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

# Verifica che le variabili siano presenti
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è impostata nelle variabili d'ambiente di Render.com.")
if not assistant_id:
    raise ValueError("La variabile ASSISTANT_ID non è impostata nelle variabili d'ambiente di Render.com.")

# Inizializza l'app FastAPI
app = FastAPI()

# Configura il middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Usato per eseguire con il server React
        "https://nickchatrath.vercel.app",  # Dominio del frontend su Vercel
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definizione dei modelli Pydantic
class RunStatus(BaseModel):
    run_id: Optional[str]
    thread_id: Optional[str]
    status: Optional[str]
    required_action: Optional[dict]
    last_error: Optional[dict]

class ThreadMessage(BaseModel):
    content: str
    role: str
    hidden: bool
    id: str
    created_at: int

class Thread(BaseModel):
    messages: List[ThreadMessage]

class CreateMessage(BaseModel):
    content: str

# Configura l'API Client di httpx
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

base_url = "https://api.openai.com/v1/assistants"

# Endpoint di test
@app.get("/test")
async def test():
    return {"message": "API funzionante"}

# Endpoint per creare un nuovo thread
@app.post("/api/new", response_model=RunStatus)
async def post_new():
    try:
        async with httpx.AsyncClient() as client:
            # Crea un nuovo thread
            create_thread_url = f"{base_url}/{assistant_id}/threads"
            # Aggiungi un nome al thread per evitare payload vuoti
            thread_creation_payload = {"name": "New Thread"}
            response = await client.post(create_thread_url, headers=headers, json=thread_creation_payload)
            logger.info(f"POST {create_thread_url} - Status Code: {response.status_code}")
            logger.debug(f"Response: {response.text}")
            
            if response.status_code not in (200, 201):
                # Log dell'errore
                logger.error(f"Errore nella creazione del thread: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            thread_data = response.json()
            thread_id = thread_data.get("id")
            run_id = thread_data.get("run_id")
            status = thread_data.get("status")
            required_action = thread_data.get("required_action")
            last_error = thread_data.get("last_error")
            
            if not thread_id:
                raise HTTPException(status_code=500, detail="Thread ID non trovato nella risposta dell'API.")
            
            # Invia un messaggio iniziale nascosto
            send_message_url = f"{base_url}/{assistant_id}/threads/{thread_id}/messages"
            hidden_message = {
                "content": "Greet the user and tell it about yourself and ask it what it is looking for.",
                "role": "system",  # Potrebbe essere 'system' o 'user' a seconda dell'API
                "metadata": {
                    "type": "hidden"
                }
            }
            send_message_response = await client.post(send_message_url, headers=headers, json=hidden_message)
            logger.info(f"POST {send_message_url} - Status Code: {send_message_response.status_code}")
            logger.debug(f"Response: {send_message_response.text}")
            
            if send_message_response.status_code not in (200, 201):
                logger.error(f"Errore nell'invio del messaggio nascosto: {send_message_response.status_code} - {send_message_response.text}")
                raise HTTPException(status_code=send_message_response.status_code, detail=send_message_response.json())
            
            # Potrebbe essere necessario aggiornare run_id o status dopo l'invio del messaggio
            # A seconda della struttura della risposta, ad esempio:
            # run_id = send_message_response.json().get("run_id", run_id)
            # status = send_message_response.json().get("status", status)
            
            return RunStatus(
                run_id=run_id,
                thread_id=thread_id,
                status=status,
                required_action=required_action,
                last_error=last_error
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTPStatusError: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))
