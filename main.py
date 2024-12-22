from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import asyncio
import logging
import json
import time
import httpx

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

# Inizializza il client OpenAI
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)  # Assicurati che la libreria supporti questa classe

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

class CreateMessage(BaseModel):
    thread_id: str
    content: str

class ResponseContent(BaseModel):
    response: str

# Endpoint di test
@app.get("/test")
async def test():
    return {"message": "API funzionante"}

# Endpoint per creare un nuovo thread
@app.post("/api/new", response_model=RunStatus)
async def post_new():
    try:
        # Crea un nuovo thread
        thread = client.beta.threads.create(
            assistant_id=assistant_id,
            name="New Thread"
        )
        logger.info(f"Thread creato con ID: {thread.id}")

        # Invia un messaggio iniziale nascosto
        hidden_message = {
            "content": "Greet the user and tell it about yourself and ask it what it is looking for.",
            "role": "system",
            "metadata": {
                "type": "hidden"
            }
        }
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="system",
            content=hidden_message["content"],
            metadata=hidden_message["metadata"]
        )
        logger.info(f"Messaggio nascosto inviato con ID: {message.id}")

        # Crea un run iniziale
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        logger.info(f"Run creato con ID: {run.id} e stato: {run.status}")

        return RunStatus(
            run_id=run.id,
            thread_id=thread.id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per inviare un messaggio a un thread
@app.post("/api/send", response_model=RunStatus)
async def send_message(message: CreateMessage):
    thread_id = message.thread_id
    user_content = message.content

    try:
        # Invia un messaggio utente
        user_message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_content
        )
        logger.info(f"Messaggio utente inviato con ID: {user_message.id}")

        # Crea un run per ottenere la risposta
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logger.info(f"Run creato con ID: {run.id} e stato: {run.status}")

        return RunStatus(
            run_id=run.id,
            thread_id=thread_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per recuperare lo stato di un run
@app.get("/api/status/{thread_id}/{run_id}", response_model=RunStatus)
async def get_status(thread_id: str, run_id: str):
    try:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        logger.info(f"Run recuperato con ID: {run.id} e stato: {run.status}")

        return RunStatus(
            run_id=run.id,
            thread_id=thread_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per recuperare la risposta dell'assistente
@app.get("/api/response/{thread_id}", response_model=ResponseContent)
async def get_response(thread_id: str):
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id).data
        logger.info(f"Messaggi recuperati per il thread ID: {thread_id}")

        # Trova l'ultimo messaggio dell'assistente
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            raise HTTPException(status_code=404, detail="Nessuna risposta trovata dall'assistente.")

        latest_message = assistant_messages[-1]
        content = latest_message.get("content", "")

        return ResponseContent(response=content)
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))

# Funzione per gestire le azioni richieste (es. function calling)
async def handle_required_action(thread_id: str, run_id: str, required_action: dict):
    try:
        tool_calls = required_action.get("submit_tool_outputs", {}).get("tool_calls", [])
        tool_output_array = []

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_name = tool_call.get("function", {}).get("name")
            function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

            if function_name == "perplexity_search":
                query = function_args.get("query", "")
                output = await perplexity_search(query)
                tool_output_array.append({
                    "tool_call_id": tool_call_id,
                    "output": output
                })

        # Invia gli output delle funzioni
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_output_array
        )
        logger.info(f"Tool outputs inviati per il run ID: {run_id}")
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI durante l'azione richiesta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore nella gestione delle azioni richieste:")
        raise HTTPException(status_code=500, detail=str(e))

# Definizione della funzione perplexity_search
async def perplexity_search(query: str) -> str:
    """
    Esegue una ricerca utilizzando l'API di Perplexity e restituisce i risultati.
    """
    try:
        # Configura l'API di Perplexity (esempio)
        perplexity_api_url = "https://api.perplexity.ai/search"
        payload = {"query": query}
        async with httpx.AsyncClient() as client_httpx:
            response = await client_httpx.post(perplexity_api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Processa i dati come necessario
                return data.get("results", "Nessun risultato trovato.")
            else:
                logger.error(f"Errore nella ricerca Perplexity: {response.status_code} - {response.text}")
                return "Errore nella ricerca Perplexity."
    except Exception as e:
        logger.exception("Errore nella funzione perplexity_search:")
        return "Errore durante l'elaborazione della richiesta."

# Endpoint per gestire l'intero flusso di invio messaggio e risposta
@app.post("/api/converse", response_model=ResponseContent)
async def converse(message: CreateMessage):
    thread_id = message.thread_id
    user_content = message.content

    try:
        # Invia il messaggio utente
        user_message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_content
        )
        logger.info(f"Messaggio utente inviato con ID: {user_message.id}")

        # Crea un run per ottenere la risposta
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logger.info(f"Run creato con ID: {run.id} e stato: {run.status}")

        # Polling per lo stato del run
        while run.status not in ["completed", "failed", "requires_action"]:
            await asyncio.sleep(2)  # Attendere 2 secondi prima del prossimo check
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logger.info(f"Run ID: {run.id} - Stato: {run.status}")

        # Gestisci le azioni richieste se necessario
        if run.status == "requires_action" and run.required_action:
            await handle_required_action(thread_id, run.id, run.required_action)
            # Recupera nuovamente lo stato dopo aver gestito l'azione
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logger.info(f"Run ID: {run.id} - Stato dopo azione: {run.status}")

        if run.status != "completed":
            logger.error(f"Run non completato: {run.status}")
            raise HTTPException(status_code=500, detail=f"Run non completato: {run.status}")

        # Recupera la risposta dell'assistente
        response_message = client.beta.threads.messages.list(thread_id=thread_id).data
        assistant_messages = [msg for msg in response_message if msg.get("role") == "assistant"]
        if not assistant_messages:
            raise HTTPException(status_code=404, detail="Nessuna risposta trovata dall'assistente.")

        latest_message = assistant_messages[-1]
        content = latest_message.get("content", "")

        return ResponseContent(response=content)
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto durante la conversazione:")
        raise HTTPException(status_code=500, detail=str(e))
