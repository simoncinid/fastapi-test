from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import asyncio

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
    run_id: str
    thread_id: str
    status: str
    required_action: Optional[dict]  # Modifica in base alla struttura effettiva
    last_error: Optional[dict]       # Modifica in base alla struttura effettiva

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
            response = await client.post(create_thread_url, headers=headers, json={})
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            thread_data = response.json()
            thread_id = thread_data.get("id")
            run_id = thread_data.get("run_id")
            status = thread_data.get("status")
            required_action = thread_data.get("required_action")
            last_error = thread_data.get("last_error")
            
            # Invia un messaggio iniziale nascosto
            send_message_url = f"{base_url}/{assistant_id}/threads/{thread_id}/messages"
            hidden_message = {
                "content": "Greet the user and tell it about yourself and ask it what it is looking for.",
                "role": "user",
                "metadata": {
                    "type": "hidden"
                }
            }
            await client.post(send_message_url, headers=headers, json=hidden_message)
            
            return RunStatus(
                run_id=run_id,
                thread_id=thread_id,
                status=status,
                required_action=required_action,
                last_error=last_error
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per verificare una run
@app.get("/api/threads/{thread_id}/runs/{run_id}", response_model=RunStatus)
async def get_run(thread_id: str, run_id: str):
    try:
        async with httpx.AsyncClient() as client:
            retrieve_run_url = f"{base_url}/{assistant_id}/threads/{thread_id}/runs/{run_id}"
            response = await client.get(retrieve_run_url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            run_data = response.json()
            return RunStatus(
                run_id=run_data.get("id"),
                thread_id=thread_id,
                status=run_data.get("status"),
                required_action=run_data.get("required_action"),
                last_error=run_data.get("last_error")
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per inviare tool outputs
@app.post("/api/threads/{thread_id}/runs/{run_id}/tool", response_model=RunStatus)
async def post_tool(thread_id: str, run_id: str, tool_outputs: List[CreateMessage]):
    try:
        async with httpx.AsyncClient() as client:
            submit_tool_url = f"{base_url}/{assistant_id}/threads/{thread_id}/runs/{run_id}/tool"
            # Adatta tool_outputs al formato richiesto dall'API
            tool_outputs_payload = [{"tool_id": "tool_abc", "output": msg.content} for msg in tool_outputs]
            response = await client.post(submit_tool_url, headers=headers, json=tool_outputs_payload)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            run_data = response.json()
            return RunStatus(
                run_id=run_data.get("id"),
                thread_id=thread_id,
                status=run_data.get("status"),
                required_action=run_data.get("required_action"),
                last_error=run_data.get("last_error")
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per recuperare i messaggi di un thread
@app.get("/api/threads/{thread_id}", response_model=Thread)
async def get_thread(thread_id: str):
    try:
        async with httpx.AsyncClient() as client:
            list_messages_url = f"{base_url}/{assistant_id}/threads/{thread_id}/messages"
            response = await client.get(list_messages_url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            messages_data = response.json()
            messages = [
                ThreadMessage(
                    content=msg.get("content", ""),
                    role=msg.get("role", ""),
                    hidden=msg.get("metadata", {}).get("type") == "hidden",
                    id=msg.get("id", ""),
                    created_at=msg.get("created_at", 0)
                )
                for msg in messages_data.get("data", [])
            ]
            
            return Thread(messages=messages)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint per inviare un nuovo messaggio in un thread
@app.post("/api/threads/{thread_id}", response_model=RunStatus)
async def post_thread(thread_id: str, message: CreateMessage):
    try:
        async with httpx.AsyncClient() as client:
            send_message_url = f"{base_url}/{assistant_id}/threads/{thread_id}/messages"
            user_message = {
                "content": message.content,
                "role": "user"
            }
            response = await client.post(send_message_url, headers=headers, json=user_message)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            
            run_data = response.json()
            return RunStatus(
                run_id=run_data.get("run_id"),
                thread_id=thread_id,
                status=run_data.get("status"),
                required_action=run_data.get("required_action"),
                last_error=run_data.get("last_error")
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
