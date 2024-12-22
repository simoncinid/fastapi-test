from typing import List, Optional

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from openai.types.beta.threads.run import RequiredAction, LastError
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Estrai le variabili necessarie dall'ambiente
api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

# Verifica che le variabili siano presenti
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è impostata nelle variabili d'ambiente di Render.com.")
if not assistant_id:
    raise ValueError("La variabile ASSISTANT_ID non è impostata nelle variabili d'ambiente di Render.com.")

# Inizializza il client OpenAI asincrono
client = AsyncOpenAI(api_key=api_key)

# Stati finali per le runs
run_finished_states = ["completed", "failed", "cancelled", "expired", "requires_action"]

# Definizione dei modelli Pydantic
class RunStatus(BaseModel):
    run_id: str
    thread_id: str
    status: str
    required_action: Optional[RequiredAction]
    last_error: Optional[LastError]

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

# Inizializza l'app FastAPI
app = FastAPI()
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

@app.post("/api/new", response_model=RunStatus)
async def post_new():
    try:
        # Crea un nuovo thread
        thread = await client.beta.threads.create()

        # Invia un messaggio iniziale nascosto
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            content="Greet the user and tell it about yourself and ask it what it is looking for.",
            role="user",
            metadata={
                "type": "hidden"
            }
        )

        # Crea una nuova run associata all'assistente specificato
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        return RunStatus(
            run_id=run.id,
            thread_id=thread.id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threads/{thread_id}/runs/{run_id}", response_model=RunStatus)
async def get_run(thread_id: str, run_id: str):
    try:
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        return RunStatus(
            run_id=run.id,
            thread_id=thread_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threads/{thread_id}/runs/{run_id}/tool", response_model=RunStatus)
async def post_tool(thread_id: str, run_id: str, tool_outputs: List[ToolOutput]):
    try:
        run = await client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=tool_outputs
        )

        return RunStatus(
            run_id=run.id,
            thread_id=thread_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threads/{thread_id}", response_model=Thread)
async def get_thread(thread_id: str):
    try:
        messages = await client.beta.threads.messages.list(
            thread_id=thread_id
        )

        result = [
            ThreadMessage(
                content=message.content[0].text.value if message.content else "",
                role=message.role,
                hidden=("type" in message.metadata and message.metadata["type"] == "hidden"),
                id=message.id,
                created_at=message.created_at
            )
            for message in messages.data
        ]

        return Thread(
            messages=result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threads/{thread_id}", response_model=RunStatus)
async def post_thread(thread_id: str, message: CreateMessage):
    try:
        # Crea un nuovo messaggio utente nel thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            content=message.content,
            role="user"
        )

        # Crea una nuova run associata all'assistente specificato
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        return RunStatus(
            run_id=run.id,
            thread_id=thread_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
