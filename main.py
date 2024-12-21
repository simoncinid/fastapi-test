from typing import List, Optional
import os
import asyncio

from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

# Leggi la chiave API dalla variabile d'ambiente
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variabile d'ambiente OPENAI_API_KEY non è impostata.")

openai.api_key = api_key

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def get_completion(message: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    return completion.choices[0].message['content']

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Esegui la chiamata sincrona in un thread separato
        response = await asyncio.to_thread(get_completion, request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    return {"message": "L'endpoint di test funziona correttamente"}
