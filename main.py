from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # usato per eseguire con il server React
        "https://nickchatrath.vercel.app",  # dominio del frontend su Vercel
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.message},
            ]
        )
        return ChatResponse(response=completion.choices[0].message['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    return {"message": "L'endpoint di test funziona correttamente"}
