from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estrai le variabili necessarie dall'ambiente
api_key = os.getenv("OPENAI_API_KEY")

# Verifica che la variabile sia presente
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è impostata nelle variabili d'ambiente di Render.com.")

# Inizializza il client OpenAI
openai.api_key = api_key

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
class OpenAIRequest(BaseModel):
    prompt: str

class OpenAIResponse(BaseModel):
    response: str

# Endpoint di test
@app.get("/test")
async def test():
    return {"message": "API funzionante"}

# Endpoint per testare le funzionalità di OpenAI
@app.post("/openai-test", response_model=OpenAIResponse)
async def openai_test(request: OpenAIRequest):
    try:
        # Crea una richiesta di completamento del chat
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.prompt}
            ]
        )
        # Estrai la risposta del modello
        assistant_reply = response['choices'][0]['message']['content']
        logger.info(f"Assistant response: {assistant_reply}")
        return OpenAIResponse(response=assistant_reply)
    except openai.error.OpenAIError as e:
        logger.error(f"Errore OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Errore imprevisto:")
        raise HTTPException(status_code=500, detail=str(e))
