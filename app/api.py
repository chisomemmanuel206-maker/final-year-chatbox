from fastapi import FastAPI
from pydantic import BaseModel
from app.inference.response_generator import ResponseGenerator
from fastapi.middleware.cors import CORSMiddleware
from app.utility.download_model import download_model

app = FastAPI()
bot = ResponseGenerator()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    result = bot.generate(req.message)
    return {
        "emotion": result["emotion"],
        "response": result["response"]
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
download_model()