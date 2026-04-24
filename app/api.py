from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.inference.response_generator import ResponseGenerator
from app.utility.download_model import download_model

app = FastAPI()

# ❌ DO NOT initialize here
bot = None


# ✅ Request schema
class ChatRequest(BaseModel):
    message: str


# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ FIXED startup
@app.on_event("startup")
def startup_event():
    global bot

    print("🔄 Downloading model...")
    download_model()

    print("🧠 Loading model...")
    bot = ResponseGenerator()

    print("✅ Model ready!")


# ✅ Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    if bot is None:
        return {"error": "Model not ready yet"}

    result = bot.generate(req.message)

    return {
        "emotion": result.get("emotion", "unknown"),
        "response": result.get("response", "")
    }


# ✅ Health check
@app.get("/")
def root():
    return {"status": "API is running 🚀"}