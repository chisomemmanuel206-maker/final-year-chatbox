from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.inference.response_generator import ResponseGenerator
from app.utility.download_model import download_model

app = FastAPI()

# Initialize bot
bot = ResponseGenerator()


# ✅ Request schema
class ChatRequest(BaseModel):
    message: str


# ✅ CORS (for Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Run model download ONLY when server starts (NOT on import)
@app.on_event("startup")
def startup_event():
    download_model()


# ✅ Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    result = bot.generate(req.message)

    return {
        "emotion": result.get("emotion", "unknown"),
        "response": result.get("response", "")
    }


# ✅ Health check (VERY IMPORTANT for Render)
@app.get("/")
def root():
    return {"status": "API is running 🚀"}