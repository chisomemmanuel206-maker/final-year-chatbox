from dotenv import load_dotenv
import os

# 👇 FORCE LOAD .env
load_dotenv()

print("KEY:", os.getenv("GEMINI_API_KEY"))