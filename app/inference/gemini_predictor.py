import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GeminiPredictor:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.url = "https://openrouter.ai/api/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",

            #  REQUIRED for OpenRouter (fixes many hidden errors)
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Mental Health Chatbot"
        }

    def generate_response(self, user_input, emotion, context):

        prompt = f"""
You are a supportive mental health chatbot.

Emotion detected: {emotion}

Conversation history:
{context}

User: {user_input}

Respond naturally, briefly, and like a human.
"""

        data = {
            #  safer model (auto picks free/available one)
            "model": "openrouter/auto",

            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            result = response.json()

            #  DEBUG (VERY IMPORTANT)
            print("FULL RESPONSE:", result)

            #  SAFE PARSE (fixes your crash)
            if "choices" in result:
                return result["choices"][0]["message"]["content"]

            #  If API returns error instead of choices
            elif "error" in result:
                print("OpenRouter API Error:", result["error"]["message"])
                return "I'm here with you."

            else:
                return "I'm here with you."

        except Exception as e:
            print("OpenRouter Exception:", e)
            return "I'm here with you."