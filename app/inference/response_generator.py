import json
import random

from app.inference.emotion_predictor import EmotionPredictor
from app.inference.gemini_predictor import GeminiPredictor
from app.inference.safety_filter import SafetyFilter
from app.inference.conversation_memory import ConversationMemory
from app.inference.response_cleaner import ResponseCleaner


class ResponseGenerator:

    def __init__(self):

        self.emotion_model = EmotionPredictor()
        self.dialog_model = GeminiPredictor()
        self.safety_filter = SafetyFilter()
        self.memory = ConversationMemory()
        self.cleaner = ResponseCleaner()

        with open("data/coping_strategies.json", "r", encoding="utf-8") as f:
            self.coping_strategies = json.load(f)

    def get_primary_emotion(self, emotions):
        return emotions[0]["emotion"] if emotions else "neutral"

    # --------------------------------
    # HUMAN FLOW CONTROL (FIXED)
    # --------------------------------
    def humanize(self, response, emotion):

        # remove unwanted tags
        bad = ["User:", "Bot:", "Assistant:", "AI:"]
        for b in bad:
            response = response.replace(b, "")

        response = response.strip()

        # 🔥 LIMIT TO 1–4 SENTENCES
        sentences = response.split(". ")
        if len(sentences) > 4:
            response = ". ".join(sentences[:4])
            if not response.endswith("."):
                response += "."

        # limit multiple questions
        if response.count("?") > 1:
            response = response.split("?")[0] + "?"

        # emotional shaping (DO NOT override response)
        if emotion == "sadness":
            if "?" not in response:
                response += " Do you want to talk about it?"

        elif emotion == "anxiety":
            if len(response.split()) > 25:
                response = response[:150].rsplit(" ", 1)[0] + "..."

        return response.strip()

    # --------------------------------
    # MAIN GENERATE FUNCTION
    # --------------------------------
    def generate(self, user_input):

        # 1. Emotion detection
        emotions = self.emotion_model.predict_emotions(user_input)
        primary_emotion = self.get_primary_emotion(emotions)
        primary_emotion = primary_emotion.lower().strip()

        # 🔥 OPTIONAL QUICK FIX (improves wrong emotion cases)
        if "fail" in user_input.lower():
            primary_emotion = "sadness"

        # 2. Memory context
        context = self.memory.get_context()

        # 3. AI response
        response = self.dialog_model.generate_response(
            user_input=user_input,
            emotion=primary_emotion,
            context=context
        )

        # 4. Humanize (FIXED)
        response = self.humanize(response, primary_emotion)

        # 5. Coping strategies (optional)
        strategies = self.coping_strategies.get(primary_emotion, [])

        if primary_emotion in ["sadness", "stress", "anxiety"] and strategies:
            if random.random() < 0.4:
                strategy = random.choice(strategies)
                response += f"\n\n💡 {strategy}"

        # 6. Clean response
        response = self.cleaner.clean(response)

        # 7. Safety filter
        response = self.safety_filter.filter_response(user_input, response)

        # 8. Save memory
        self.memory.add_turn(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }