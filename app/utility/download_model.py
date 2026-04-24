import os
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1dVpdIhj2PW1ibc8gS7kthor51Ns6r9Gq"
MODEL_DIR = "models/emotion_classifier"

def download_model():
    os.makedirs("models", exist_ok=True)

    # IMPORTANT: prevents re-downloading on every Render restart
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        print("Model already exists. Skipping download.")
        return

    print("Downloading model from Google Drive...")

    gdown.download_folder(
        url=FOLDER_URL,
        output="models",
        quiet=False,
        use_cookies=False
    )

    print("Model downloaded successfully.")