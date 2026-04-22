import os
import gdown

MODEL_PATH = "models/emotion_classifier"
FILE_ID = "1ihs31LbNNfD390lcsMbdJFcT2SIuWOs-"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    print("Model downloaded successfully.")