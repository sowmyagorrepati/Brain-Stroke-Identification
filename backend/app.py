from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import gdown

from gradcam import generate_gradcam
from utils import preprocess_image

# --------------------------------------------------
# Model configuration
# --------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "stroke_model.keras")
MODEL_URL = "https://drive.google.com/uc?id=1wMKmUJ5negmjHJLUGLzjmSjo-qgZCJzE"
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

# --------------------------------------------------
# Download model if not present
# --------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --------------------------------------------------
# Load model and labels
# --------------------------------------------------
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Brain Stroke Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # GitHub Pages
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_array = preprocess_image(img)

    preds = model.predict(img_array)[0]
    pred_class = int(np.argmax(preds))
    confidence = float(preds[pred_class])

    gradcam_img = generate_gradcam(model, img_array, pred_class)

    _, buffer = cv2.imencode(".png", gradcam_img)
    gradcam_hex = buffer.tobytes().hex()

    return {
        "prediction": class_names[pred_class],
        "confidence": round(confidence * 100, 2),
        "gradcam": gradcam_hex
    }
@app.get("/")
def health():
    return {"status": "Brain Stroke Detection API is running"}
