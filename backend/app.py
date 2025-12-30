from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from gradcam import generate_gradcam
from utils import preprocess_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model/stroke_model.keras")
class_names = np.load("model/labels.npy")

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
    gradcam_base64 = buffer.tobytes().hex()

    return {
        "prediction": class_names[pred_class],
        "confidence": round(confidence * 100, 2),
        "gradcam": gradcam_base64
    }
