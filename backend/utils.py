import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 300

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)
