import numpy as np
import tensorflow as tf

# Load model once
model = tf.keras.models.load_model("models/best_model.keras")

print("✅ Model Loaded Successfully")


def predict_drowsiness(face):

    prediction = model.predict(face, verbose=0)[0][0]

    confidence = prediction * 100

    if prediction >= 0.5:
        status = "Drowsy"
    else:
        status = "Alert"

    fatigue = int(confidence)

    return status, confidence, fatigue