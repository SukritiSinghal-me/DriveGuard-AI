import tensorflow as tf

# -----------------------------
# Load Trained Model
# -----------------------------
model = tf.keras.models.load_model("models/best_model.keras")

print("✅ Model Loaded Successfully")


# -----------------------------
# Prediction Function
# -----------------------------
def predict_drowsiness(face):
    """
    Predicts whether the driver is Alert or Drowsy.

    Parameters:
        face : Preprocessed face image
               Shape -> (1, 128, 128, 3)

    Returns:
        status      : "Alert" or "Drowsy"
        confidence  : Prediction confidence (%)
        fatigue     : Fatigue score (0-100)
    """

    # Model prediction (0 to 1)
    prediction = model.predict(face, verbose=0)[0][0]

    # -----------------------------
    # Drowsy
    # -----------------------------
    if prediction >= 0.5:

        status = "Drowsy"

        confidence = prediction * 100

        fatigue = int(confidence)

    # -----------------------------
    # Alert
    # -----------------------------
    else:

        status = "Alert"

        confidence = (1 - prediction) * 100

        fatigue = int(prediction * 100)

    return status, confidence, fatigue