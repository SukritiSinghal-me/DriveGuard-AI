import cv2
from utils.predict import predict_drowsiness

image_path = "dataset/test/Drowsy/A0014.png"

img = cv2.imread(image_path)

if img is None:
    print("Image not found:", image_path)
    exit()

img = cv2.resize(img, (128,128))
img = img / 255.0

status, confidence, fatigue = predict_drowsiness(img)

print("Status :", status)
print("Confidence :", confidence)
print("Fatigue :", fatigue)