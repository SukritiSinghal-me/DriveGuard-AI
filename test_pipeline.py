import cv2

from utils.detector import detect_face
from utils.preprocess import preprocess_face
from utils.predict import predict_drowsiness
from utils.alarm import start_alarm, stop_alarm

# Open Webcam
cap = cv2.VideoCapture(0)

print("Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect Face
    face, coords = detect_face(frame)

    if face is not None:

        # Preprocess
        processed_face = preprocess_face(face)

        # Prediction
        status, confidence, fatigue = predict_drowsiness(processed_face)

        x, y, w, h = coords

        # Box Color
        if status == "Alert":
            color = (0, 255, 0)
            stop_alarm()
        else:
            color = (0, 0, 255)
            start_alarm()

        # Draw Face Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Status
        cv2.putText(
            frame,
            f"{status}",
            (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}%",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Fatigue
        cv2.putText(
            frame,
            f"Fatigue : {fatigue}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    else:

        stop_alarm()

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
stop_alarm()
cv2.destroyAllWindows()