import cv2

from utils.detector import detect_face
from utils.preprocess import preprocess_face
from utils.predict import predict_drowsiness

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    face, box = detect_face(frame)

    if box is not None:

        x, y, w, h = box

        processed = preprocess_face(face)

        status, confidence, fatigue = predict_drowsiness(processed)

        # Draw face box
        color = (0,255,0)

        if status == "Drowsy":
            color = (0,0,255)

        cv2.rectangle(frame,
                      (x,y),
                      (x+w,y+h),
                      color,
                      2)

        cv2.putText(frame,
                    f"{status} ({confidence:.1f}%)",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()