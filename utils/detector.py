import cv2

# Load OpenCV Haar Cascade
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame):
    """
    Detects the largest face in the frame.

    Returns:
        face_image
        (x, y, w, h)
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None

    # Largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    x, y, w, h = faces[0]

    face = frame[y:y+h, x:x+w]

    return face, (x, y, w, h)