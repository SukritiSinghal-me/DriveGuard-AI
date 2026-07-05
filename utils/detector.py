import cv2

# ---------------------------------
# Load Haar Cascade Classifier
# ---------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if cascade loaded successfully
if face_detector.empty():
    raise IOError("❌ Failed to load Haar Cascade XML file.")


def detect_face(frame):
    """
    Detects the largest face in the given frame.

    Parameters
    ----------
    frame : numpy.ndarray
        Input BGR image

    Returns
    -------
    face : numpy.ndarray or None
        Cropped face image

    coords : tuple or None
        (x, y, w, h)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # No face detected
    if len(faces) == 0:
        return None, None

    # Select the largest face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Crop face
    face = frame[y:y+h, x:x+w]

    return face, (x, y, w, h)