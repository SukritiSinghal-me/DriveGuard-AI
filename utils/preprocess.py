
import cv2
import numpy as np


IMG_SIZE = 128


def preprocess_face(face):

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=0)

    return face