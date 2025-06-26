import cv2
import numpy

def convert_face_to_numpy_array(face: numpy.ndarray) -> numpy.ndarray:
    face_array = cv2.resize(face, (48, 48))
    face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
    face_array = face_array.astype("float32") / 255.0
    face_array = numpy.expand_dims(face_array, axis=-1)
    face_array = numpy.expand_dims(face_array, axis=0)
    return face_array

def get_emotion(emotion: str) -> str:
    return emotion[0].upper() + emotion[1:] if emotion else "Unknown"