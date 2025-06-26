def get_app_name() -> str:
    return "Moodly"

def get_face_detector_prototext_file() -> str:
    return "models/deploy.prototxt"


def get_face_detector_caffe_model() -> str:
    return "models/face_detector.caffemodel"


def get_facial_emotion_recognition_model() -> str:
    return "models/facial_emotion_recognition.h5"


def get_facial_emotion_recognition_model_classes() -> list[str]:
    return ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
