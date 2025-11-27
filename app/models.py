from ultralytics import YOLO


import tflite_runtime.interpreter as tflite
from .config import (
    PERSON_MODEL_PATH,
    FACE_MODEL_PATH,
    EMOTION_MODEL_PATH,
    ACTION_MODEL_PATH,
    EMBED_MODEL_PATH,
)

def load_models():
    # YOLO models
    person_model = YOLO(PERSON_MODEL_PATH)
    face_model = YOLO(FACE_MODEL_PATH)

    # Emotion TFLite
    emotion_interpreter = tflite.Interpreter(model_path=EMOTION_MODEL_PATH)
    emotion_interpreter.allocate_tensors()

    # Action TFLite
    action_interpreter = tflite.Interpreter(model_path=ACTION_MODEL_PATH)
    action_interpreter.allocate_tensors()

    # Face embedding TFLite
    face_embedding_interpreter = tflite.Interpreter(
        model_path=EMBED_MODEL_PATH  # dùng hẳn EMBED_MODEL_PATH từ config
        # hoặc 'models/face_embedding_model_256.tflite' nếu bạn fix cứng
    )
    face_embedding_interpreter.allocate_tensors()

    return (
        person_model,
        face_model,
        emotion_interpreter,
        action_interpreter,
        face_embedding_interpreter,
    )

def get_emotion_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def get_action_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def get_face_embedding_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details