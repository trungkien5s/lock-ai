from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras import layers, Model
from .config import PERSON_MODEL_PATH, FACE_MODEL_PATH, EMOTION_MODEL_PATH, ACTION_MODEL_PATH, EMBED_MODEL_PATH

def load_models():
    """
    Tải mô hình YOLO cho nhận diện người và khuôn mặt, mô hình TFLite cho nhận diện cảm xúc và hành vi,
    và mô hình TensorFlow/Keras cho trích xuất embedding khuôn mặt.
    / Load YOLO models for person and face detection, TFLite models for emotion and action recognition,
    and TensorFlow/Keras model for face embedding extraction.
    """
    # Tải mô hình YOLO / Load YOLO models
    person_model = YOLO(PERSON_MODEL_PATH)   # Mô hình nhận diện người / Person model
    face_model = YOLO(FACE_MODEL_PATH)       # Mô hình nhận diện khuôn mặt / Face model
    
    # Tải mô hình nhận diện cảm xúc TensorFlow Lite / Load TFLite emotion recognition model
    emotion_interpreter = tf.lite.Interpreter(model_path=EMOTION_MODEL_PATH)
    emotion_interpreter.allocate_tensors()
    
    # Tải mô hình nhận diện hành vi TensorFlow Lite / Load TFLite action recognition model
    action_interpreter = tf.lite.Interpreter(model_path=ACTION_MODEL_PATH)
    action_interpreter.allocate_tensors()

    # Tải mô hình TensorFlow Lite cho trích xuất embedding khuôn mặt
    # Load TensorFlow Lite model for face embedding extraction
    face_embedding_interpreter = tf.lite.Interpreter(model_path='models/face_embedding_model_256.tflite') # Đổi đường dẫn nếu cần
    face_embedding_interpreter.allocate_tensors()

    return person_model, face_model, emotion_interpreter, action_interpreter, face_embedding_interpreter

def get_emotion_model_details(interpreter):
    """Lấy thông tin đầu vào và đầu ra của mô hình cảm xúc
    / Get input and output details for the emotion model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def get_action_model_details(interpreter):
    """Lấy thông tin đầu vào và đầu ra của mô hình hành vi
    / Get input and output details for the action model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def get_face_embedding_model_details(interpreter):
    """Lấy thông tin đầu vào và đầu ra của mô hình embedding khuôn mặt
    / Get input and output details for the face embedding model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details