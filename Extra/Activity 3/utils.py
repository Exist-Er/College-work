import os
import cv2
import face_recognition


def get_face_embeddings(image_path):
    """Detects faces and returns embeddings for the first face found."""
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0] if face_encodings else None


def get_all_face_embeddings(image_path):
    """Detects all faces and returns locations and embeddings."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings
