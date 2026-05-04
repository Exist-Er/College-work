import os
import pickle
import numpy as np
from sklearn import svm
from utils import get_face_embeddings

TRAIN_DIR = "data/training"
MODEL_PATH = "models/face_classifier.pkl"


def train_system():
    X = []
    y = []

    print("Starting embedding extraction...")

    for student_name in os.listdir(TRAIN_DIR):
        student_path = os.path.join(TRAIN_DIR, student_name)
        if not os.path.isdir(student_path):
            continue

        print(f"Processing student: {student_name}")
        for image_name in os.listdir(student_path):
            image_path = os.path.join(student_path, image_name)
            embedding = get_face_embeddings(image_path)

            if embedding is not None:
                X.append(embedding)
                y.append(student_name)
            else:
                print(f"  [!] No face detected in {image_name}")

    if not X:
        print(
            "No training data found. Please add student photos to data/training/<student_name>/"
        )
        return

    print(f"Training SVM with {len(X)} samples...")
    clf = svm.SVC(gamma="scale", probability=True)
    clf.fit(X, y)

    # Saving the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_system()
