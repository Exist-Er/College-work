import os
import pickle
import pandas as pd
import cv2
import face_recognition
from datetime import datetime
from utils import get_all_face_embeddings

MODEL_PATH = "models/face_classifier.pkl"
TRAIN_DIR = "data/training"
OUTPUT_DIR = "output"
CONFIDENCE_THRESHOLD = 0.6


def mark_attendance(image_path):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    all_students = sorted(
        [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    )

    print(f"Processing group photo: {image_path}")
    face_locations, face_encodings = get_all_face_embeddings(image_path)

    present_students = set()

    if not face_encodings:
        print("No faces detected in the photo.")
    else:
        for encoding in face_encodings:
            probs = clf.predict_proba([encoding])[0]
            max_idx = probs.argmax()
            confidence = probs[max_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                name = clf.classes_[max_idx]
                present_students.add(name)
                print(f"Detected: {name} ({confidence:.2f})")
            else:
                print(f"Detected unknown face (Confidence: {confidence:.2f})")

    attendance_data = []
    for student in all_students:
        status = "Present" if student in present_students else "Absent"
        attendance_data.append(
            {
                "Student Name": student,
                "Status": status,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    df = pd.DataFrame(attendance_data)

    # Save to CSV
    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(save_path, index=False)

    print("-" * 30)
    print(f"Attendance Report saved to: {save_path}")
    print(df)
    return save_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python attendance.py <path_to_group_photo>")
    else:
        mark_attendance(sys.argv[1])
