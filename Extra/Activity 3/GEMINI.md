# GEMINI.md - Face Detection-Based Attendance System

## Project Overview
This project is an automated attendance system that identifies students from group photos using face recognition. It uses a hybrid approach:
- **Face Detection & Embedding**: Uses the `face_recognition` library (built on `dlib`) to detect faces and extract 128-dimensional embeddings.
- **Classification**: Uses a Support Vector Machine (SVM) classifier from `scikit-learn` to identify students based on their embeddings.
- **Reporting**: Generates attendance reports in CSV format, comparing detected students against a master list derived from the training data directory.

### Tech Stack
- **Language**: Python 3.x
- **Libraries**: `face_recognition`, `opencv-python`, `scikit-learn`, `pandas`, `numpy`, `pickle`.

## Project Structure
- `data/training/`: Contains subdirectories for each student, each holding multiple training images.
- `data/test/`: Contains group photos for attendance marking.
- `models/`: Stores the trained SVM classifier (`face_classifier.pkl`).
- `output/`: Stores generated attendance CSV reports.
- `train.py`: Script to extract embeddings and train the classifier.
- `attendance.py`: Main script to identify faces in group photos and mark attendance.
- `utils.py`: Utility functions for face detection and embedding extraction.

## Building and Running

### Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure CMake and a C++ compiler are installed for `dlib` compilation.*

### Workflow
1. **Training**:
   Prepare student folders in `data/training/` and run:
   ```bash
   python train.py
   ```
2. **Attendance Marking**:
   Run the attendance script on a target image:
   ```bash
   python attendance.py <path_to_image>
   ```

## Development Conventions
- **Data Organization**: The system relies on directory-based labeling. Each folder in `data/training/` must be named after the student it contains.
- **Confidence Threshold**: The system uses a `CONFIDENCE_THRESHOLD` (default: `0.6`) in `attendance.py` to filter out low-confidence predictions (potential unknown faces).
- **Modularity**: Face detection and encoding logic should reside in `utils.py` to maintain separation between training/inference scripts and core image processing.
- **SVM Configuration**: The SVM is trained with `probability=True` to enable confidence-based filtering during inference.
