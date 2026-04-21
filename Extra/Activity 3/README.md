# Face Detection-Based Attendance System

A machine learning system that automates attendance by identifying students in a class group photo.

## 🚀 Getting Started

### 1. Installation
Install the required Python libraries:
```bash
pip install -r requirements.txt
```
*Note: `face_recognition` requires `dlib`. If you face issues, ensure you have CMake and C++ build tools installed on your system.*

### 2. Prepare Training Data
Organize your student photos in the `data/training/` directory. Create one folder per student:
```text
data/training/
├── John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── Jane_Smith/
│   ├── imageA.png
│   └── ...
```
*Aim for 5–15 clear, varied photos per student.*

### 3. Train the Model
Run the training script to extract face embeddings and train the SVM classifier:
```bash
python train.py
```
This will generate `models/face_classifier.pkl`.

### 4. Mark Attendance
Place your class group photo in `data/test/` and run:
```bash
python attendance.py data/test/class_photo.jpg
```
The results will be printed to the console and saved as a CSV in the `output/` directory.

## 📂 Project Structure
- `train.py`: Extracts face encodings and trains the SVM model.
- `attendance.py`: Processes group photos and marks attendance.
- `utils.py`: Contains shared functions for face detection and encoding.
- `models/`: Stores the trained classifier.
- `output/`: Stores generated attendance CSV reports.

## ⚙️ Configuration
You can adjust the `CONFIDENCE_THRESHOLD` in `attendance.py` (default: 0.6). 
- Higher value = More strict (avoids false positives).
- Lower value = More lenient (useful if training data is limited).
