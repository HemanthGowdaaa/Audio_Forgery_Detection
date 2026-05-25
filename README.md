# Audio Copy-Move Forgery Detection using MFCC + SVM

This project detects audio copy-move forgery using:

- MFCC Feature Extraction
- Support Vector Machine (SVM)
- Traditional Machine Learning

---

# Dataset

Use TIMIT dataset `.wav` files.

Place original audio files inside:

dataset/original/

Example:

dataset/original/sample1.wav
dataset/original/sample2.wav

---

# Install Requirements

pip install -r requirements.txt

---

# Run Project

python main.py

---

# What the Project Does

1. Loads original audio files
2. Creates forged audio automatically
3. Extracts MFCC features
4. Trains SVM classifier
5. Evaluates detection performance

---

# Labels

0 = Original Audio  
1 = Forged Audio

---

# Features Used

- 13 MFCC coefficients
- Mean
- Standard Deviation

Final feature vector size:

26 features

---

# Model Used

SVM with:
- RBF kernel
- StandardScaler

---

# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Suitable For

- Final Year Project
- Research Prototype
- Beginner Learning