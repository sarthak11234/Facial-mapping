# Facial Landmark Detection with PyTorch

## 📌 Overview
This project implements **facial landmark detection** using PyTorch and the **iBUG 300-W Large Face Landmark Dataset**.
The model predicts **68 facial keypoints** (eyes, eyebrows, nose, lips, jawline) from grayscale images, which can be used for:
- Face alignment
- Expression recognition
- Augmented reality filters
- Face tracking in video

---

## ✨ Features
- **Custom data preprocessing pipeline**:
  - Face cropping
  - Resizing to `224×224`
  - Random rotation (±10°)
  - Color jitter
  - Normalization
- **Custom PyTorch Dataset** for landmark files (`.pts`) and images
- **ResNet18-based model** adapted for:
  - **Grayscale input**
  - **136 output values** (68 landmarks × 2 coordinates)
- **Training & validation loops** with Mean Squared Error loss
- **Visualization** of predicted landmarks on face images

---

## 📂 Dataset
- **Name:** iBUG 300-W Large Face Landmark Dataset
- **Download link:** [http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)
- **Contents:**
  - Face images in different poses and lighting conditions
  - `.pts` files containing 68 facial keypoints per image

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/face-landmark-detection.git
cd face-landmark-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the Dataset
```bash
wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
tar -xvzf ibug_300W_large_face_landmark_dataset.tar.gz
```

### 4️⃣ Train the Model
```bash
python train.py
```

### 5️⃣ Run the Demo Notebook
Open **`demo.ipynb`** in Jupyter Notebook or VS Code and execute the cells.

---

## 📊 Results

### Example Predictions
| Input Image | Predicted Landmarks |
|-------------|---------------------|
| ![](results/sample1.jpg) | ![](results/sample1_pred.jpg) |
| ![](results/sample2.jpg) | ![](results/sample2_pred.jpg) |

---

## 📈 Training Loss Curve
*(Add your loss curve image here)*

---

## 🛠 Requirements
- Python 3.8+
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Pillow
- imutils
- scikit-image

Install them via:
```bash
pip install -r requirements.txt
```

---

## 🛠 Project Structure
```plaintext
face-landmark-detection/
│
├── dataset.py         # Dataset & transformations
├── model.py           # ResNet18 model definition
├── train.py           # Training & validation loop
├── utils.py           # Helper functions (plotting, printing)
├── demo.ipynb         # Jupyter Notebook demo
├── requirements.txt   # Dependencies
├── README.md          # Documentation
└── results/           # Sample predictions
```

---
