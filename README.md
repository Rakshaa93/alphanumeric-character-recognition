# Alphanumeric Character Recognition using CNN

## Project Overview
This project implements a **single Convolutional Neural Network (CNN)** that can recognize **handwritten digits (0–9)** and **uppercase alphabets (A–Z)**.

The model combines:
- **MNIST dataset** for digits
- **EMNIST Letters dataset** (via TensorFlow Datasets) for alphabets

The goal is to demonstrate an **end-to-end deep learning pipeline**, including dataset handling, preprocessing, model training, evaluation, and prediction on custom images.

---

## Tech Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow (Keras API)  
- **Datasets:** MNIST, EMNIST (via TensorFlow Datasets)  
- **Libraries:** NumPy, Matplotlib  

---

##  Dataset Used

### MNIST (Digits)
- Digits: `0–9`
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 (grayscale)

### EMNIST Letters
- Alphabets: `A–Z`
- Loaded using `tensorflow_datasets`
- Image size: 28 × 28 (grayscale)

### Class Mapping
| Label Range | Class |
|-----------|-------|
| 0 – 9 | Digits (0–9) |
| 10 – 35 | Alphabets (A–Z) |

Total classes = **36**

---

##  Model Architecture
The CNN architecture used in this project:

1. **Conv2D (32 filters, 3×3, ReLU)**
2. **MaxPooling2D (2×2)**
3. **Conv2D (64 filters, 3×3, ReLU)**
4. **MaxPooling2D (2×2)**
5. **Flatten**
6. **Dense (128 units, ReLU)**
7. **Dense (36 units, Softmax)**

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  

---

##  How to Run the Project

### Clone the Repository
```bash
git clone <your-repo-url>
cd alphanumeric-character-recognition
```

### Create & Activate Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train.py
```

---

## Predict Using Your Own Image
- Place your image inside the images/ folder
- Update the image path inside train.py
```bash
img_path = r"Your path"
```

## Image Guidelines
- Single character (digit or alphabet)
- Black character on white background
- PNG or JPG format

---

## Results
- Digits-only accuracy (MNIST): ~99%
- Combined digits + alphabets accuracy: ~85–90%
- Lower accuracy for alphabets is expected due to similar shapes (e.g., 0 vs O, 1 vs I).

---

## Future Enhancements
- Support lowercase alphabets
- Add confusion matrix visualization
- Build a Streamlit web application
- Deploy model as an API