# 🌿 Plant Disease Detection using ResNet50

## 📌 Overview

This project is a **deep learning image classification model** that detects plant diseases from leaf images using transfer learning with ResNet50.

The model is trained on a dataset of plant leaf images and can classify different plant conditions such as healthy leaves and various diseases.

---

## 🧠 Model Architecture

The model uses **ResNet50 (pretrained on ImageNet)** as a feature extractor, followed by custom classification layers.

- Base Model: ResNet50 (frozen layers)
- Global Average Pooling
- Dense Layer (ReLU)
- Output Layer (Softmax)

---

## 📊 Dataset

The dataset consists of labeled images of plant leaves organized into folders by class.

Dataset structure:

dataset/
├── class_1/
├── class_2/
└── ...

The dataset is automatically split using TensorFlow:

- **80% Training**
- **20% Validation**

---

## ▶️ Training the Model

Run:

python train.py

The script will:

- Check if a trained model already exists
- Train a new model if not found
- Save the model as:

plant_disease_model.keras

---

## 🔍 Making Predictions

Run:

python predict.py

---

## 📂 Project Structure

plant-disease-resnet50
│
├── dataset/
├── config.py
├── train.py
├── predict.py
├── plant_disease_model.keras
└── README.md

---

## 📈 Results

The model achieves high accuracy using transfer learning (typically **85–95%** depending on dataset size and training time).

---

## 🚀 Features

- Transfer learning with ResNet50
- Automatic dataset splitting
- Model saving & loading
- Image prediction script
- Clean and modular structure

---

## 🔧 Future Improvements

- Fine-tune ResNet50 layers
- Add data augmentation
- Implement Grad-CAM visualization
- Deploy as a web app

---

## 📚 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 👤 Author

GitHub: https://github.com/Abdalmomen123

---
