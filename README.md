<div align="center">

# 🤖 AI Facial Expression Recognition Bot
### Real-Time Deep Learning Computer Vision System for Human Affect Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

</div>

---


## 📌 Requirements


AI Facial Expression Recognition Bot

1. Download FER-2013 dataset from Kaggle
2. Place dataset folder as:
   dataset/train/*
   dataset/test/*
3. Install requirements:
   pip install tensorflow keras opencv-python numpy
4. Train model:
   python train_emotion_model.py
5. Run bot:
   python emotion_bot.py
python emotion_bot.py


## 📌 Project Overview

The **AI Facial Expression Recognition Bot** is an end-to-end computer vision and deep learning system capable of detecting human faces and classifying emotional states in real-time from webcam video streams. 

Trained on the standard **FER-2013** dataset, the system utilizes Convolutional Neural Networks (CNN) to extract facial micro-features and accurately predict one of seven universal emotion classes:

* 😠 **Angry**
* 🤢 **Disgust**
* 😨 **Fear**
* 😄 **Happy**
* 😐 **Neutral**
* 😢 **Sad**
* 😲 **Surprise**

---

## ⚙️ System Architecture & Pipeline

```text
  [ Webcam / Video Input ]
             │
             ▼
  [ Haar Cascade / OpenCV ] ───► Face Detection & Bounding Box Extraction
             │
             ▼
  [ Preprocessing Pipeline ] ──► Grayscale Conversion ➔ Resizing (48x48) ➔ Normalization
             │
             ▼
  [ Deep CNN (Keras / TF) ] ───► Feature Extraction & Softmax Classification
             │
             ▼
  [ Real-Time UI Overlay ] ────► Bounding Box, Emotion Label & Confidence Score


📂 Repository Structure

Emotion_AI_Bot_Project/
├── dataset/                   
│   ├── train/
│   └── test/
├── .gitignore                    
├── emotion_bot.py               
├── emotion_model.h5             
├── train_emotion_model.py       
├── README.md                    
└── LICENSE                      "# Emotion_AI_Bot_8848" 
