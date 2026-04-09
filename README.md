# 🌿 AgriDrone AI — Precision Agriculture Intelligence System
**Dr. D. Y. Patil Institute of Technology, Pimpri, Pune**

AgriDrone AI is a deep learning-based system designed to identify crop diseases and provide real-time treatment advice. Using a MobileNetV2 architecture, it classifies vegetable and fruit leaf images with high accuracy.

---

## 🎯 KEY FEATURES
- **Disease Detection:** Identify 38 different conditions across crops like Apple, Tomato, Corn, and Potato.
- **Treatment Advisor:** Instant organic and chemical recommendations for detected diseases.
- **Precision Metrics:** Provides confidence scores and class probability visualizations.
- **Automated Reports:** Downloadable PDF health reports for farm records.

---

## 🛠️ TECH STACK
- **Frontend:** HTML5, CSS3 (Tailwind-style), JavaScript
- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Image Processing:** Pillow, NumPy

---

## 📦 SETUP INSTRUCTIONS

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Clone the repository and install the required dependencies:
pip install -r requirements.txt

###3. Dataset & Model
This project uses the Plant Disease Detection Dataset from Kaggle.
Download: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mgmitesh/plant-disease-detection-dataset)
Structure: Place the unzipped folders into a dataset/ directory.
Training: If you wish to retrain the model, run:
python train.py

###4. Running the App
Start the Flask server to launch the web dashboard:
python app.py
<img width="1908" height="979" alt="image" src="https://github.com/user-attachments/assets/e71fd5b2-0aa8-49fe-b025-b4bb00f68131" />

