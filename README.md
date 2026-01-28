# 🌱 KrishiCare – AI-Based Crop Recommendation System

KrishiCare is a Flask-based web application that helps farmers and agricultural users
identify suitable crops based on soil image analysis and machine learning.

The system combines basic soil image processing with a trained ML model
to recommend crops using real agricultural datasets.

## 🚀 Features

- Upload soil image
- Soil type detection using OpenCV
- Crop recommendation using Machine Learning
- Trained on Kaggle Crop Recommendation dataset
- Clean and simple web interface
- Flask backend with modular architecture

## 🧠 How It Works

1. User uploads a soil image through the web interface
2. The image is processed to identify soil characteristics
3. A trained Machine Learning model predicts the most suitable crop
4. The result is displayed on the same page

## 🛠️ Tech Stack

- Python
- Flask
- OpenCV
- Scikit-learn
- Pandas
- NumPy
- HTML & CSS

---

## 📊 Dataset Used

- Crop Recommendation Dataset from Kaggle  
  (Contains soil nutrients, climate conditions, and crop labels)


## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/KrishiCare.git

# Go to project folder
cd KrishiCare

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the ML model
python train_model.py

# Run the Flask app
python app.py
