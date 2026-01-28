import cv2
import numpy as np
import joblib

# Load the trained model we just created
# Ensure 'soil_model.pkl' is in the same folder
try:
    model = joblib.load("soil_model.pkl")
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("⚠️ Warning: soil_model.pkl not found. Run train_soil_model.py first.")

def extract_features(image_path):
    # MUST MATCH the training script exactly
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    cv2.normalize(hist_hue, hist_hue)
    cv2.normalize(hist_sat, hist_sat)
    cv2.normalize(hist_val, hist_val)
    return np.concatenate([hist_hue, hist_sat, hist_val]).flatten()

def predict_soil_type(image_path):
    if not MODEL_LOADED:
        return "Model Error"
    
    # 1. Read image and get numbers
    features = extract_features(image_path)
    
    if features is None:
        return "Invalid Image"
        
    # 2. Reshape for the model (1 sample)
    features = features.reshape(1, -1)
    
    # 3. Ask the model to predict
    prediction = model.predict(features)
    
    # 4. Return the result (e.g., 'Black Soil')
    return prediction[0]