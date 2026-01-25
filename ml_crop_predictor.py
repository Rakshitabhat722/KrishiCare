import joblib
import numpy as np
import os

# Load model safely using absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_crop(features):
    """
    features = [N, P, K, temperature, humidity, ph, rainfall]
    """
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]
