import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === CONFIGURATION ===
# This folder must contain subfolders like 'Black Soil', 'Yellow Soil', etc.
DATASET_PATH = "dataset/soil_images" 
MODEL_FILENAME = "soil_model.pkl"

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Resize and convert to HSV for color detection
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate Color Histograms
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    cv2.normalize(hist_hue, hist_hue)
    cv2.normalize(hist_sat, hist_sat)
    cv2.normalize(hist_val, hist_val)
    
    return np.concatenate([hist_hue, hist_sat, hist_val]).flatten()

def train():
    print(f"Checking for images in: {os.path.abspath(DATASET_PATH)}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: The folder '{DATASET_PATH}' does not exist.")
        print("Please create 'dataset/soil_images' and put your Kaggle folders inside.")
        return

    data = []
    labels = []
    
    # Loop through folders
    found_folders = False
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        
        if os.path.isdir(folder_path):
            found_folders = True
            print(f"   - Processing folder: {folder_name}...")
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                features = extract_features(img_path)
                
                if features is not None:
                    data.append(features)
                    labels.append(folder_name)

    if not found_folders:
        print("❌ No soil folders found inside 'dataset/soil_images'.")
        print("Make sure you unzipped the Kaggle dataset correctly.")
        return

    if len(data) == 0:
        print("❌ Found folders, but no images were readable.")
        return

    print(f"✅ Found {len(data)} images. Training model now...")
    
    X = np.array(data)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    joblib.dump(clf, MODEL_FILENAME)
    print(f"🎉 Success! Model saved as '{MODEL_FILENAME}'")

if __name__ == "__main__":
    train()