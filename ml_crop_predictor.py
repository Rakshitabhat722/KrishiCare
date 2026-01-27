import joblib
import numpy as np

# Load your trained model
# Make sure 'crop_model.pkl' is in the same folder or adjust path
model = joblib.load('crop_model.pkl')

def predict_crop(data):
    """
    input: list of values [N, P, K, temp, humidity, ph, rainfall]
    output: List of top 4 recommended crops
    """
    # Reshape data for the model (2D array)
    data_array = np.array(data).reshape(1, -1)
    
    # 1. Get probabilities for all crops instead of just one prediction
    probabilities = model.predict_proba(data_array)[0]
    
    # 2. Get the class labels (the crop names)
    classes = model.classes_
    
    # 3. Sort them: create pairs of (probability, crop_name) and sort descending
    # This gives us the crops with the highest confidence scores
    sorted_indices = np.argsort(probabilities)[::-1]
    
    top_results = []
    
    # 4. Get the top 4 crops
    for i in range(4):
        index = sorted_indices[i]
        crop_name = classes[index]
        confidence = probabilities[index]
        
        # Only include if confidence is somewhat decent (optional)
        if confidence > 0.0: 
            top_results.append(crop_name)
            
    return top_results
