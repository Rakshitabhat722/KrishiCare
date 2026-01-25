import cv2
import numpy as np

def predict_soil_type(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return "Unknown Soil"

    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    blue, green, red = avg_color

    if red > green and red > blue:
        return "Red Soil"
    elif green > red and green > blue:
        return "Black Soil"
    else:
        return "Loamy Soil"
