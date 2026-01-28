from flask import Flask, render_template, request
import os
from ml_crop_predictor import predict_crop
from soil_model import predict_soil_type

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get uploaded image
    image = request.files.get("soil_image")
    image_path = None
    
    if image:
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

    # 2. Get location
    location = request.form.get("location")

    # 3. Soil prediction
    soil_type = predict_soil_type(image_path) if image_path else "Unknown"

    # 4. Dummy environmental values
    N, P, K = 90, 42, 43
    temperature = 25
    humidity = 80
    ph = 6.5
    rainfall = 200

    # 5. Crop prediction
    crop = predict_crop([N, P, K, temperature, humidity, ph, rainfall])

    # 6. RETURN THE NEW RESULT PAGE
    return render_template(
        "result.html",    # <--- This is the key change!
        soil_type=soil_type,
        crop=crop,
        location=location
    )

if __name__ == "__main__":
    app.run(debug=True)