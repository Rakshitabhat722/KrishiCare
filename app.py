from flask import Flask, render_template, request
import os
from soil_model import predict_soil_type
from ml_crop_predictor import predict_crop

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # 1. Get uploaded image
    if "soilImage" not in request.files:
        return "No image uploaded"

    file = request.files["soilImage"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_soil.jpg")
    file.save(image_path)

    # 2. Predict soil type from image
    soil_type = predict_soil_type(image_path)

    # 3. (Temporary) values from dataset / assumption
    # In real systems these come from sensors / soil test
    N = 90
    P = 42
    K = 43
    temperature = 25
    humidity = 80
    ph = 6.5
    rainfall = 200

    # 4. ML-based crop prediction
    crop = predict_crop([N, P, K, temperature, humidity, ph, rainfall])

    # 5. Send results back to UI
    return render_template(
        "index.html",
        soil_type=soil_type,
        crops=[crop]
    )


if __name__ == "__main__":
    app.run(debug=True)
