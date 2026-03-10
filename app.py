from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("deepfake_model_final.h5")

IMG_SIZE = 224

# Create upload folder automatically
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Image preprocessing
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            image_path = filepath

            img = preprocess_image(filepath)

            prob = model.predict(img)[0][0]

            if prob > 0.5:
                prediction = "FAKE IMAGE"
                confidence = round(prob * 100, 2)
            else:
                prediction = "REAL IMAGE"
                confidence = round((1 - prob) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)