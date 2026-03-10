Deepfake Image Detection System

Download the trained model from:
https://drive.google.com/file/d/1SJ3f1pGrsMho7tb8xNwct42O2iAqeD9u/view?usp=sharing


This project detects whether an image is REAL or FAKE using a trained deep learning model (ResNet50).

Project Components:
- Flask Web Application
- Trained Deepfake Detection Model
- Image Upload Interface
- Real/Fake Prediction with Confidence Score

Requirements:
Python 3.x

Install required libraries:
pip install flask tensorflow opencv-python numpy matplotlib seaborn scikit-learn

How to Run the Project:

1. Open terminal in the project folder.

2. Run the Flask application:
python app.py

3. Open the browser and go to:
http://127.0.0.1:5000

4. Upload an image to check if it is REAL or FAKE.

Notes:
- The dataset used for training is not included due to large size.
- The trained model file "deepfake_model_final.h5" is included.
