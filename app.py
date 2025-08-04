# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("crop_recommendation_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "🌾 Crop Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()

        # Extract numeric features from JSON
        input_features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]

        # Convert and scale
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict and decode
        prediction_encoded = model.predict(input_scaled)[0]
        crop_name = label_encoder.inverse_transform([prediction_encoded])[0]

        return jsonify({'predicted_crop': crop_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
