# # app.py
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and label encoder
# model = joblib.load("crop_recommendation_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")
# scaler = joblib.load("scaler.pkl")

# @app.route('/')
# def home():
#     return "🌾 Crop Prediction API is running!"

# @app.route('/predict', methods=['POST'])
# def predict_crop():
#     try:
#         data = request.get_json()

#         # Extract numeric features from JSON
#         input_features = [
#             data['N'],
#             data['P'],
#             data['K'],
#             data['temperature'],
#             data['humidity'],
#             data['ph'],
#             data['rainfall']
#         ]

#         # Convert and scale
#         input_array = np.array(input_features).reshape(1, -1)
#         input_scaled = scaler.transform(input_array)

#         # Predict and decode
#         prediction_encoded = model.predict(input_scaled)[0]
#         crop_name = label_encoder.inverse_transform([prediction_encoded])[0]

#         return jsonify({'predicted_crop': crop_name})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# if __name__ == '__main__':
#     app.run(debug=True)


# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and label encoder
try:
    model = joblib.load("crop_recommendation_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    model = None
    label_encoder = None
    scaler = None

@app.route('/')
def home():
    return jsonify({
        "message": "🌾 Crop Prediction API is running!",
        "status": "healthy",
        "models_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        # Check if models are loaded
        if not all([model, label_encoder, scaler]):
            return jsonify({'error': 'Models not loaded properly'}), 500

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract numeric features from JSON
        input_features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]

        # Convert and scale
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict and decode
        prediction_encoded = model.predict(input_scaled)[0]
        
        # Get prediction probabilities for confidence
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(prediction_proba) * 100)
        
        crop_name = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get top 3 predictions for alternatives
        top_indices = np.argsort(prediction_proba)[-3:][::-1]
        alternatives = [label_encoder.inverse_transform([idx])[0] for idx in top_indices[1:]]

        return jsonify({
            'predicted_crop': crop_name,
            'confidence': round(confidence, 1),
            'alternatives': alternatives,
            'input_data': data
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)