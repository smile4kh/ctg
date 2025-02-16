from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load("fetal_health_rf_model_balanced.pkl")

# Ensure feature order matches model training
expected_features = model.feature_names_in_  # Get correct feature order

@app.route('/')
def home():
    return "Fetal Health Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all required features exist (set missing ones to 0)
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Fill missing features with 0

        # Reorder columns to match model training
        input_df = input_df[expected_features]

        # Make a prediction
        prediction = model.predict(input_df)[0]

        # Map numeric output to readable labels
        diagnosis_map = {1: "Normal", 2: "Suspicious", 3: "Pathological"}
        diagnosis = diagnosis_map.get(prediction, "Unknown")

        # Return response
        return jsonify({
            "diagnosis": diagnosis,
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
