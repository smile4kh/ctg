from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load("fetal_health_rf_model_balanced.pkl")
    print("✅ Model loaded successfully!")
    print("Model trained with features:", model.feature_names_in_)
except Exception as e:
    print("❌ Error loading model:", str(e))

# Define the expected feature order based on training data
expected_features = [
    "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "histogram_min", "histogram_max"
]

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON request data
        data = request.get_json()
        print("\n📥 Received Data:", data)

        # Convert input data into DataFrame
        input_data = pd.DataFrame([data])

        # Check if all required features are present
        if not all(feature in input_data.columns for feature in expected_features):
            missing_features = [f for f in expected_features if f not in input_data.columns]
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Ensure input data is ordered correctly
        input_data = input_data[expected_features]

        # Print processed input before prediction
        print("🧐 Processed Input Data:\n", input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        print("🔍 Model Prediction:", prediction)

        # Convert numerical prediction to class label
        diagnosis_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        diagnosis = diagnosis_mapping.get(prediction, "Unknown")

        # Print final output before sending response
        print("📢 Diagnosis:", diagnosis)

        return jsonify({"diagnosis": diagnosis, "prediction": int(prediction)})

    except Exception as e:
        print("❌ Error during prediction:", traceback.format_exc())
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
