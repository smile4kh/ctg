import os
import traceback
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the model
MODEL_PATH = "fetal_health_rf_model_balanced.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# Define the expected feature names in the correct order
expected_features = [
    "baseline_value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations", "severe_decelerations",
    "prolongued_decelerations", "abnormal_short_term_variability",
    "histogram_min", "histogram_max", "histogram_mean", "histogram_median"
]

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        print("\n📥 Received Data:", data)

        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure the correct feature order
        if not all(feature in input_data.columns for feature in expected_features):
            return jsonify({"error": "Missing or incorrect features"}), 400

        input_data = input_data[expected_features]  # Reorder columns

        print("🧐 Processed Input Data:\n", input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        print("🔍 Model Prediction:", prediction)

        # Map prediction to diagnosis
        diagnosis_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        diagnosis = diagnosis_mapping.get(prediction, "Unknown")

        print("📢 Diagnosis:", diagnosis)

        return jsonify({"diagnosis": diagnosis, "prediction": int(prediction)})

    except Exception as e:
        error_message = traceback.format_exc()
        print("❌ Error during prediction:", error_message)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Run the Flask app locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
