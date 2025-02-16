from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Load the trained model
model = joblib.load("fetal_health_rf_model_balanced.pkl")

@app.route('/')
def home():
    return "Fetal Health Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Ensure feature names match the model's training data
        input_data = {
            "baseline value": data.get("baseline_value", 0),
            "accelerations": data.get("accelerations", 0),
            "fetal_movement": data.get("fetal_movement", 0),
            "uterine_contractions": data.get("uterine_contractions", 0),
            "light_decelerations": data.get("light_decelerations", 0),
            "severe_decelerations": data.get("severe_decelerations", 0),
            "prolongued_decelerations": data.get("prolongued_decelerations", 0),
            "abnormal_short_term_variability": data.get("abnormal_short_term_variability", 0),
            "histogram_min": data.get("histogram_min", 0),
            "histogram_max": data.get("histogram_max", 0),
            "histogram_mean": data.get("histogram_mean", 0),
            "histogram_median": data.get("histogram_median", 0)
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure column order matches the model training
        expected_features = model.feature_names_in_
        input_df = input_df[expected_features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Map prediction to readable diagnosis
        diagnosis_map = {1: "Normal", 2: "Suspicious", 3: "Pathological"}
        diagnosis = diagnosis_map.get(prediction, "Unknown")
        
        return jsonify({"diagnosis": diagnosis, "prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
