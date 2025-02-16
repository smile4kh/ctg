from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS correctly
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# ✅ Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load the trained model
model_path = "fetal_health_rf_model_balanced.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    print("❌ Error: Model file not found!")
    model = None  # Prevent execution errors

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # ✅ Convert data to DataFrame
        df = pd.DataFrame([data])

        # ✅ Ensure features match trained model
        expected_features = list(model.feature_names_in_)
        df = df.reindex(columns=expected_features, fill_value=0)

        # ✅ Debugging: Print received data for verification
        print("🔍 Received Data:\n", df)

        # ✅ Make prediction
        prediction = model.predict(df)[0]

        # ✅ Interpret the prediction
        diagnosis = {
            1: "Normal",
            2: "Suspicious",
            3: "Pathological"
        }.get(int(prediction), "Unknown")

        # ✅ Return result as JSON
        return jsonify({
            "prediction": int(prediction),
            "diagnosis": diagnosis
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App (For Local Testing)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

