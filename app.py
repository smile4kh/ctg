from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load the trained model
model_path = "fetal_health_rf_model_balanced.pkl"

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print("📌 Model trained with features:", model.feature_names_in_)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure feature names match the trained model
        expected_features = list(model.feature_names_in_)
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
