from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Define Model Path
MODEL_PATH = "fetal_health_rf_model_balanced.pkl"

# ✅ Load the Model Safely
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    expected_features = list(model.feature_names_in_)  # Ensure model is defined before using it
    print("🔹 Expected feature order:", expected_features)

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevent Flask from crashing if the model isn't loaded

# ✅ Default Route (To Check If API is Running)
@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🚨 Check if model is loaded before making predictions
        if model is None:
            return jsonify({"error": "Model is not loaded"}), 500

        # ✅ Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        df = pd.DataFrame([data])

        # ✅ Ensure feature names match the model's expected features
        if set(df.columns) != set(expected_features):
            return jsonify({"error": "Feature names do not match model's expected features."}), 400

        # ✅ Reorder features to match the trained model
        df = df[expected_features]

        # ✅ Make Prediction
        prediction = model.predict(df)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start Flask Application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

