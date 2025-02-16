from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Explicitly Allow All Origins (Fix CORS Issues)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model_path = "fetal_health_rf_model_balanced.pkl"

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        df = pd.DataFrame([data])
        expected_features = list(model.feature_names_in_)
        df = df.reindex(columns=expected_features, fill_value=0)

        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
