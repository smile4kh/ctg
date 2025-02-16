from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model
MODEL_PATH = "fetal_health_rf_model_balanced.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Print expected feature names for debugging
try:
    expected_features = list(model.feature_names_in_)
    print("🔹 Expected feature order:", expected_features)
except AttributeError:
    print("⚠️ Model does not store feature names. Check training data.")
    expected_features = None  # Prevent crashes if feature names aren't available

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Fetal Health Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Check if model has expected features
        if expected_features is None:
            return jsonify({"error": "Model does not store feature names."}), 500

        # Ensure correct feature order
        if set(df.columns) != set(expected_features):
            return jsonify({
                "error": "Feature names do not match. Ensure all required features are included."
            }), 400

        # Reorder features to match training data
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

