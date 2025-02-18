import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import ctg_analysis  # ✅ Import the module
from ctg_analysis import classify_ctg  # ✅ Import the function directly

# Initialize Flask app
app = Flask(__name__)

# Define the folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle image upload and processing
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            print(f"✅ File successfully saved at: {filepath}")

            # ✅ Use `ctg_analysis.py` to extract CTG features
            features = ctg_analysis.process_ctg_image(filepath)
            if features is None:
                return jsonify({"error": "Failed to process image"}), 500

            # ✅ Classify CTG result using NICE/ML model
            diagnosis = classify_ctg(features)

            return jsonify({
                "message": "File uploaded successfully",
                "result": diagnosis,
                "features": features,
                "plot_url": features["plot_url"],  # ✅ Send plot path
                "filepath": filepath
            })

        return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")  # Log the error
        return jsonify({"error": f"Server error: {str(e)}"}), 500  # Ensure JSON response

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
