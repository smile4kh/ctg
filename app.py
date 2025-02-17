import os
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import ctg_analysis  # ✅ Import ctg_analysis.py for real processing

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
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ✅ Use ctg_analysis to process the image
        features = ctg_analysis.process_ctg_image(filepath)
        if features is None:
            return jsonify({"error": "Failed to process image"}), 500

        # ✅ Classify CTG result using NICE/ML model
        diagnosis = ctg_analysis.classify_ctg(features)

        return jsonify({
            "message": "File uploaded successfully",
            "result": diagnosis,
            "features": features,
            "filepath": filepath
        })

    return jsonify({"error": "Invalid file type"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
