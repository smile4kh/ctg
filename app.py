import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Configure Flask
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

        # Process the image
        result = process_ctg_image(filepath)

        return jsonify({"message": "File uploaded successfully", "result": result, "filepath": filepath})

    return jsonify({"error": "Invalid file type"}), 400

# Function to process the CTG image
def process_ctg_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Unable to load image"

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection (example preprocessing)
        edges = cv2.Canny(gray, 50, 150)

        # Here, you can apply AI/ML model for analysis (Example: Using TensorFlow)
        # For now, we just return a dummy message
        return "CTG image processed successfully"

    except Exception as e:
        return f"Error in processing: {str(e)}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
