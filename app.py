from flask import Flask, request, render_template
import os
from ctg_analysis import process_ctg_image, classify_ctg

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure uploads folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Process the CTG image
        features = process_ctg_image(filepath)
        diagnosis = classify_ctg(features)

        return f"<h3>Diagnosis: {diagnosis}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
