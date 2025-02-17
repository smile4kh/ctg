import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import joblib

# Load the trained machine learning model and scaler
rf_model = joblib.load("rf_ctg_model.pkl")
scaler = joblib.load("scaler.pkl")

def process_ctg_image(image_path):
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny Edge Detection to detect waveform
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours (detect the CTG waveform)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and take the largest one (assuming it's the CTG trace)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Extract Y-values of waveform to estimate heart rate
    heart_rate_values = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            heart_rate_values.append(y)
    
    if not heart_rate_values:
        print("Error: No valid waveform detected.")
        return None

    # Normalize Y-values to heart rate (50-200 bpm)
    min_y, max_y = min(heart_rate_values), max(heart_rate_values)
    bpm_values = [int((y - min_y) / (max_y - min_y) * 150 + 50) for y in heart_rate_values]
    bpm_values = sorted(bpm_values)

    # Detect peaks in waveform (to estimate variability and decelerations)
    peaks, _ = find_peaks(bpm_values, height=100)

    # Compute baseline, variability, and decelerations
    baseline = np.mean(bpm_values)
    variability = np.std(bpm_values)
    decelerations = len([bpm for bpm in bpm_values if bpm < 110])

    # Detect sinusoidal patterns using Fourier Transform
    fft_values = np.abs(np.fft.fft(bpm_values))
    frequencies = np.fft.fftfreq(len(bpm_values))
    dominant_freq = frequencies[np.argmax(fft_values)]
    is_sinusoidal = 0.05 < dominant_freq < 0.2  # Detect repeating sinusoidal waves

    # ✅ Save waveform plot instead of using plt.show()
    plt.plot(bpm_values)
    plt.title("Extracted Fetal Heart Rate")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.savefig("static/uploads/ctg_plot.png")  # Save the plot
    plt.close()  # Prevents GUI error

    return {
        "Baseline": baseline,
        "Variability": variability,
        "Decelerations": decelerations,
        "Is_Sinusoidal": is_sinusoidal
    }

def classify_ctg(features):
    if features is None:
        return "Error in feature extraction"

    # Classify using RCOG/NICE criteria
    baseline_category = "Reassuring" if 110 <= features["Baseline"] <= 160 else "Abnormal"
    variability_category = "Reassuring" if features["Variability"] >= 5 else "Abnormal"
    deceleration_category = "Reassuring" if features["Decelerations"] == 0 else "Abnormal"

    if features["Is_Sinusoidal"]:
        return "Pathological (Sinusoidal Pattern Detected)"

    categories = [baseline_category, variability_category, deceleration_category]
    if categories.count("Reassuring") == 3:
        return "Normal (RCOG/NICE)"
    elif "Abnormal" in categories:
        return "Pathological (RCOG/NICE)"
    
    feature_values = np.array([features["Baseline"], features["Variability"], features["Decelerations"]]).reshape(1, -1)
    ml_prediction = rf_model.predict(feature_values)[0]
    ml_classes = {0: "Normal", 1: "Suspicious", 2: "Pathological"}

    return f"{ml_classes[ml_prediction]} (ML Prediction)"
