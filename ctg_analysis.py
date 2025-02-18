import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import joblib

# Load the trained machine learning model and scaler
rf_model = joblib.load("rf_ctg_model.pkl")
scaler = joblib.load("scaler.pkl")

def process_ctg_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Error: Unable to load image.")

        # Apply Gaussian Blur to remove noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Use Canny Edge Detection to detect waveform
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours (detect the CTG waveform)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract Y-values of waveform to estimate heart rate
        heart_rate_values = [point[0][1] for contour in contours for point in contour]
        if not heart_rate_values:
            raise ValueError("Error: No valid waveform detected.")

        # Normalize Y-values to heart rate (50-200 bpm)
        min_y, max_y = min(heart_rate_values), max(heart_rate_values)
        bpm_values = [int((y - min_y) / (max_y - min_y) * 150 + 50) for y in heart_rate_values]
        bpm_values = sorted(bpm_values)

        # Compute baseline, variability, and decelerations
        baseline = np.mean(bpm_values)
        variability = np.std(bpm_values)
        decelerations = len([bpm for bpm in bpm_values if bpm < 110])

        # ✅ Save waveform plot instead of using plt.show()
        plot_path = "static/uploads/ctg_plot.png"
        plt.figure(figsize=(6, 3))
        plt.plot(bpm_values)
        plt.title("Extracted Fetal Heart Rate")
        plt.xlabel("Time")
        plt.ylabel("Heart Rate (bpm)")
        plt.savefig(plot_path)  # Save the plot
        plt.close()  # Prevents GUI error

        return {
            "Baseline": baseline,
            "Variability": variability,
            "Decelerations": decelerations,
            "plot_url": plot_path  # Return plot path for frontend
        }

    except Exception as e:
        print(f"❌ CTG PROCESSING ERROR: {str(e)}")
        return None  # Prevents breaking Flask

