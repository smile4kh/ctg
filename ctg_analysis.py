import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
rf_model = joblib.load("rf_ctg_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
ctg_data = pd.read_csv("Cardiotocographic.csv")

# Define feature columns
X = ctg_data.drop(columns=['NSP'])
y = ctg_data['NSP'] - 1  # Adjusting target labels

# Function to classify CTG using RCOG/NICE + ML
def classify_ctg_combined(features):
    baseline_category = "Reassuring" if 110 <= features["Baseline"] <= 160 else                         "Non-reassuring" if 100 <= features["Baseline"] <= 109 or 161 <= features["Baseline"] <= 180 else "Abnormal"
    
    variability_category = "Reassuring" if features["Variability"] >= 5 else                            "Non-reassuring" if 5 > features["Variability"] >= 40 else "Abnormal"

    deceleration_category = "Reassuring" if features["Decelerations"] == 0 else                             "Non-reassuring" if 1 <= features["Decelerations"] <= 2 else "Abnormal"

    categories = [baseline_category, variability_category, deceleration_category]
    
    if categories.count("Reassuring") == 3:
        return "Normal (RCOG/NICE)"
    elif categories.count("Non-reassuring") == 1 and categories.count("Reassuring") == 2:
        return "Suspicious (RCOG/NICE)"
    elif categories.count("Abnormal") >= 1 or categories.count("Non-reassuring") >= 2:
        return "Pathological (RCOG/NICE)"

    feature_values = np.array([features[key] for key in X.columns]).reshape(1, -1)
    ml_prediction = rf_model.predict(feature_values)[0]

    ml_classes = {0: "Normal", 1: "Suspicious", 2: "Pathological"}
    return f"{ml_classes[ml_prediction]} (ML Prediction)"

# Example Test Case
new_ctg_trace = {
    "Baseline": 145,
    "Variability": 8,
    "Decelerations": 1,
    "AC": 0.004, "FM": 0, "UC": 0.008, "DL": 0.002, "DS": 0, "DP": 0,
    "ASTV": 18, "MSTV": 2.0, "ALTV": 5, "Min": 60, "Max": 170, "Nmax": 5,
    "Nzeros": 0, "Mode": 140, "Mean": 135, "Median": 136, "Variance": 12, "Tendency": 0
}

# Run classification
classification_result = classify_ctg_combined(new_ctg_trace)
print("CTG Classification:", classification_result)
