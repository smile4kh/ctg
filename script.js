// ✅ Ensure JavaScript runs after the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    // ✅ Check if the analyze button exists before adding event listener
    const analyzeButton = document.getElementById("analyzeButton");
    if (!analyzeButton) {
        console.error("❌ Error: 'analyzeButton' not found in the DOM!");
        return;
    }

    // ✅ Define the API URL (Ensure this is correct)
    const API_URL = "https://ctg-3.onrender.com/predict"; 

    analyzeButton.addEventListener("click", function () {
        // ✅ Ensure result display elements exist
        const predictionResult = document.getElementById("predictionResult");
        if (!predictionResult) {
            console.error("❌ Error: 'predictionResult' not found in the DOM!");
            return;
        }

        // ✅ Input Data to send to API
        const inputData = {
            baseline_value: 120,
            accelerations: 0.003,
            fetal_movement: 0.4,
            uterine_contractions: 0.005,
            light_decelerations: 0.002,
            severe_decelerations: 0.0,
            prolonged_decelerations: 0.001,
            abnormal_short_term_variability: 0.5,
            histogram_min: 0,
            histogram_max: 15,
            histogram_mean: 2.5,
            histogram_median: 3
        };

        console.log("🔹 Sending data to API:", inputData);

        // ✅ Send data to the Flask API
        fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(inputData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`❌ HTTP Error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("🔹 Prediction result:", data);

            // ✅ Ensure the response contains a valid prediction
            if (data.prediction !== undefined) {
                const diagnosis = getDiagnosis(data.prediction);
                predictionResult.innerHTML = `<strong>CTG Interpretation:</strong> ${diagnosis}`;
            } else {
                predictionResult.innerHTML = `<strong>⚠️ Error:</strong> Invalid response from API`;
            }
        })
        .catch(error => {
            console.error("❌ Error sending data to API:", error);
            predictionResult.innerHTML = `<strong>API Error:</strong> Failed to connect.`;
        });
    });

    // ✅ Function to Map Prediction Values to Diagnosis
    function getDiagnosis(prediction) {
        switch (prediction) {
            case 1: return "🟢 Normal CTG";
            case 2: return "⚠️ Suspicious CTG (Needs Further Evaluation)";
            case 3: return "🔴 Pathological CTG (Immediate Attention Required)";
            default: return "❌ Unknown Diagnosis";
        }
    }
});
