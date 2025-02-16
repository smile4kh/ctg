// ✅ Define the API URL (Update if needed)
const API_URL = "https://ctg-3.onrender.com/predict"; 

document.getElementById("analyzeButton").addEventListener("click", function() {
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

    console.log("Sending data to API:", inputData); // Debugging

    // Send data to the Flask API
    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(inputData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Prediction result:", data);

        // ✅ Ensure the response contains a valid prediction
        if (data.prediction !== undefined) {
            const diagnosis = getDiagnosis(data.prediction);
            document.getElementById("predictionResult").innerHTML = 
                `<strong>CTG Interpretation:</strong> ${diagnosis}`;
        } else {
            document.getElementById("predictionResult").innerHTML = 
                `<strong>Error:</strong> Invalid response from API`;
        }
    })
    .catch(error => {
        console.error("Error sending data to API:", error);
        document.getElementById("predictionResult").innerHTML = 
            `<strong>API Error:</strong> Failed to connect.`;
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
