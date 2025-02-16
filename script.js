function getDiagnosis(prediction) {
    if (prediction === 1) {
        return "🟢 Normal";
    } else if (prediction === 2) {
        return "⚠️ Suspicious (Needs Further Evaluation)";
    } else if (prediction === 3) {
        return "🔴 Pathological (Requires Immediate Attention)";
    } else {
        return "Unknown Diagnosis";
    }
}

fetch(API_URL, requestOptions)
    .then(response => response.json())
    .then(data => {
        let prediction = data.prediction;
        let diagnosis = getDiagnosis(prediction);

        document.getElementById("result").innerHTML = `
            <strong>CTG Interpretation:</strong> ${diagnosis} <br>
            <strong>Prediction:</strong> ${prediction}
        `;
    })
    .catch(error => {
        console.error("Error sending data to API:", error);
        document.getElementById("result").innerHTML = "<strong>API Error:</strong> Failed to connect.";
    });
