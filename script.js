document.addEventListener("DOMContentLoaded", async function () {
    console.log("✅ script.js is loaded and running!");

    let analyzeBtn = document.getElementById("analyzeBtn");
    let fileInput = document.getElementById("ctgUpload");
    let canvas = document.getElementById("ctgCanvas");
    let ctx = canvas.getContext("2d");

    if (!analyzeBtn) {
        console.error("❌ Analyze button not found!");
        return;
    }

    // ✅ Set TensorFlow.js Backend
    await tf.setBackend('webgl');
    await tf.ready();
    console.log("✅ TensorFlow.js WebGL backend activated!");

    analyzeBtn.addEventListener("click", async function () {
        console.log("📸 Analyze button clicked!");

        if (fileInput.files.length === 0) {
            alert("⚠️ Please upload a CTG image first!");
            return;
        }

        let file = fileInput.files[0];
        let img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = async function () {
            console.log("✅ Image loaded!");
            canvas.width = img.width / 2;
            canvas.height = img.height / 2;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);

            console.log("🛠 Converting image to tensor...");
            let tensor = tf.browser.fromPixels(canvas).toFloat().div(255);
            console.log("✅ Tensor created:", tensor.shape);

            console.log("🎚 Applying Sobel Edge Detection...");
            let edgeTensor = await applyCustomSobelFilter(tensor);
            console.log("✅ Edge Detection Applied!");

            console.log("📊 Extracting CTG Features...");
            let interpretation = interpretCTG(edgeTensor);
            console.log("✅ Interpretation Complete!");

            console.log("🖼 Displaying processed image...");
            edgeTensor = normalizeTensor(edgeTensor);
            await tf.browser.toPixels(edgeTensor, canvas);
            console.log("✅ Processing complete!");

            document.getElementById("analysisResult").innerHTML = `<strong>CTG Interpretation:</strong> ${interpretation}`;

            // ✅ Send extracted features to Flask API
            let apiUrl = "https://ctg-3.onrender.com/predict";  // Render API URL
            let requestData = {
                "baseline_value": 120,
                "accelerations": 0.003,
                "fetal_movement": 0.4,
                "uterine_contractions": 0.005,
                "light_decelerations": 0.002,
                "severe_decelerations": 0.0,
                "prolongued_decelerations": 0.001,
                "abnormal_short_term_variability": 0.5,
                "histogram_min": 0,
                "histogram_max": 15,
                "histogram_mean": 2.5,
                "histogram_median": 3
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
