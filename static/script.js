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
            let extractedFeatures = extractCTGFeatures(edgeTensor);
            console.log("✅ Features Extracted:", extractedFeatures);

            console.log("🖼 Displaying processed image...");
            edgeTensor = normalizeTensor(edgeTensor);
            await tf.browser.toPixels(edgeTensor, canvas);
            console.log("✅ Processing complete!");

            document.getElementById("analysisResult").innerHTML = `<strong>CTG Interpretation:</strong> Features extracted successfully!`;

            // ✅ Send extracted features to Flask API
            let apiUrl = "https://ctg-3.onrender.com/predict";  // Render API URL
            let requestData = extractedFeatures;  // Use extracted features

            console.log("📡 Sending data to API:", requestData);

            try {
                let response = await fetch(apiUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestData)
                });

                if (!response.ok) {
                    throw new Error(`❌ HTTP error! Status: ${response.status}`);
                }

                let result = await response.json();
                console.log("✅ API Response:", result);

                if (result.prediction !== undefined) {
                    let diagnosis = interpretPrediction(result.prediction);
                    document.getElementById("analysisResult").innerHTML += `<br><strong>Diagnosis:</strong> ${diagnosis}`;
                } else {
                    document.getElementById("analysisResult").innerHTML += `<br><strong>Error:</strong> ${result.error}`;
                }

            } catch (error) {
                console.error("❌ Error sending data to API:", error);
                document.getElementById("analysisResult").innerHTML += `<br><strong>API Error:</strong> Failed to connect.`;
            }
        };
    });

    // ✅ Sobel Edge Detection
    async function applyCustomSobelFilter(imageTensor) {
        await tf.ready();
        
        console.log("🔍 Applying custom Sobel filter...");

        const sobelX = tf.tensor2d([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], [3, 3]);

        const sobelY = tf.tensor2d([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], [3, 3]);

        let grayTensor = imageTensor.mean(2).expandDims(-1);
        const edgesX = tf.conv2d(grayTensor, sobelX.reshape([3, 3, 1, 1]), 1, "same");
        const edgesY = tf.conv2d(grayTensor, sobelY.reshape([3, 3, 1, 1]), 1, "same");

        let edgeTensor = tf.sqrt(tf.add(tf.square(edgesX), tf.square(edgesY)));

        sobelX.dispose();
        sobelY.dispose();
        edgesX.dispose();
        edgesY.dispose();
        grayTensor.dispose();

        return edgeTensor;
    }

    // ✅ Normalize tensor to [0,1] range
    function normalizeTensor(tensor) {
        const minVal = tensor.min();
        const maxVal = tensor.max();
        return tensor.sub(minVal).div(maxVal.sub(minVal));
    }

    // ✅ Extract features dynamically from tensor
    function extractCTGFeatures(tensor) {
        // Extract statistical features (min, max, mean, variance)
        let min = tensor.min().dataSync()[0];
        let max = tensor.max().dataSync()[0];
        let mean = tensor.mean().dataSync()[0];
        let variance = tensor.sub(mean).square().mean().dataSync()[0];

        return {
            "baseline_value": Math.round(mean * 150),  // Simulating baseline heart rate
            "accelerations": max * 0.005,  
            "fetal_movement": mean * 0.4,
            "uterine_contractions": variance * 0.005,
            "light_decelerations": min * 0.002,
            "severe_decelerations": 0.0,
            "prolongued_decelerations": variance * 0.001,
            "abnormal_short_term_variability": mean * 0.5,
            "histogram_min": Math.round(min * 10),
            "histogram_max": Math.round(max * 15),
            "histogram_mean": Math.round(mean * 5),
            "histogram_median": Math.round(variance * 3)
        };
    }

    // ✅ Interpret prediction results
    function interpretPrediction(prediction) {
        if (prediction === 1) {
            return "Normal CTG";
        } else if (prediction === 2) {
            return "Suspicious CTG (Needs Further Evaluation)";
        } else if (prediction === 3) {
            return "Pathological CTG (Requires Immediate Attention)";
        } else {
            return "Unknown Prediction";
        }
    }
});
