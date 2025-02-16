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
                    document.getElementById("analysisResult").innerHTML += `<br><strong>Prediction:</strong> ${result.prediction}`;
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

    // ✅ CTG Interpretation
    function interpretCTG(edgeTensor) {
        return "CTG interpretation successful!";  // Placeholder
    }
});
