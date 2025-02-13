document.addEventListener("DOMContentLoaded", async function () {
    console.log("Script.js is loaded!");

    let analyzeBtn = document.getElementById("analyzeBtn");
    let fileInput = document.getElementById("ctgUpload");
    let canvas = document.getElementById("ctgCanvas");
    let ctx = canvas.getContext("2d");

    if (!analyzeBtn) {
        console.error("Button not found!");
        return;
    }

    // ✅ Set WebGL Backend
    await tf.setBackend('webgl');
    await tf.ready();
    console.log("TensorFlow.js WebGL backend activated!");

    analyzeBtn.addEventListener("click", async function () {
        console.log("Analyze button clicked!");

        if (fileInput.files.length === 0) {
            alert("Please upload a CTG image first!");
            return;
        }

        let file = fileInput.files[0];
        let img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = async function () {
            console.log("Image loaded!");
            canvas.width = img.width / 2;
            canvas.height = img.height / 2;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);

            console.log("Converting image to tensor...");
            let tensor = tf.browser.fromPixels(canvas).toFloat().div(255);
            console.log("Tensor created:", tensor.shape);

            console.log("Applying Sobel Edge Detection...");
            let edgeTensor = await applyCustomSobelFilter(tensor);
            console.log("Edge Detection Applied!");

            console.log("Extracting CTG Features...");
            let interpretation = interpretCTG(edgeTensor);
            console.log("Interpretation Complete!");

            console.log("Displaying processed image...");
            edgeTensor = normalizeTensor(edgeTensor);
            await tf.browser.toPixels(edgeTensor, canvas);
            console.log("Processing complete!");

            document.getElementById("analysisResult").innerHTML = `<strong>CTG Interpretation:</strong> ${interpretation}`;
        };
    });

    // ✅ Sobel Edge Detection
    async function applyCustomSobelFilter(imageTensor) {
        await tf.ready();
        
        console.log("Applying custom Sobel filter...");

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

    // ✅ Improved CTG Interpretation (More Adaptive)
    function interpretCTG(edgeTensor) {
        const data = edgeTensor.arraySync();
        let pixelSum = 0;
        let pixelCount = 0;
        let variabilitySum = 0;
        let maxPixel = 0;
        let minPixel = 1;

        let decelerationCount = 0;
        let sustainedLowVariability = 0;
        let lastValue = 0;
        let lowVarSegments = 0;
        let segmentLength = 30; // Adjusted window for better detection
        let windowSum = 0;

        // ✅ Analyze pixel intensity across the image
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < data[i].length; j++) {
                let pixelValue = data[i][j][0];  // Use first channel (grayscale)
                pixelSum += pixelValue;
                pixelCount++;

                if (pixelValue > maxPixel) maxPixel = pixelValue;
                if (pixelValue < minPixel) minPixel = pixelValue;

                // ✅ Calculate variability using a moving average window
                if (i > 0) {
                    let diff = Math.abs(pixelValue - data[i - 1][j][0]);
                    variabilitySum += diff;

                    // ✅ Detect decelerations (Sudden Drops)
                    if (diff > 0.1 && pixelValue < lastValue - 0.08) {
                        decelerationCount++;
                    }

                    // ✅ Detect sustained low variability (Flat Tracing)
                    windowSum += diff;
                    if (i % segmentLength === 0) {
                        if (windowSum / segmentLength < 0.012) {
                            lowVarSegments++;
                        }
                        windowSum = 0;
                    }
                }
                lastValue = pixelValue;
            }
        }

        let avgPixel = pixelSum / pixelCount;
        let variability = variabilitySum / pixelCount;
        let range = maxPixel - minPixel;

        console.log(`Avg Pixel: ${avgPixel}, Variability: ${variability}, Range: ${range}, Decelerations: ${decelerationCount}, Low Variability Segments: ${lowVarSegments}`);

        // ✅ **New Interpretation Rules** (More Balanced)
        if (avgPixel > 0.5 && variability > 0.04 && range > 0.25 && decelerationCount < 3 && lowVarSegments < 2) {
            return "Normal CTG (Healthy FHR & Variability)";
        } else if (lowVarSegments > 2 && lowVarSegments < 5) {
            return "Suspicious CTG (Mild Reduced Variability)";
        } else if (decelerationCount > 5 || lowVarSegments > 5) {
            return "Pathological CTG (Late Decelerations & Severe Low Variability)";
        } else {
            return "Suspicious CTG (Needs Further Evaluation)";
        }
    }
});
