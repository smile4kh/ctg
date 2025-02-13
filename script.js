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

            console.log("Applying Custom Sobel Edge Detection...");
            let edgeTensor = await applyCustomSobelFilter(tensor);
            console.log("Edge Detection Applied!");

            console.log("Normalizing Edge Tensor for Display...");
            edgeTensor = normalizeTensor(edgeTensor); // ✅ Fix: Normalize output

            console.log("Displaying processed image...");
            await tf.browser.toPixels(edgeTensor, canvas);
            console.log("Processing complete!");

            document.getElementById("analysisResult").innerText = "CTG AI Processing Completed!";
        };
    });

    // ✅ Custom Sobel Edge Detection
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

    // ✅ Fix: Normalize tensor to [0,1] range
    function normalizeTensor(tensor) {
        const minVal = tensor.min();
        const maxVal = tensor.max();
        return tensor.sub(minVal).div(maxVal.sub(minVal)); // Scale to [0,1]
    }
});
