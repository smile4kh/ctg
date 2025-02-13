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

    // ✅ Set WebGL Backend (Fix for Sobel Edge Detection)
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
            let edgeTensor = await applySobelFilter(tensor);
            console.log("Edge Detection Applied!");

            console.log("Displaying processed image...");
            await tf.browser.toPixels(edgeTensor, canvas);
            console.log("Processing complete!");

            document.getElementById("analysisResult").innerText = "CTG AI Processing Completed!";
        };
    });

    async function applySobelFilter(imageTensor) {
        await tf.ready();
        
        // ✅ Ensure TensorFlow.js Image Processing is Loaded
        if (!tf.image.sobelEdges) {
            console.error("Sobel edge detection not found in TensorFlow.js!");
            return imageTensor;
        }

        let sobelEdges = tf.image.sobelEdges(imageTensor);
        let edgeTensor = sobelEdges.slice([0, 0, 0, 0], [-1, -1, -1, 1]).abs().max(2).expandDims(-1);

        return edgeTensor;
    }
});
