document.addEventListener("DOMContentLoaded", function () {
    console.log("Script.js is loaded!");

    let analyzeBtn = document.getElementById("analyzeBtn");
    let fileInput = document.getElementById("ctgUpload");
    let canvas = document.getElementById("ctgCanvas");
    let ctx = canvas.getContext("2d");

    if (!analyzeBtn) {
        console.error("Button not found!");
        return;
    }

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
            canvas.width = img.width / 2; 
            canvas.height = img.height / 2;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);

            // Convert Image to Tensor
            let tensor = tf.browser.fromPixels(canvas).toFloat().div(255);
            
            // Apply Edge Detection (Sobel Filter)
            let edgeTensor = applySobelFilter(tensor);

            // Display Processed Image
            await tf.browser.toPixels(edgeTensor, canvas);

            document.getElementById("analysisResult").innerText = "CTG AI Processing Completed!";
        };
    });

    // Function to Apply Sobel Edge Detection
    function applySobelFilter(imageTensor) {
        let sobelX = tf.image.sobelEdges(imageTensor).slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        let edgeTensor = sobelX.abs().max(2).expandDims(-1);
        return edgeTensor;
    }
});
