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

    analyzeBtn.addEventListener("click", function () {
        console.log("Analyze button clicked!");
        
        if (fileInput.files.length === 0) {
            alert("Please upload a CTG image first!");
            return;
        }

        let file = fileInput.files[0];
        let img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = function () {
            canvas.width = img.width / 2; 
            canvas.height = img.height / 2;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);
            document.getElementById("analysisResult").innerText = "Processing CTG...";
        };
    });
});
