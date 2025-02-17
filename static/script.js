document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("ctgUpload");
    const analyzeButton = document.getElementById("analyzeBtn");
    const analysisResult = document.getElementById("analysisResult");

    analyzeButton.addEventListener("click", function () {
        if (!fileInput.files.length) {
            analysisResult.innerHTML = "<p style='color:red;'>Please select an image file first.</p>";
            return;
        }

        let file = fileInput.files[0];

        // Ensure only images are uploaded
        if (!file.type.startsWith("image/")) {
            analysisResult.innerHTML = "<p style='color:red;'>Only image files (JPG, PNG) are allowed.</p>";
            return;
        }

        // Prepare FormData
        let formData = new FormData();
        formData.append("file", file);

        // Show loading message
        analysisResult.innerHTML = "<p style='color:blue;'>Uploading and analyzing... Please wait.</p>";

        // Send to Flask Backend
        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(result => {
            analysisResult.innerHTML = `<p style='color:green;'><strong>Diagnosis:</strong> ${result}</p>`;
        })
        .catch(error => {
            analysisResult.innerHTML = "<p style='color:red;'>Error uploading file. Please try again.</p>";
        });
    });
});
