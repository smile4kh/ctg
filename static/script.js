document.getElementById("analyzeBtn").addEventListener("click", function() {
    let fileInput = document.getElementById("ctgUpload");
    let analysisResult = document.getElementById("analysisResult");

    if (fileInput.files.length === 0) {
        analysisResult.textContent = "Please upload an image first.";
        return;
    }

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            analysisResult.textContent = "Error: " + data.error;
        } else {
            analysisResult.textContent = data.result;
        }
    })
    .catch(error => {
        analysisResult.textContent = "Error processing the image.";
    });
});
