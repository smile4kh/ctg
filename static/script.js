document.getElementById("analyzeBtn").addEventListener("click", function() {
    let fileInput = document.getElementById("ctgUpload");
    let analysisResult = document.getElementById("analysisResult");
    let plotImage = document.getElementById("ctgPlot");

    if (fileInput.files.length === 0) {
        analysisResult.textContent = "Please upload an image first.";
        return;
    }

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            analysisResult.textContent = "Error: " + data.error;
        } else {
            analysisResult.textContent = data.result;
            plotImage.src = data.plot_url;  // ✅ Update image src
            plotImage.style.display = "block";  // ✅ Ensure it's visible
        }
    })
    .catch(error => {
        analysisResult.textContent = "Error processing the image.";
    });
});
