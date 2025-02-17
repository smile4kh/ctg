document.getElementById("analyzeBtn").addEventListener("click", function() {
    let fileInput = document.getElementById("ctgUpload");
    let analysisResult = document.getElementById("analysisResult");

    let file = fileInput.files[0];
    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => {
        analysisResult.textContent = data.result;
        document.getElementById("ctgPlot").src = data.plot_url;
    });
});
