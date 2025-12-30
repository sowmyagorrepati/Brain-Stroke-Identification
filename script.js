const API_URL = "https://brain-stroke-identification.onrender.com/predict";

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const loading = document.getElementById("loading");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
  }
});

async function analyzeImage() {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please upload a CT scan image.");
    return;
  }

  loading.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    document.getElementById("prediction").innerText = data.prediction;
    document.getElementById("confidence").innerText = data.confidence;
    document.getElementById("confidenceFill").style.width =
      data.confidence + "%";

    // Grad-CAM decode
    const bytes = data.gradcam.match(/.{1,2}/g).map(b => parseInt(b, 16));
    const blob = new Blob([new Uint8Array(bytes)], { type: "image/png" });
    document.getElementById("gradcam").src = URL.createObjectURL(blob);

    // Explanation text
    let explanation = "";
    if (data.prediction === "Ischemia") {
      explanation =
        "Red and yellow regions indicate areas that strongly influenced the model's decision, often corresponding to reduced blood flow. Blue regions have minimal influence.";
    } else if (data.prediction === "Bleeding") {
      explanation =
        "Highlighted red regions represent areas the model associates with hemorrhage. These regions are most critical for classification.";
    } else {
      explanation =
        "No strongly activated regions associated with stroke were identified. Highlighted areas have minimal diagnostic relevance.";
    }

    document.getElementById("explanation").innerText = explanation;

  } catch (err) {
    alert("Error connecting to backend.");
    console.error(err);
  } finally {
    loading.style.display = "none";
  }
}
