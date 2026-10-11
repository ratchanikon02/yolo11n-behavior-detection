
import os
import io
import base64
import threading
from pathlib import Path
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

MODEL_PATH = next(
    (
        str(p)
        for p in [
            Path("best.onnx"),
            Path("best (43).onnx"),
            Path("best.pt"),
            Path("best (43).pt"),
        ]
        if p.is_file() and p.stat().st_size > 1000
    ),
    None,
)

if not MODEL_PATH:
    raise FileNotFoundError("Please upload your ONNX model.")

print("Loading model:", MODEL_PATH, flush=True)
model = YOLO(MODEL_PATH, task="detect")
model_lock = threading.Lock()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO Image Detection Application</title>
<style>
* { box-sizing: border-box; }
body {
    font-family: Arial, sans-serif;
    background: #f5f7fb;
    color: #172033;
    margin: 0;
    padding: 28px;
}
.container { max-width: 1250px; margin: auto; }
h1 { margin-bottom: 8px; }
.subtitle { color: #64748b; margin-bottom: 24px; }
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
.panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
}
.panel h3 { margin-top: 0; }
.dropzone {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    min-height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    overflow: hidden;
    text-align: center;
}
.dropzone.dragover {
    border-color: #2563eb;
    background: #eff6ff;
}
img {
    display: block;
    max-width: 100%;
    max-height: 450px;
    object-fit: contain;
    margin: auto;
}
button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 18px;
    cursor: pointer;
    margin-top: 12px;
}
button:hover { background: #1d4ed8; }
.controls { margin-top: 20px; }
input[type=range] { width: 100%; }
#status { font-weight: bold; margin-top: 15px; }
#summary { white-space: pre-line; line-height: 1.8; }
@media(max-width: 750px) {
    .grid { grid-template-columns: 1fr; }
    body { padding: 14px; }
}
</style>
</head>
<body>
<div class="container">
<h1>YOLO Image Detection Application</h1>
<p class="subtitle">
Upload, drag & drop, or press Ctrl+V to paste an image.
Detection starts automatically.
</p>

<div class="grid">
    <div class="panel">
        <h3>Input Image</h3>
        <div id="dropzone" class="dropzone">
            <div id="placeholder">
                Click to upload<br><br>
                Drag an image here<br><br>
                Or press Ctrl+V
            </div>
            <img id="preview" hidden>
        </div>
        <input id="fileInput" type="file"
               accept="image/*" hidden>
        <button id="uploadBtn">Choose Image</button>
        <button id="clearBtn" style="background:#64748b">
            Clear
        </button>
    </div>

    <div class="panel">
        <h3>Detection Result</h3>
        <img id="result" hidden>
        <p id="emptyResult">Waiting for image...</p>
        <p id="status"></p>
    </div>
</div>

<div class="panel controls">
    <label>
        Confidence Threshold:
        <strong id="confValue">0.25</strong>
    </label>
    <input id="confidence" type="range"
           min="0.1" max="0.9" step="0.05" value="0.25">
    <h3>Detected Behaviors</h3>
    <div id="summary">No image selected.</div>
</div>
</div>

<script>
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const statusText = document.getElementById("status");
const summary = document.getElementById("summary");
const confidence = document.getElementById("confidence");

let currentFile = null;
let requestId = 0;
let previewURL = null;

document.getElementById("uploadBtn").onclick =
    () => fileInput.click();

dropzone.onclick = () => fileInput.click();

fileInput.onchange = () => {
    if (fileInput.files[0]) {
        useImage(fileInput.files[0]);
    }
};

document.addEventListener("paste", (event) => {
    const items = event.clipboardData?.items || [];

    for (const item of items) {
        if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
                event.preventDefault();
                useImage(file);
                return;
            }
        }
    }
});

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");

    const file = [...event.dataTransfer.files]
        .find(f => f.type.startsWith("image/"));

    if (file) useImage(file);
});

function useImage(file) {
    currentFile = file;
    fileInput.value = "";

    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = URL.createObjectURL(file);

    preview.src = previewURL;
    preview.hidden = false;
    document.getElementById("placeholder").hidden = true;

    detectImage();
}

async function detectImage() {
    if (!currentFile) return;

    const id = ++requestId;
    const form = new FormData();
    form.append("image", currentFile);
    form.append("confidence", confidence.value);

    result.hidden = true;
    document.getElementById("emptyResult").hidden = false;
    statusText.textContent = "Detecting...";
    summary.textContent = "Processing image...";

    const start = performance.now();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (id !== requestId) return;
        if (!response.ok) {
            throw new Error(data.error || "Detection failed");
        }

        result.src = data.image;
        result.hidden = false;
        document.getElementById("emptyResult").hidden = true;
        summary.textContent = data.summary;

        const seconds = (
            (performance.now() - start) / 1000
        ).toFixed(1);

        statusText.textContent = "Completed in " + seconds + "s";
    } catch (error) {
        if (id !== requestId) return;
        statusText.textContent = "Error: " + error.message;
        summary.textContent = "Detection failed.";
    }
}

confidence.addEventListener("input", () => {
    document.getElementById("confValue").textContent =
        confidence.value;
});

confidence.addEventListener("change", () => {
    if (currentFile) detectImage();
});

document.getElementById("clearBtn").onclick = () => {
    ++requestId;
    currentFile = null;
    fileInput.value = "";
    preview.hidden = true;
    result.hidden = true;

    document.getElementById("placeholder").hidden = false;
    document.getElementById("emptyResult").hidden = false;

    statusText.textContent = "";
    summary.textContent = "No image selected.";

    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null;
};
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(error="No image uploaded"), 400

        confidence = float(request.form.get("confidence", 0.25))
        confidence = max(0.1, min(0.9, confidence))

        image = Image.open(uploaded.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Reduce image size before inference
        image.thumbnail((800, 800))

        with model_lock:
            prediction = model.predict(
                source=image,
                conf=confidence,
                imgsz=320,
                device="cpu",
                max_det=30,
                verbose=False,
            )[0]

        annotated = Image.fromarray(
            prediction.plot()[:, :, ::-1]
        )

        counts = Counter(
            int(c) for c in prediction.boxes.cls.tolist()
        )

        if counts:
            summary = "\\n".join(
                f"{CLASS_NAMES.get(c, str(c))}: {n}"
                for c, n in sorted(counts.items())
            )
        else:
            summary = "No behavior detected."

        buffer = io.BytesIO()
        annotated.save(
            buffer,
            format="JPEG",
            quality=75,
            optimize=True
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        return jsonify(
            image="data:image/jpeg;base64," + encoded,
            summary=summary
        )

    except Exception as e:
        print("Detection error:", repr(e), flush=True)
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True
    )


