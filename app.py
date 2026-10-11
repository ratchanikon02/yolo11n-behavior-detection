
import os
from pathlib import Path
from collections import Counter

# Reduce CPU usage on Render Free
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import gradio as gr
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "YOLO Image Detection Application"

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

# Prefer ONNX for CPU inference
MODEL_PATH = next(
    (
        str(p)
        for p in [
            Path("best.onnx"),
            Path("best (43).onnx"),
            Path("best (43).pt"),
            Path("best.pt"),
        ]
        if p.is_file() and p.stat().st_size > 1000
    ),
    None,
)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Model not found. Upload best.onnx or best.pt to GitHub."
    )

print(f"Loading model: {MODEL_PATH}", flush=True)
model = YOLO(MODEL_PATH, task="detect")


def detect(image, confidence):
    if image is None:
        return None, "Please upload or paste an image."

    try:
        image = image.convert("RGB")

        # Reduce processing time on free CPU
        image.thumbnail((960, 960))

        result = model.predict(
            source=image,
            conf=float(confidence),
            imgsz=416,
            device="cpu",
            verbose=False,
            max_det=30,
        )[0]

        annotated = Image.fromarray(
            result.plot()[:, :, ::-1]
        )

        counts = Counter(
            int(cls) for cls in result.boxes.cls.tolist()
        )

        if not counts:
            summary = "No behavior detected."
        else:
            summary = "\n".join(
                f"{CLASS_NAMES.get(cls, str(cls))}: {count}"
                for cls, count in sorted(counts.items())
            )

        return annotated, summary

    except Exception as e:
        return None, f"Detection error: {e}"


with gr.Blocks(title=APP_TITLE) as demo:

    gr.Markdown(f"# {APP_TITLE}")

    gr.Markdown(
        "Upload, drag & drop, or paste an image using Ctrl+V."
    )

    with gr.Row():
        image_input = gr.Image(
            label="Input Image",
            type="pil",
            image_mode="RGB",
            sources=["upload", "clipboard"],
            height=420,
        )

        image_output = gr.Image(
            label="Detection Result",
            height=420,
            interactive=False,
        )

    confidence = gr.Slider(
        minimum=0.10,
        maximum=0.90,
        value=0.25,
        step=0.05,
        label="Confidence Threshold",
    )

    text_output = gr.Textbox(
        label="Detected Behaviors",
        lines=5,
    )

    clear_button = gr.Button("Clear Image")

    image_input.change(
        fn=detect,
        inputs=[image_input, confidence],
        outputs=[image_output, text_output],
        show_progress="minimal",
        concurrency_limit=1,
    )

    confidence.release(
        fn=detect,
        inputs=[image_input, confidence],
        outputs=[image_output, text_output],
        show_progress="minimal",
        concurrency_limit=1,
    )

    clear_button.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[image_input, image_output, text_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

