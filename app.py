
import os
from pathlib import Path
from collections import Counter

import gradio as gr
from PIL import Image
from ultralytics import YOLO

# ใช้ไฟล์โมเดลจริงขนาด 5.26 MB
MODEL_PATH = Path(__file__).parent / "best (43).pt"
model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

def detect(image, confidence):
    if image is None:
        return None, "Please upload or paste an image."

    result = model.predict(
        source=image.convert("RGB"),
        conf=confidence,
        imgsz=640,
        device="cpu",
        verbose=False
    )[0]

    annotated = Image.fromarray(
        result.plot()[:, :, ::-1]
    )

    counts = Counter(
        int(cls) for cls in result.boxes.cls.tolist()
    )

    if counts:
        summary = "\n".join(
            f"{CLASS_NAMES.get(cls, str(cls))}: {count}"
            for cls, count in sorted(counts.items())
        )
    else:
        summary = "No behavior detected."

    return annotated, summary


# รองรับ Ctrl+V จาก Clipboard
PASTE_JS = """
() => {
    if (window.yoloPasteInstalled) return;
    window.yoloPasteInstalled = true;

    document.addEventListener("paste", (event) => {
        const items = event.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (!item.type.startsWith("image/")) continue;

            const file = item.getAsFile();
            if (!file) continue;

            const input = document.querySelector(
                '#yolo_image_input input[type="file"]'
            );

            if (!input) return;

            const transfer = new DataTransfer();
            transfer.items.add(file);

            input.files = transfer.files;
            input.dispatchEvent(
                new Event("change", { bubbles: true })
            );

            event.preventDefault();
            break;
        }
    });
}
"""

with gr.Blocks(
    title="YOLO11n Behavior Detection",
    js=PASTE_JS
) as demo:

    gr.Markdown(
        "# YOLO11n Behavior Detection\n"
        "Upload, drag & drop, or paste an image using Ctrl+V."
    )

    with gr.Row():
        image_input = gr.Image(
            label="Input Image",
            type="pil",
            sources=["upload", "clipboard"],
            image_mode="RGB",
            height=400,
            interactive=True,
            elem_id="yolo_image_input"
        )

        image_output = gr.Image(
            label="Detection Result",
            height=400,
            interactive=False
        )

    confidence = gr.Slider(
        minimum=0.1,
        maximum=0.9,
        value=0.25,
        step=0.05,
        label="Confidence Threshold"
    )

    text_output = gr.Textbox(
        label="Detected Behaviors",
        lines=6
    )

    clear_button = gr.Button("Clear")

    image_input.change(
        fn=detect,
        inputs=[image_input, confidence],
        outputs=[image_output, text_output]
    )

    confidence.change(
        fn=detect,
        inputs=[image_input, confidence],
        outputs=[image_output, text_output]
    )

    clear_button.click(
        fn=lambda: (None, None, ""),
        outputs=[
            image_input,
            image_output,
            text_output
        ]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
