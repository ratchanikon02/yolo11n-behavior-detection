
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
from collections import Counter

import numpy as np
import onnxruntime as ort
import gradio as gr

from PIL import Image, ImageDraw, ImageFont


MODEL_PATH = Path(__file__).parent / "best (43).onnx"

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

COLORS = [
    "#FF6B6B",
    "#4D96FF",
    "#FFD93D",
    "#6BCB77",
    "#B983FF",
]

# โหลด ONNX โดยไม่ต้องใช้ PyTorch
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
input_name = input_info.name
output_name = session.get_outputs()[0].name

shape = input_info.shape
IMG_H = shape[2] if isinstance(shape[2], int) else 416
IMG_W = shape[3] if isinstance(shape[3], int) else 416

NUM_CLASSES = len(CLASS_NAMES)


def preprocess(image):
    image = image.convert("RGB")
    original = np.asarray(image)

    h, w = original.shape[:2]

    # Letterbox ให้ตรงกับรูปแบบ YOLO
    scale = min(IMG_W / w, IMG_H / h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR
    )

    canvas = np.full(
        (IMG_H, IMG_W, 3),
        114,
        dtype=np.uint8
    )

    left = (IMG_W - new_w) // 2
    top = (IMG_H - new_h) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = np.asarray(resized)

    tensor = (
        canvas.astype(np.float32)
        .transpose(2, 0, 1)[None] / 255.0
    )

    return tensor, scale, left, top


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = max(0, box[2] - box[0]) * max(
        0, box[3] - box[1]
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    return intersection / (
        area1 + area2 - intersection + 1e-7
    )


def nms(boxes, scores, threshold=0.45):
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0 and len(keep) < 100:
        idx = order[0]
        keep.append(idx)

        if len(order) == 1:
            break

        overlap = iou_one_to_many(
            boxes[idx], boxes[order[1:]]
        )

        order = order[1:][overlap <= threshold]

    return keep


def detect(image, confidence):
    if image is None:
        return None, "Please upload or paste an image."

    image = image.convert("RGB")

    # ลดขนาดภาพต้นฉบับขนาดใหญ่มาก
    image.thumbnail((1024, 1024))

    tensor, scale, left, top = preprocess(image)

    raw = session.run(
        [output_name],
        {input_name: tensor}
    )[0]

    # รองรับ YOLO11 Detect ONNX:
    # (1, 4 + classes, predictions)
    predictions = np.squeeze(raw)

    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected ONNX output: {raw.shape}"
        )

    expected = 4 + NUM_CLASSES

    if predictions.shape[0] == expected:
        predictions = predictions.T
    elif predictions.shape[1] != expected:
        raise ValueError(
            f"Unexpected output shape: {raw.shape}. "
            "Expected raw YOLO11 detection output."
        )

    xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    valid = scores >= float(confidence)

    xywh = xywh[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    counts = Counter()
    width, height = image.size

    if len(scores) > 0:
        boxes = np.zeros_like(xywh)

        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        # แปลงพิกัดกลับเป็นขนาดภาพจริง
        boxes[:, [0, 2]] = (
            boxes[:, [0, 2]] - left
        ) / scale

        boxes[:, [1, 3]] = (
            boxes[:, [1, 3]] - top
        ) / scale

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]], 0, width
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]], 0, height
        )

        # NMS แยกตามคลาส
        for cls in np.unique(class_ids):
            indices = np.where(class_ids == cls)[0]

            kept = nms(
                boxes[indices],
                scores[indices]
            )

            for idx in indices[kept]:
                x1, y1, x2, y2 = boxes[idx]

                cls_id = int(class_ids[idx])
                score = float(scores[idx])

                counts[cls_id] += 1

                color = COLORS[cls_id % len(COLORS)]

                draw.rectangle(
                    [float(x1), float(y1),
                     float(x2), float(y2)],
                    outline=color,
                    width=3
                )

                label = (
                    f"{CLASS_NAMES[cls_id]} "
                    f"{score:.0%}"
                )

                text_box = draw.textbbox(
                    (0, 0), label
                )

                text_w = text_box[2] - text_box[0]
                text_h = text_box[3] - text_box[1]

                label_y = max(
                    0, int(y1) - text_h - 8
                )

                draw.rectangle(
                    [
                        int(x1),
                        label_y,
                        int(x1) + text_w + 10,
                        label_y + text_h + 8
                    ],
                    fill=color
                )

                draw.text(
                    (int(x1) + 5, label_y + 3),
                    label,
                    fill="black"
                )

    if counts:
        summary = "\n".join(
            f"{CLASS_NAMES[cls]}: {count}"
            for cls, count in sorted(counts.items())
        )
    else:
        summary = "No behavior detected."

    return annotated, summary


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
    title="YOLO11n Behavior Detection"
) as demo:

    gr.Markdown(
        "# YOLO11n Behavior Detection\n"
        "Upload, drag & drop, or paste using Ctrl+V."
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
        outputs=[image_output, text_output],
        trigger_mode="always_last"
    )

    confidence.change(
        fn=detect,
        inputs=[image_input, confidence],
        outputs=[image_output, text_output],
        trigger_mode="always_last"
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
    port = int(os.environ.get("PORT", "10000"))

    demo.queue(
        default_concurrency_limit=1,
        max_size=3
    ).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        js=PASTE_JS
    )
