"""Interactive SAM2 image annotator (Gradio UI).

Pick a frame from an image directory, click points (positive/negative) and/or
draw a box with two clicks, and save a DAVIS-palette PNG mask.

Example:
    python preproc/interactive_image_annotator.py \
        --ckpt preproc/checkpoints/sam2.1_hiera_base_plus.pt \
        --cfg  configs/sam2.1/sam2.1_hiera_b+.yaml

Then in the browser: load a directory of frames, click to prompt, save. The
output lands at ``<out_dir>/<seq_name>/<frame_stem>.png`` — pass
``<out_dir>/<seq_name>`` as ``compute_masks_sam2.py --input_mask_dir``.
"""

import os
import sys
from argparse import ArgumentParser

# Make the bundled sam2 submodule importable (mirrors compute_masks_sam2.py).
_basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_basedir, "sam2"))

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# DAVIS 2017 palette — same bytes shipped in sam2/tools/vos_inference.py.
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# tab10-inspired colors for up to 10 objects before wrap-around.
OBJ_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def empty_state():
    return {
        "img_dir": None,
        "frame_names": [],
        "frame_idx": 0,
        "image": None,            # np.uint8 (H, W, 3)
        "objects": {},            # obj_id -> {points, box, mask}
        "pending_box": None,      # (x, y) first corner while drawing a box
    }


def ensure_obj(state, obj_id):
    if obj_id not in state["objects"]:
        state["objects"][obj_id] = {"points": [], "box": None, "mask": None}


def list_frames(img_dir):
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] in IMG_EXTS]
    try:
        files.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        files.sort()
    return files


def obj_color(obj_id):
    return OBJ_COLORS[(int(obj_id) - 1) % len(OBJ_COLORS)]


# ---------------------------------------------------------------------------
# SAM2 inference
# ---------------------------------------------------------------------------

def run_predict_for_obj(predictor, state, obj_id):
    obj = state["objects"][obj_id]
    if not obj["points"] and obj["box"] is None:
        obj["mask"] = None
        return
    coords = labels = None
    if obj["points"]:
        coords = np.array([[p[0], p[1]] for p in obj["points"]], dtype=np.float32)
        labels = np.array([p[2] for p in obj["points"]], dtype=np.int32)
    box = np.array(obj["box"], dtype=np.float32) if obj["box"] is not None else None
    masks, _, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        box=box,
        multimask_output=False,
    )
    obj["mask"] = masks[0].astype(bool)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_overlay(state):
    if state["image"] is None:
        return None
    img = state["image"]
    overlay = img.astype(np.float32)

    # Alpha-blend masks.
    for obj_id, obj in sorted(state["objects"].items()):
        if obj["mask"] is None:
            continue
        color = np.array(obj_color(obj_id), dtype=np.float32)
        m = obj["mask"]
        overlay[m] = overlay[m] * 0.5 + color * 0.5
    overlay = overlay.astype(np.uint8)

    # Mask contours.
    for obj_id, obj in sorted(state["objects"].items()):
        if obj["mask"] is None:
            continue
        color = obj_color(obj_id)
        m = obj["mask"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    # Boxes.
    for obj_id, obj in sorted(state["objects"].items()):
        if obj["box"] is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in obj["box"]]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), obj_color(obj_id), 2)

    # Points.
    for obj_id, obj in sorted(state["objects"].items()):
        for x, y, label in obj["points"]:
            c = (0, 220, 0) if label == 1 else (220, 0, 0)
            cv2.circle(overlay, (int(x), int(y)), 6, c, -1)
            cv2.circle(overlay, (int(x), int(y)), 6, (255, 255, 255), 2)

    # Pending box first corner.
    if state["pending_box"] is not None:
        x, y = state["pending_box"]
        cv2.drawMarker(overlay, (int(x), int(y)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)

    return overlay


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_mask(state, out_dir, seq_name):
    if state["image"] is None:
        return "No image loaded."
    if not out_dir or not out_dir.strip():
        return "Set a mask output directory first."
    if not seq_name or not seq_name.strip():
        return "Set a sequence name first."
    if not state["objects"] or all(o["mask"] is None for o in state["objects"].values()):
        return "No masks to save — add at least one prompt."

    H, W = state["image"].shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)
    # Paint highest obj_id last so it ends up on top (matches vos_inference).
    for obj_id in sorted(state["objects"]):
        m = state["objects"][obj_id]["mask"]
        if m is None:
            continue
        combined[m] = int(obj_id)

    target_dir = os.path.join(out_dir.strip(), seq_name.strip())
    os.makedirs(target_dir, exist_ok=True)

    frame_stem = os.path.splitext(state["frame_names"][state["frame_idx"]])[0]
    out_path = os.path.join(target_dir, f"{frame_stem}.png")

    mask_img = Image.fromarray(combined, mode="P")
    mask_img.putpalette(DAVIS_PALETTE)
    mask_img.save(out_path)
    return f"Saved mask -> {out_path}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(predictor, img_path="./preproc/example/bike-packing", out_dir="./outputs/annotator_masks"):
    def _set_predictor_image(state):
        """SAM2 image predictor caches embeddings per image — refresh on load/frame change."""
        predictor.set_image(state["image"])

    with gr.Blocks(title="SAM2 Interactive Annotator") as demo:
        state = gr.State(empty_state())

        gr.Markdown(
            "## SAM2 Interactive Image Annotator\n"
            "Load a single image or a directory of frames, click to add **positive/negative points** "
            "or draw a **box** (two clicks = two corners), then save. The output is a DAVIS-palette PNG mask."
        )

        with gr.Row():
            img_path_tb = gr.Textbox(label="Image path (file or directory)", value=img_path, scale=4)
            out_dir_tb = gr.Textbox(label="Mask output directory", value=out_dir, scale=3)
            seq_name_tb = gr.Textbox(label="Output sequence name", scale=2)
            load_btn = gr.Button("Load", scale=1, variant="primary")

        frame_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Frame", visible=False)

        with gr.Row():
            with gr.Column(scale=4):
                image_display = gr.Image(
                    label="Click to prompt",
                    type="numpy",
                    interactive=False,
                )
            with gr.Column(scale=1):
                mode_radio = gr.Radio(
                    choices=["Point (+)", "Point (-)", "Box"],
                    value="Point (+)",
                    label="Prompt mode",
                )
                obj_id_num = gr.Number(value=1, precision=0, label="Active object ID", minimum=1)
                undo_btn = gr.Button("Undo last prompt")
                clear_obj_btn = gr.Button("Clear active object")
                clear_all_btn = gr.Button("Clear all")
                save_btn = gr.Button("Save mask", variant="primary")
                status_tb = gr.Textbox(label="Status", value="", interactive=False, lines=2)

        # -- Load: accept either a single image file or a directory of frames.
        # For a directory, index all frame filenames and load frame 0. For a
        # file, load just that one image. SAM2's embedding is primed on the
        # loaded frame; other frames in a directory are loaded lazily in
        # do_frame_change when the user navigates to them.
        def do_load(img_path, state):
            state = empty_state()
            if not img_path or not os.path.exists(img_path):
                return state, None, gr.update(), gr.update(), f"Bad path: {img_path!r}"

            if os.path.isfile(img_path):
                if os.path.splitext(img_path)[1] not in IMG_EXTS:
                    return state, None, gr.update(), gr.update(), f"Unsupported image: {img_path!r}"
                img_dir = os.path.dirname(os.path.abspath(img_path))
                frames = [os.path.basename(img_path)]
                auto_seq = gr.update()  # single file: don't touch seq name
                status = f"Loaded single image {img_path}"
            else:
                img_dir = img_path
                frames = list_frames(img_dir)
                if not frames:
                    return state, None, gr.update(), gr.update(), f"No images under {img_dir}"
                auto_seq = gr.update(value=os.path.basename(os.path.normpath(img_dir)))
                status = f"Loaded {len(frames)} frames from {img_dir}"

            state["img_dir"] = img_dir
            state["frame_names"] = frames
            state["frame_idx"] = 0
            state["image"] = np.array(
                Image.open(os.path.join(img_dir, frames[0])).convert("RGB")
            )
            _set_predictor_image(state)
            return (
                state,
                render_overlay(state),
                gr.update(
                    minimum=0,
                    maximum=max(len(frames) - 1, 0),
                    value=0,
                    visible=len(frames) > 1,
                ),
                auto_seq,
                status,
            )

        load_btn.click(
            do_load,
            inputs=[img_path_tb, state],
            outputs=[state, image_display, frame_slider, seq_name_tb, status_tb],
        )

        # -- Frame change: switch to another frame in the loaded dir. Loads the
        # new image, re-runs SAM2's set_image to refresh the cached embedding,
        # and clears any prior prompts (they were tied to the old embedding).
        def do_frame_change(frame_idx, state):
            if state["img_dir"] is None:
                return state, None, "Load an image first."
            frame_idx = int(frame_idx)
            if frame_idx == state["frame_idx"] and state["image"] is not None:
                return state, render_overlay(state), gr.update()
            state["frame_idx"] = frame_idx
            state["image"] = np.array(
                Image.open(
                    os.path.join(state["img_dir"], state["frame_names"][frame_idx])
                ).convert("RGB")
            )
            # Prompts are frame-specific: switching frames drops them.
            state["objects"] = {}
            state["pending_box"] = None
            _set_predictor_image(state)
            return (
                state,
                render_overlay(state),
                f"Frame {frame_idx}: {state['frame_names'][frame_idx]} (prompts cleared)",
            )

        frame_slider.change(
            do_frame_change,
            inputs=[frame_slider, state],
            outputs=[state, image_display, status_tb],
        )

        # -- Click on image ---------------------------------------------
        def do_click(state, mode, obj_id, evt: gr.SelectData):
            if state["image"] is None:
                return state, None, "Load an image first."
            obj_id = int(obj_id)
            ensure_obj(state, obj_id)
            # Gradio floors click coords to integer pixels, so this is a whole
            # number cast to float — no sub-pixel precision. SAM2 accepts true
            # floats, but pixel-level is fine for human clicks. For sub-pixel
            # prompts (e.g. programmatic seeding), bypass this handler.
            x, y = float(evt.index[0]), float(evt.index[1])

            # Safety net: reject clicks reported outside the image bounds.
            H, W = state["image"].shape[:2]
            if not (0 <= x < W and 0 <= y < H):
                return (
                    state,
                    render_overlay(state),
                    f"Click ({int(x)}, {int(y)}) is outside image {W}x{H}; ignored.",
                )

            if mode == "Point (+)":
                state["objects"][obj_id]["points"].append((x, y, 1))
                state["pending_box"] = None
            elif mode == "Point (-)":
                state["objects"][obj_id]["points"].append((x, y, 0))
                state["pending_box"] = None
            elif mode == "Box":
                if state["pending_box"] is None:
                    state["pending_box"] = (x, y)
                    return (
                        state,
                        render_overlay(state),
                        f"Box corner 1 at ({int(x)}, {int(y)}). Click the opposite corner.",
                    )
                x0, y0 = state["pending_box"]
                state["objects"][obj_id]["box"] = [
                    min(x0, x), min(y0, y), max(x0, x), max(y0, y)
                ]
                state["pending_box"] = None

            run_predict_for_obj(predictor, state, obj_id)
            H, W = state["image"].shape[:2]
            return (
                state,
                render_overlay(state),
                f"Obj {obj_id}: updated. Click at ({int(x)}, {int(y)}) / image {W}x{H}.",
            )

        image_display.select(
            do_click,
            inputs=[state, mode_radio, obj_id_num],
            outputs=[state, image_display, status_tb],
        )

        # -- Undo / clear -----------------------------------------------
        def do_undo(state, obj_id):
            if state["image"] is None:
                return state, None, "Load an image first."
            obj_id = int(obj_id)
            if state["pending_box"] is not None:
                state["pending_box"] = None
                return state, render_overlay(state), "Cleared pending box corner."
            if obj_id not in state["objects"]:
                return state, render_overlay(state), f"Obj {obj_id}: nothing to undo."
            obj = state["objects"][obj_id]
            if obj["points"]:
                obj["points"].pop()
            elif obj["box"] is not None:
                obj["box"] = None
            else:
                return state, render_overlay(state), f"Obj {obj_id}: nothing to undo."
            if obj["points"] or obj["box"] is not None:
                run_predict_for_obj(predictor, state, obj_id)
            else:
                obj["mask"] = None
            return state, render_overlay(state), f"Obj {obj_id}: undid last prompt."

        undo_btn.click(
            do_undo,
            inputs=[state, obj_id_num],
            outputs=[state, image_display, status_tb],
        )

        def do_clear_obj(state, obj_id):
            if state["image"] is None:
                return state, None, "Load an image first."
            obj_id = int(obj_id)
            state["objects"].pop(obj_id, None)
            state["pending_box"] = None
            return state, render_overlay(state), f"Cleared obj {obj_id}."

        clear_obj_btn.click(
            do_clear_obj,
            inputs=[state, obj_id_num],
            outputs=[state, image_display, status_tb],
        )

        def do_clear_all(state):
            if state["image"] is None:
                return state, None, "Load an image first."
            state["objects"] = {}
            state["pending_box"] = None
            return state, render_overlay(state), "Cleared all objects."

        clear_all_btn.click(
            do_clear_all,
            inputs=[state],
            outputs=[state, image_display, status_tb],
        )

        # -- Save --------------------------------------------------------
        def do_save(state, out_dir, seq_name):
            return save_mask(state, out_dir, seq_name)

        save_btn.click(
            do_save,
            inputs=[state, out_dir_tb, seq_name_tb],
            outputs=[status_tb],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Interactive SAM2 image annotator (Gradio).")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="preproc/checkpoints/sam2.1_hiera_large.pt",
        help="Path to the SAM2 image-predictor checkpoint.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 hydra config (resolved through the bundled sam2 package).",
    )
    parser.add_argument("--img_path", type=str, default="./preproc/example/bike-packing", help="Pre-fill the image path textbox (file or directory).")
    parser.add_argument("--out_dir", type=str, default="./outputs/annotator_masks", help="Pre-fill the mask output directory textbox.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM2 ({args.cfg}) from {args.ckpt} on {device}")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(args.cfg, args.ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    demo = build_ui(predictor, img_path=args.img_path, out_dir=args.out_dir)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
