import cv2
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


"""
Script 1 - Local annotation and SAM2 preview.

Purpose:
    - Navigate inside a video.
    - Add positive / negative prompts.
    - Preview SAM2 masks on selected frames.
    - Save annotations as a timestamped JSON file.

Input:
    data/input/video_without_cut.mp4
    SAM_2/checkpoints/sam2.1_hiera_small.pt

Output:
    SAM_2/annotations/annotations_video_without_cut_YYYYMMDD_HHMMSS.json
"""

# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = PROJECT_ROOT / "SAM_2"

VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "video_without_cut.mp4"

CHECKPOINT_PATH = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR = SAM2_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ANNOTATION_DIR = SAM2_ROOT / "annotations"
ANNOTATION_DIR.mkdir(exist_ok=True)


# -----------------------------
# Run identifiers
# -----------------------------

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ANNOTATION_PATH = ANNOTATION_DIR / f"annotations_{VIDEO_PATH.stem}_{run_timestamp}.json"


# -----------------------------
# Device
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_default_dtype(torch.float32)
torch.set_grad_enabled(False)

print(f"Using device: {DEVICE}")


# -----------------------------
# General parameters
# -----------------------------

current_frame_idx = 50
active_obj_id = 1

frame = None
original_frame = None
cap = None
image_predictor = None

total_frames = 0
fps = 0.0
save_counter = 0


# -----------------------------
# Annotation structure
# -----------------------------

annotations = {
    "video": str(VIDEO_PATH),
    "video_name": VIDEO_PATH.name,
    "created_at": run_timestamp,
    "fps": None,
    "total_frames": None,
    "objects": {}
}


OBJECT_COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (128, 255, 0),
    8: (255, 128, 0),
    9: (128, 0, 255),
}


# -----------------------------
# Utility functions
# -----------------------------

def get_obj_color(obj_id: int):
    return OBJECT_COLORS.get(obj_id, (200, 200, 200))


def ensure_object_entry(obj_id: int):
    obj_key = str(obj_id)

    if obj_key not in annotations["objects"]:
        annotations["objects"][obj_key] = {
            "label": f"object_{obj_id}",
            "prompts": []
        }


def get_prompt_entry(obj_id: int, frame_idx: int):
    ensure_object_entry(obj_id)

    obj_key = str(obj_id)
    prompts = annotations["objects"][obj_key]["prompts"]

    for prompt in prompts:
        if int(prompt["frame"]) == int(frame_idx):
            return prompt

    new_prompt = {
        "frame": int(frame_idx),
        "points": [],
        "labels": []
    }

    prompts.append(new_prompt)
    return new_prompt


def load_frame(frame_idx: int):
    global frame, original_frame, current_frame_idx

    if frame_idx < 0:
        frame_idx = 0

    if frame_idx >= total_frames:
        frame_idx = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()

    if not ret:
        print(f"Error: could not read frame {frame_idx}.")
        return False

    current_frame_idx = int(frame_idx)
    original_frame = img.copy()
    frame = img.copy()

    return True


def draw_points(img):
    out = img.copy()

    for obj_key, obj_data in annotations["objects"].items():
        obj_id = int(obj_key)
        color = get_obj_color(obj_id)

        for prompt in obj_data["prompts"]:
            if int(prompt["frame"]) != int(current_frame_idx):
                continue

            points = prompt["points"]
            labels = prompt["labels"]

            for (x, y), label in zip(points, labels):
                x, y = int(x), int(y)

                if int(label) == 1:
                    cv2.circle(out, (x, y), 6, color, -1)
                else:
                    cv2.circle(out, (x, y), 6, color, 2)
                    cv2.line(out, (x - 5, y - 5), (x + 5, y + 5), color, 2)
                    cv2.line(out, (x - 5, y + 5), (x + 5, y - 5), color, 2)

                cv2.putText(
                    out,
                    f"obj {obj_id}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

    return out


def update_window_title():
    title = (
        f"Annotation + SAM Preview | "
        f"frame={current_frame_idx}/{total_frames - 1} | "
        f"active_obj={active_obj_id}"
    )
    cv2.setWindowTitle("Annotation Tool", title)


def show_current_frame():
    """
    Display the current frame and preview SAM2 masks for the prompts
    added on this frame.
    """
    global frame

    vis = original_frame.copy()

    image_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    image_predictor.set_image(image_rgb)

    with torch.inference_mode():
        if DEVICE == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_context = torch.autocast(device_type="cpu", enabled=False)

        with autocast_context:
            for obj_key, obj_data in annotations["objects"].items():
                obj_id = int(obj_key)
                color = get_obj_color(obj_id)

                for prompt in obj_data["prompts"]:
                    if int(prompt["frame"]) != int(current_frame_idx):
                        continue

                    points = prompt["points"]
                    labels = prompt["labels"]

                    if len(points) == 0:
                        continue

                    point_coords = np.array(points, dtype=np.float32)
                    point_labels = np.array(labels, dtype=np.int32)

                    masks, scores, _ = image_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )

                    best_idx = int(np.argmax(scores))
                    mask = masks[best_idx].astype(np.uint8)

                    overlay = vis.copy()
                    overlay[mask > 0] = color
                    vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)

                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, color, 2)

                    print(
                        f"Frame {current_frame_idx} | "
                        f"Object {obj_id} | "
                        f"local score: {scores[best_idx]:.4f}"
                    )

    frame = draw_points(vis)

    cv2.putText(
        frame,
        f"Frame {current_frame_idx} | Active object {active_obj_id}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    update_window_title()
    cv2.imshow("Annotation Tool", frame)


def add_point(x: int, y: int, label: int):
    prompt = get_prompt_entry(active_obj_id, current_frame_idx)

    prompt["points"].append([int(x), int(y)])
    prompt["labels"].append(int(label))

    sign = "+" if int(label) == 1 else "-"
    print(f"Object {active_obj_id} | Frame {current_frame_idx} | point {sign}: ({x}, {y})")

    show_current_frame()


def remove_last_point():
    obj_key = str(active_obj_id)

    if obj_key not in annotations["objects"]:
        print("No point to remove.")
        return

    for prompt in annotations["objects"][obj_key]["prompts"]:
        if int(prompt["frame"]) == int(current_frame_idx):
            if len(prompt["points"]) > 0:
                removed_point = prompt["points"].pop()
                removed_label = prompt["labels"].pop()
                print(f"Removed last point: {removed_point}, label={removed_label}")
            else:
                print("No point to remove on this frame.")
            break

    show_current_frame()


def reset_current_frame_annotations():
    obj_key = str(active_obj_id)

    if obj_key not in annotations["objects"]:
        print("No annotations for this object.")
        return

    prompts = annotations["objects"][obj_key]["prompts"]

    annotations["objects"][obj_key]["prompts"] = [
        p for p in prompts if int(p["frame"]) != int(current_frame_idx)
    ]

    print(f"Reset annotations for object {active_obj_id} on frame {current_frame_idx}.")
    show_current_frame()


def save_annotations():
    """
    Save all annotation prompts to a timestamped JSON file.
    """
    annotations["fps"] = fps
    annotations["total_frames"] = total_frames

    for obj_data in annotations["objects"].values():
        obj_data["prompts"] = sorted(obj_data["prompts"], key=lambda p: int(p["frame"]))

    with open(ANNOTATION_PATH, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)

    print(f"Annotations saved: {ANNOTATION_PATH}")


def save_current_view():
    global save_counter

    out_path = OUTPUT_DIR / f"annotation_preview_frame_{current_frame_idx}_{run_timestamp}_{save_counter}.jpg"
    cv2.imwrite(str(out_path), frame)

    print(f"Current view saved: {out_path}")

    save_counter += 1


def change_frame(new_idx: int):
    ok = load_frame(new_idx)

    if ok:
        show_current_frame()
        print(f"Current frame: {current_frame_idx}")


def mouse_callback(event, x, y, flags, param):
    """
    Convert mouse clicks into SAM2 prompts:
    left click = positive point, right click = negative point.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        add_point(x, y, label=1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        add_point(x, y, label=0)


# -----------------------------
# Main program
# -----------------------------

print("Starting local annotation tool with SAM preview...")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

print("Loading SAM2 image predictor...")

image_model = build_sam2(MODEL_CFG, str(CHECKPOINT_PATH), device=DEVICE)
image_model = image_model.float()
image_predictor = SAM2ImagePredictor(image_model)

print("SAM2 image predictor loaded.")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

annotations["fps"] = fps
annotations["total_frames"] = total_frames

print(f"Video loaded: {VIDEO_PATH}")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Annotation output: {ANNOTATION_PATH}")

if not load_frame(current_frame_idx):
    cap.release()
    raise RuntimeError("Could not load initial frame.")

cv2.namedWindow("Annotation Tool")
cv2.setMouseCallback("Annotation Tool", mouse_callback)
show_current_frame()


print("\nKeyboard commands:")
print("Navigation:")
print("  q: previous frame")
print("  d: next frame")
print("  b: -25 frames")
print("  f: +25 frames")
print("")
print("Objects:")
print("  keys 1 to 9: select active object")
print("  left click: positive point")
print("  right click: negative point")
print("")
print("Actions:")
print("  s: save annotations JSON")
print("  v: save current visual preview")
print("  u: remove last point for active object on current frame")
print("  r: reset active object annotations on current frame")
print("  ESC: quit")
print("")

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == 27:
        break

    elif ord("1") <= key <= ord("9"):
        active_obj_id = key - ord("0")
        print(f"Active object = {active_obj_id}")
        show_current_frame()

    elif key == ord("q"):
        change_frame(current_frame_idx - 1)

    elif key == ord("d"):
        change_frame(current_frame_idx + 1)

    elif key == ord("b"):
        change_frame(current_frame_idx - 25)

    elif key == ord("f"):
        change_frame(current_frame_idx + 25)

    elif key == ord("s"):
        save_annotations()

    elif key == ord("v"):
        save_current_view()

    elif key == ord("u"):
        remove_last_point()

    elif key == ord("r"):
        reset_current_frame_annotations()

cv2.destroyAllWindows()
cap.release()

save_annotations()
print("Program ended.")