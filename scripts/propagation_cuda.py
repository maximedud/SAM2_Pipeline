import json
import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from sam2.build_sam import build_sam2_video_predictor

"""
Script 2 - SAM2 video propagation and export.

Purpose:
    - Load annotation prompts created by annotation_sam_preview.py.
    - Run SAM2 video propagation on GPU.
    - Export a segmented video, CSV summary and contour coordinates.

Input:
    data/input/video_without_cut.mp4
    SAM_2/annotations/annotations_video_without_cut_*.json
    SAM_2/checkpoints/sam2.1_hiera_small.pt

Output:
    SAM_2/outputs/gpu_propagation_*/
        - video_segmented_with_gaze.mp4
        - object_coordinates.csv
        - object_contours.json
        - previews/
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = PROJECT_ROOT / "SAM_2"

VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "video_without_cut.mp4"
CHECKPOINT_PATH = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

ANNOTATION_DIR = SAM2_ROOT / "annotations"
annotation_files = sorted(
    ANNOTATION_DIR.glob(f"annotations_{VIDEO_PATH.stem}_*.json")
)

if not annotation_files:
    raise FileNotFoundError(f"No annotation file found for {VIDEO_PATH.stem}")

ANNOTATION_PATH = annotation_files[-1]

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = SAM2_ROOT / "outputs" / f"gpu_propagation_{RUN_TIMESTAMP}"
PREVIEW_DIR = OUTPUT_DIR / "previews"
MASK_DIR = OUTPUT_DIR / "masks"

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "video_segmented_with_gaze.mp4"
CSV_PATH = OUTPUT_DIR / "object_coordinates.csv"
CONTOURS_JSON_PATH = OUTPUT_DIR / "object_contours.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


REQUIRE_CUDA = True
PREVIEW_STEP = 25
SAVE_PREVIEWS = True
SAVE_MASKS = False
SAVE_CONTOURS_JSON = True
DRAW_FAKE_GAZE = True
# Demo-only visual placeholder. This is not real eye-tracking data. It will be added latter
OUTPUT_FPS = None
MIN_CONTOUR_AREA = 10


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


def get_obj_color(obj_id: int):
    return OBJECT_COLORS.get(obj_id, (200, 200, 200))


def load_annotations(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return total_frames, fps, width, height


def read_exact_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def mask_to_bbox_centroid_area(mask: np.ndarray):
    """
    Convert a binary mask into simple geometric descriptors:
    detection flag, bounding box, centroid and area.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return {
            "detected": 0,
            "x_min": "",
            "y_min": "",
            "x_max": "",
            "y_max": "",
            "centroid_x": "",
            "centroid_y": "",
            "area": 0
        }

    return {
        "detected": 1,
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "area": int(mask.sum())
    }


def mask_to_contours(mask: np.ndarray):
    """
    Convert a binary SAM2 mask into contour coordinates.
    Small contours are filtered using MIN_CONTOUR_AREA.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    exported = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < MIN_CONTOUR_AREA:
            continue

        points = contour.squeeze(axis=1)

        if points.ndim != 2:
            continue

        exported.append({
            "area": area,
            "points": points.astype(int).tolist()
        })

    return exported


def draw_fake_gaze(frame, frame_idx: int, total_frames: int):
    """
    Draw a synthetic gaze point for visual demonstration only.
    This function does not use real eye-tracking data.
    """
    h, w = frame.shape[:2]

    waypoints = [
        (0.30, 0.35),
        (0.45, 0.30),
        (0.55, 0.45),
        (0.65, 0.55),
        (0.50, 0.40),
        (0.35, 0.60),
    ]

    t = frame_idx / max(total_frames - 1, 1)
    scaled_t = t * (len(waypoints) - 1)

    i = int(np.floor(scaled_t))
    i = min(i, len(waypoints) - 2)

    alpha = scaled_t - i

    x1, y1 = waypoints[i]
    x2, y2 = waypoints[i + 1]

    x = int(((1 - alpha) * x1 + alpha * x2) * w)
    y = int(((1 - alpha) * y1 + alpha * y2) * h)

    cv2.circle(frame, (x, y), 13, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 21, (255, 255, 255), 2)

    return frame


def draw_masks_on_frame(frame, frame_masks, frame_idx: int, total_frames: int):
    """
    Overlay SAM2 masks, contours, object labels and optional fake gaze point
    on the original video frame.
    """
    out = frame.copy()

    for obj_id in sorted(frame_masks.keys()):
        mask = frame_masks[obj_id]
        color = get_obj_color(obj_id)

        overlay = out.copy()
        overlay[mask > 0] = color
        out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        stats = mask_to_bbox_centroid_area(mask)

        if stats["detected"] == 1:
            x_min = stats["x_min"]
            y_min = stats["y_min"]

            cv2.putText(
                out,
                f"obj {obj_id}",
                (x_min, max(y_min - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

    cv2.putText(
        out,
        f"Frame {frame_idx}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if DRAW_FAKE_GAZE:
        out = draw_fake_gaze(out, frame_idx, total_frames)

    return out


def get_annotated_frames(annotation_data):
    frames = set()

    for obj_data in annotation_data["objects"].values():
        for prompt in obj_data["prompts"]:
            frames.add(int(prompt["frame"]))

    return sorted(frames)


def build_sampled_frames(total_frames: int, preview_step: int, annotated_frames):
    sampled = set(range(0, total_frames, preview_step))
    sampled.update(annotated_frames)
    sampled.add(total_frames - 1)
    return sampled


def open_video_writer(path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    return writer


print("Starting GPU propagation export script...")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

annotation_data = load_annotations(ANNOTATION_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Video: {VIDEO_PATH}")
print(f"Annotations: {ANNOTATION_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

if REQUIRE_CUDA and DEVICE != "cuda":
    raise RuntimeError("CUDA is not available. This script is intended to run on GPU.")

total_frames, fps, width, height = get_video_info(VIDEO_PATH)
if fps <= 0:
    raise RuntimeError("Invalid FPS value read from video.")

output_fps = fps if OUTPUT_FPS is None else OUTPUT_FPS

print(f"Video info: {total_frames} frames | {fps} fps | {width}x{height}")

annotated_frames = get_annotated_frames(annotation_data)
sampled_frames = build_sampled_frames(total_frames, PREVIEW_STEP, annotated_frames)

print(f"Annotated frames: {annotated_frames}")
print(f"Preview frames to save: {len(sampled_frames)}")

print("Loading SAM2 video predictor...")

predictor = build_sam2_video_predictor(
    MODEL_CFG,
    str(CHECKPOINT_PATH),
    device=DEVICE,
)

print("SAM2 video predictor loaded.")

writer = open_video_writer(OUTPUT_VIDEO_PATH, output_fps, width, height)

csv_rows = []
contours_output = {
    "video": str(VIDEO_PATH),
    "video_name": VIDEO_PATH.name,
    "annotation_file": str(ANNOTATION_PATH),
    "created_at": RUN_TIMESTAMP,
    "frames": {}
}

print("Initializing inference state...")

with torch.inference_mode():
    if DEVICE == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.autocast(device_type="cpu", enabled=False)

    with autocast_context:
        inference_state = predictor.init_state(video_path=str(VIDEO_PATH))

        print("Adding prompts from annotations JSON...")

        for obj_key, obj_data in annotation_data["objects"].items():
            obj_id = int(obj_key)

            for prompt in obj_data["prompts"]:
                frame_idx = int(prompt["frame"])
                points = np.array(prompt["points"], dtype=np.float32)
                labels = np.array(prompt["labels"], dtype=np.int32)

                if len(points) == 0:
                    continue

                print(
                    f"Adding prompts | object {obj_id} | "
                    f"frame {frame_idx} | points={len(points)}"
                )

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

        print("Running video propagation and exporting video...")

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            frame_idx = int(out_frame_idx)

            raw_frame = read_exact_frame(VIDEO_PATH, frame_idx)

            if raw_frame is None:
                print(f"Could not read frame {frame_idx}. Skipping.")
                continue

            frame_masks = {}

            for i, out_obj_id in enumerate(out_obj_ids):
                obj_id = int(out_obj_id)

                mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)

                if mask.ndim == 3:
                    mask = mask[0]

                frame_masks[obj_id] = mask

                stats = mask_to_bbox_centroid_area(mask)
                contours = mask_to_contours(mask)

                csv_rows.append({
                    "frame": frame_idx,
                    "object_id": obj_id,
                    "detected": stats["detected"],
                    "x_min": stats["x_min"],
                    "y_min": stats["y_min"],
                    "x_max": stats["x_max"],
                    "y_max": stats["y_max"],
                    "centroid_x": stats["centroid_x"],
                    "centroid_y": stats["centroid_y"],
                    "area": stats["area"],
                    "n_contours": len(contours)
                })

                if SAVE_CONTOURS_JSON:
                    frame_key = str(frame_idx)
                    obj_key = str(obj_id)

                    if frame_key not in contours_output["frames"]:
                        contours_output["frames"][frame_key] = {}

                    contours_output["frames"][frame_key][obj_key] = contours

                if SAVE_MASKS and frame_idx in sampled_frames:
                    mask_path = MASK_DIR / f"mask_frame_{frame_idx:06d}_obj_{obj_id}.png"
                    cv2.imwrite(str(mask_path), mask * 255)

            rendered_frame = draw_masks_on_frame(raw_frame, frame_masks, frame_idx, total_frames)
            writer.write(rendered_frame)

            if SAVE_PREVIEWS and frame_idx in sampled_frames:
                preview_path = PREVIEW_DIR / f"preview_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(preview_path), rendered_frame)
                print(f"Saved preview: {preview_path}")

            if frame_idx % 25 == 0:
                print(f"Exported frame {frame_idx}/{total_frames}")

writer.release()

print("Sorting and saving CSV...")

csv_rows = sorted(
    csv_rows,
    key=lambda row: (int(row["frame"]), int(row["object_id"]))
)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "frame",
        "object_id",
        "detected",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "centroid_x",
        "centroid_y",
        "area",
        "n_contours"
    ]

    writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
    writer_csv.writeheader()
    writer_csv.writerows(csv_rows)

if SAVE_CONTOURS_JSON:
    print("Saving contours JSON...")

    with open(CONTOURS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(contours_output, f, indent=2)

print(f"Final video saved: {OUTPUT_VIDEO_PATH}")
print(f"CSV saved: {CSV_PATH}")

if SAVE_CONTOURS_JSON:
    print(f"Contours JSON saved: {CONTOURS_JSON_PATH}")

print("Propagation export script completed.")