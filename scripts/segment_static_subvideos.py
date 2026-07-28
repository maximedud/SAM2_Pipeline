#!/usr/bin/env python3
"""
Segment all static subvideos using SAM2.

For every annotation JSON whose type is "static", the script:

1. Opens the associated static subvideo.
2. Extracts the annotated frame.
3. Applies all point and box prompts with SAM2ImagePredictor.
4. Saves one binary mask per object.
5. Saves the original frame and a combined overlay.
6. Writes a metadata.json file describing the generated outputs.

Static segments are processed independently because they contain a single
annotated reference frame and therefore do not require temporal propagation.

The generated masks can later be reused during reconstruction without
running SAM2 again.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = PROJECT_ROOT / "SAM_2"

ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SUBVIDEOS_DIR = PROJECT_ROOT / "data" / "subvideos"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "static_segmentations"

CHECKPOINT_PATH = (
    SAM2_ROOT
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# When True, existing outputs containing metadata.json are skipped.
SKIP_ALREADY_DONE = True

# Transparency of each colored mask in the final overlay.
MASK_ALPHA = 0.45


# ---------------------------------------------------------------------------
# SAM2 imports
# ---------------------------------------------------------------------------

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_json(json_path: Path) -> dict[str, Any]:
    """Load and return a JSON annotation file."""

    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON file: {json_path}"
        ) from error



def canonical_segment_name(name: str) -> str:
    """
    Return the canonical segment identifier.

    Legacy annotations may contain a final global-frame suffix such as
    ``_018200_018567``. Output folders intentionally omit this suffix because
    frame boundaries are already stored in the dataset index.
    """

    return re.sub(r"_\d{6}_\d{6}$", "", name.strip())


def resolve_subvideo_path(
    annotation: dict[str, Any],
    annotation_path: Path,
) -> Path:
    """
    Resolve the subvideo path stored in an annotation JSON.

    Old annotation files may contain paths from another machine or mount
    point. If the stored path no longer exists, the script searches for a
    video having the same filename in data/subvideos.
    """

    raw_path = annotation.get("subvideo_path")

    if not raw_path:
        raise ValueError(
            f"No 'subvideo_path' found in {annotation_path}"
        )

    stored_path = Path(raw_path)

    if stored_path.exists():
        return stored_path

    fallback_path = PROJECT_ROOT / "data" / "subvideos" / stored_path.name

    if fallback_path.exists():
        return fallback_path

    # Additional fallback using the JSON stem.
    possible_extensions = [".mp4", ".avi", ".mov", ".mkv"]

    for extension in possible_extensions:
        candidate = (
            PROJECT_ROOT
            / "data"
            / "subvideos"
            / f"{annotation_path.stem}{extension}"
        )

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to find the static subvideo.\n"
        f"Stored path: {stored_path}\n"
        f"Fallback directory: {PROJECT_ROOT / 'data' / 'subvideos'}"
    )


def extract_video_frame(
    video_path: Path,
    frame_index: int,
) -> np.ndarray:
    """
    Extract one BGR frame from a video using an exact frame index.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_index < 0 or frame_index >= total_frames:
        capture.release()
        raise IndexError(
            f"Frame {frame_index} is outside video range "
            f"[0, {total_frames - 1}] for {video_path}"
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError(
            f"Unable to read frame {frame_index} from {video_path}"
        )

    return frame


def object_color(object_id: int) -> tuple[int, int, int]:
    """
    Return a deterministic BGR color for an object identifier.

    The palette is deliberately stable so that the same object ID receives
    the same color in every output.
    """

    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (255, 80, 255),
        (80, 255, 255),
        (255, 160, 80),
        (160, 80, 255),
        (80, 160, 255),
        (160, 255, 80),
        (255, 80, 160),
        (80, 255, 160),
    ]

    return palette[(object_id - 1) % len(palette)]


def save_binary_mask(
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a boolean mask as an 8-bit PNG image."""

    binary_mask = (mask.astype(np.uint8) * 255)

    success = cv2.imwrite(str(output_path), binary_mask)

    if not success:
        raise RuntimeError(f"Unable to save mask: {output_path}")


def add_mask_to_overlay(
    overlay: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """
    Blend one colored object mask into an existing BGR overlay image.
    """

    result = overlay.copy()
    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        return result

    color_array = np.asarray(color, dtype=np.float32)

    original_pixels = result[mask_bool].astype(np.float32)

    blended_pixels = (
        (1.0 - alpha) * original_pixels
        + alpha * color_array
    )

    result[mask_bool] = np.clip(
        blended_pixels,
        0,
        255,
    ).astype(np.uint8)

    return result


def draw_object_label(
    image: np.ndarray,
    mask: np.ndarray,
    object_id: int,
    color: tuple[int, int, int],
) -> None:
    """
    Draw the object identifier close to the mask centroid.
    """

    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return

    center_y, center_x = coordinates.mean(axis=0).astype(int)

    label = f"obj {object_id}"

    cv2.putText(
        image,
        label,
        (int(center_x), int(center_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def collect_object_prompts(
    object_data: dict[str, Any],
    annotated_frame: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Collect prompts associated with the selected annotated frame.

    Multiple prompt entries for the same object and frame are merged.
    """

    all_points: list[list[float]] = []
    all_labels: list[int] = []
    selected_box: np.ndarray | None = None

    prompts = object_data.get("prompts", [])

    for prompt in prompts:
        prompt_frame = int(prompt.get("frame", annotated_frame))

        if prompt_frame != annotated_frame:
            continue

        points = prompt.get("points", [])
        labels = prompt.get("labels", [])

        if len(points) != len(labels):
            raise ValueError(
                "The number of points does not match the number of labels."
            )

        all_points.extend(points)
        all_labels.extend(labels)

        box = prompt.get("box")

        if box is not None:
            selected_box = np.asarray(box, dtype=np.float32)

    point_coords: np.ndarray | None = None
    point_labels: np.ndarray | None = None

    if all_points:
        point_coords = np.asarray(all_points, dtype=np.float32)
        point_labels = np.asarray(all_labels, dtype=np.int32)

    return point_coords, point_labels, selected_box


# ---------------------------------------------------------------------------
# SAM2 inference
# ---------------------------------------------------------------------------

def build_predictor() -> SAM2ImagePredictor:
    """Load the SAM2 image predictor on the best available device."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}")

    sam2_model = build_sam2(
        MODEL_CONFIG,
        str(CHECKPOINT_PATH),
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)

    return predictor


def predict_object_mask(
    predictor: SAM2ImagePredictor,
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    box: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """
    Predict one object mask and return the highest-scoring candidate.
    """

    if point_coords is None and box is None:
        raise ValueError(
            "The object has neither point prompts nor a bounding box."
        )

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

    best_index = int(np.argmax(scores))

    best_mask = masks[best_index].astype(bool)
    best_score = float(scores[best_index])

    return best_mask, best_score


# ---------------------------------------------------------------------------
# Static annotation processing
# ---------------------------------------------------------------------------

def process_static_annotation(
    predictor: SAM2ImagePredictor,
    annotation_path: Path,
) -> None:
    """
    Segment one static subvideo annotation and save all generated outputs.
    """

    annotation = load_json(annotation_path)

    if annotation.get("type") != "static":
        return

    raw_segment_name = str(
        annotation.get("name", annotation_path.stem)
    ).strip()
    segment_name = canonical_segment_name(raw_segment_name)

    output_dir = OUTPUT_ROOT / segment_name
    masks_dir = output_dir / "masks"
    metadata_path = output_dir / "metadata.json"

    overlay_path = output_dir / "overlay.png"

    if (
        SKIP_ALREADY_DONE
        and metadata_path.exists()
        and overlay_path.exists()
    ):
        print(f"[SKIP] {segment_name}")
        return

    video_path = resolve_subvideo_path(
        annotation,
        annotation_path,
    )

    annotated_frame = int(
        annotation.get("annotated_frame", 0)
    )

    print()
    print("=" * 70)
    print(f"[STATIC] {segment_name}")
    print(f"[VIDEO]  {video_path}")
    print(f"[FRAME]  {annotated_frame}")
    print("=" * 70)

    bgr_frame = extract_video_frame(
        video_path,
        annotated_frame,
    )

    if bgr_frame.size == 0:
        raise RuntimeError(
            f"Extracted frame is empty: {video_path}, frame {annotated_frame}"
        )

    rgb_frame = cv2.cvtColor(
        bgr_frame,
        cv2.COLOR_BGR2RGB,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    raw_frame_path = output_dir / "raw.png"

    if not cv2.imwrite(str(raw_frame_path), bgr_frame):
        raise RuntimeError(
            f"Unable to save original frame: {raw_frame_path}"
        )

    predictor.set_image(rgb_frame)

    overlay = bgr_frame.copy()

    object_metadata: dict[str, Any] = {}

    objects = annotation.get("objects", {})

    if not objects:
        print(f"[WARNING] No objects found in {annotation_path.name}")

    for object_id_text, object_data in objects.items():
        object_id = int(object_id_text)

        point_coords, point_labels, box = collect_object_prompts(
            object_data,
            annotated_frame,
        )

        try:
            mask, score = predict_object_mask(
                predictor=predictor,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
            )
        except ValueError as error:
            print(
                f"[WARNING] Object {object_id} skipped: {error}"
            )
            continue

        mask_path = masks_dir / f"object_{object_id}.png"

        save_binary_mask(mask, mask_path)

        color = object_color(object_id)

        overlay = add_mask_to_overlay(
            overlay=overlay,
            mask=mask,
            color=color,
            alpha=MASK_ALPHA,
        )

        draw_object_label(
            image=overlay,
            mask=mask,
            object_id=object_id,
            color=color,
        )

        object_metadata[str(object_id)] = {
            "label": object_data.get(
                "label",
                f"object_{object_id}",
            ),
            "score": score,
            "area_pixels": int(mask.sum()),
            "mask_path": str(mask_path),
            "point_count": (
                int(len(point_coords))
                if point_coords is not None
                else 0
            ),
            "positive_point_count": (
                int(np.sum(point_labels == 1))
                if point_labels is not None
                else 0
            ),
            "negative_point_count": (
                int(np.sum(point_labels == 0))
                if point_labels is not None
                else 0
            ),
            "box": (
                box.tolist()
                if box is not None
                else None
            ),
        }

        print(
            f"[OBJECT {object_id}] "
            f"score={score:.4f} "
            f"area={int(mask.sum())}"
        )

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(
            f"Unable to save overlay: {overlay_path}"
        )

    height, width = bgr_frame.shape[:2]

    output_metadata = {
        "segment_name": segment_name,
        "original_annotation_name": raw_segment_name,
        "type": "static",
        "annotation_path": str(annotation_path),
        "subvideo_path": str(video_path),
        "annotated_frame": annotated_frame,
        "origin_start_frame": annotation.get(
            "origin_start_frame"
        ),
        "origin_end_frame": annotation.get(
            "origin_end_frame"
        ),
        "fps": annotation.get("fps"),
        "width": width,
        "height": height,
        "raw_frame": str(raw_frame_path),
        "overlay_image": str(overlay_path),
        "masks_directory": str(masks_dir),
        "objects": object_metadata,
        "model_config": MODEL_CONFIG,
        "checkpoint": str(CHECKPOINT_PATH),
        "mask_alpha": MASK_ALPHA,
        "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "output_complete": True,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[DONE] Overlay:  {overlay_path}")
    print(f"[DONE] Metadata: {metadata_path}")


def main() -> None:
    """Process every static JSON annotation in the annotation directory."""

    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Annotations directory not found: {ANNOTATIONS_DIR}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(
        ANNOTATIONS_DIR.glob("*.json")
    )

    static_annotation_paths: list[Path] = []

    for annotation_path in annotation_paths:
        try:
            annotation = load_json(annotation_path)
        except (RuntimeError, OSError) as error:
            print(
                f"[WARNING] Invalid JSON ignored: "
                f"{annotation_path.name}: {error}"
            )
            continue

        if annotation.get("type") == "static":
            static_annotation_paths.append(annotation_path)

    print(
        f"[INFO] {len(static_annotation_paths)} "
        "static annotation file(s) found."
    )

    if not static_annotation_paths:
        return

    predictor = build_predictor()

    successful_count = 0
    failed_count = 0

    for annotation_path in static_annotation_paths:
        try:
            process_static_annotation(
                predictor,
                annotation_path,
            )
            successful_count += 1

        except Exception as error:
            failed_count += 1
            print()
            print(
                f"[ERROR] Failed to process "
                f"{annotation_path.name}"
            )
            print(f"[ERROR] {type(error).__name__}: {error}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("[SUMMARY]")
    print(f"Successful: {successful_count}")
    print(f"Failed:     {failed_count}")
    print(f"Output:     {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()


