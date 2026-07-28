import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


"""
Propagate SAM2 masks through one annotated dynamic subvideo.

This script is the third step of the segmentation pipeline. It processes one
dynamic subvideo produced by exploration_extraction.py and annotated with
annotation_batch_subvideos.py.

The script:
    - reads the requested segment from data/subvideos/index.json;
    - locates the matching subvideo and annotation JSON;
    - validates all object prompts before starting inference;
    - extracts the subvideo frames to a temporary directory for SAM2;
    - optionally downsizes long subvideos during inference to reduce GPU memory;
    - adds all point and box prompts to the SAM2 video predictor;
    - propagates object masks through the complete subvideo;
    - restores masks to the original resolution when downscaling was used;
    - saves one binary mask per frame and object;
    - creates image overlays and a reconstructed overlay video;
    - exports one CSV row per expected object and per frame, including explicit
      detected=0 rows when a mask is absent;
    - stores metadata describing the propagation settings and outputs;
    - removes temporary frames even when an error interrupts processing.

Usage:
    python SAM_2/scripts/propagation_subvid.py video_task3_10

Force a new propagation even when complete outputs already exist:
    python SAM_2/scripts/propagation_subvid.py video_task3_10 --force
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = PROJECT_ROOT / "SAM_2"
SUBVIDEOS_DIR = PROJECT_ROOT / "data" / "subvideos"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
INDEX_PATH = SUBVIDEOS_DIR / "index.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "propagations"
TEMP_ROOT = OUTPUT_ROOT / "_tmp_frames"
CHECKPOINT_PATH = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OVERLAY_ALPHA = 0.35
REQUIRE_CUDA = True
SKIP_ALREADY_DONE = True
DOWNSCALE_FRAME_THRESHOLD = 700
MAX_SAM_WIDTH = 960

OBJECT_COLORS = {
    1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
    7: (128, 255, 0), 8: (255, 128, 0), 9: (128, 0, 255),
    10: (0, 128, 255), 11: (255, 0, 128), 12: (0, 255, 128),
    13: (128, 128, 255), 14: (255, 128, 128), 15: (128, 255, 255),
    16: (200, 200, 0), 17: (0, 200, 200), 18: (200, 0, 200),
}


def get_object_color(object_id: int):
    return OBJECT_COLORS.get(int(object_id), (200, 200, 200))


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file: {path}") from error


def load_index_entries() -> list:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    entries = load_json(INDEX_PATH)
    if not isinstance(entries, list):
        raise RuntimeError(f"Expected a list in index file: {INDEX_PATH}")
    return [entry for entry in entries if isinstance(entry, dict)]


def find_segment_entry(segment_name: str) -> dict:
    matches = [entry for entry in load_index_entries() if entry.get("name") == segment_name]
    if not matches:
        raise RuntimeError(f"Segment '{segment_name}' was not found in {INDEX_PATH}")
    segment = matches[0]
    if segment.get("type") != "video":
        raise RuntimeError(f"Segment '{segment_name}' is not declared as type 'video'.")
    for field_name in ("name", "start_frame", "end_frame"):
        if field_name not in segment:
            raise RuntimeError(f"Missing '{field_name}' in index entry for {segment_name}.")
    return segment


def resolve_video_path(segment: dict) -> Path:
    filename = segment.get("filename")
    if filename:
        indexed_path = SUBVIDEOS_DIR / filename
        if indexed_path.exists():
            return indexed_path
    fallback_path = SUBVIDEOS_DIR / f"{segment['name']}.mp4"
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Subvideo not found for segment: {segment['name']}")


def normalize_prompt(prompt: dict, total_frames: int, object_id: int):
    if not isinstance(prompt, dict):
        return None
    try:
        frame_idx = int(prompt.get("frame"))
    except (TypeError, ValueError):
        return None
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"Skipping out-of-range prompt: object {object_id}, frame {frame_idx}.")
        return None
    points = prompt.get("points", []) if isinstance(prompt.get("points", []), list) else []
    labels = prompt.get("labels", []) if isinstance(prompt.get("labels", []), list) else []
    valid_length = min(len(points), len(labels))
    normalized_points, normalized_labels = [], []
    for point, label in zip(points[:valid_length], labels[:valid_length]):
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                normalized_points.append([float(point[0]), float(point[1])])
                normalized_labels.append(int(label))
            except (TypeError, ValueError):
                pass
    box = prompt.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            box = [float(value) for value in box]
        except (TypeError, ValueError):
            box = None
    else:
        box = None
    if not normalized_points and box is None:
        return None
    return {"frame": frame_idx, "points": normalized_points, "labels": normalized_labels, "box": box}


def load_validated_prompts(annotation_path: Path, total_frames: int):
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    annotation = load_json(annotation_path)
    objects = annotation.get("objects", {})
    if not isinstance(objects, dict) or not objects:
        raise RuntimeError(f"No annotation objects found in: {annotation_path}")
    prompts_by_object = {}
    for object_key, object_data in objects.items():
        try:
            object_id = int(object_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(object_data, dict):
            continue
        raw_prompts = object_data.get("prompts", [])
        if not isinstance(raw_prompts, list):
            continue
        valid_prompts = []
        for raw_prompt in raw_prompts:
            prompt = normalize_prompt(raw_prompt, total_frames, object_id)
            if prompt is not None:
                valid_prompts.append(prompt)
        if valid_prompts:
            prompts_by_object[object_id] = sorted(valid_prompts, key=lambda p: p["frame"])
    if not prompts_by_object:
        raise RuntimeError(f"No valid prompts found in annotation: {annotation_path}")
    return prompts_by_object


def inspect_video(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open subvideo: {video_path}")
    info = {
        "n_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "original_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "original_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if info["n_frames"] <= 0 or info["fps"] <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return info


def extract_frames_to_temp(video_path: Path, temp_dir: Path) -> dict:
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_video(video_path)
    use_downscale = info["n_frames"] > DOWNSCALE_FRAME_THRESHOLD and info["original_width"] > MAX_SAM_WIDTH
    if use_downscale:
        scale = MAX_SAM_WIDTH / info["original_width"]
        sam_width = int(info["original_width"] * scale)
        sam_height = int(info["original_height"] * scale)
    else:
        scale = 1.0
        sam_width = info["original_width"]
        sam_height = info["original_height"]
    capture = cv2.VideoCapture(str(video_path))
    extracted = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (sam_width, sam_height), interpolation=cv2.INTER_AREA)
        frame_path = temp_dir / f"{extracted:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"Could not write temporary frame: {frame_path}")
        extracted += 1
    capture.release()
    if extracted == 0:
        raise RuntimeError(f"No frame extracted from subvideo: {video_path}")
    return {**info, "n_frames": extracted, "sam_width": sam_width, "sam_height": sam_height,
            "scale": scale, "downscaled": scale != 1.0}


def overlay_mask(frame, mask, color, alpha=OVERLAY_ALPHA):
    output = frame.copy()
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return output
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[:, :] = color
    output[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def mask_statistics(mask):
    if mask is None:
        return {"detected": 0, "x_min": -1, "y_min": -1, "x_max": -1, "y_max": -1,
                "centroid_x": -1, "centroid_y": -1, "area": 0}
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"detected": 0, "x_min": -1, "y_min": -1, "x_max": -1, "y_max": -1,
                "centroid_x": -1, "centroid_y": -1, "area": 0}
    return {"detected": 1, "x_min": int(xs.min()), "y_min": int(ys.min()),
            "x_max": int(xs.max()), "y_max": int(ys.max()),
            "centroid_x": float(xs.mean()), "centroid_y": float(ys.mean()), "area": int(len(xs))}


def output_paths(segment_name: str) -> dict:
    output_dir = OUTPUT_ROOT / segment_name
    return {
        "output_dir": output_dir,
        "masks_dir": output_dir / "masks",
        "overlays_dir": output_dir / "overlays",
        "metadata": output_dir / "metadata.json",
        "overlay_video": output_dir / f"{segment_name}_overlay.mp4",
        "csv": output_dir / f"{segment_name}_masks.csv",
    }


def outputs_are_complete(paths: dict) -> bool:
    return all(path.exists() for path in (paths["metadata"], paths["overlay_video"], paths["csv"]))


def write_csv(csv_path: Path, rows: list):
    fieldnames = ["segment_name", "local_frame", "global_frame", "object_id", "detected",
                  "x_min", "y_min", "x_max", "y_max", "centroid_x", "centroid_y", "area", "mask_path"]
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_prompts(predictor, inference_state, prompts_by_object, scale: float):
    count = 0
    for object_id, prompts in prompts_by_object.items():
        for prompt in prompts:
            points, labels, box = prompt["points"], prompt["labels"], prompt["box"]
            if scale != 1.0:
                points = [[p[0] * scale, p[1] * scale] for p in points]
                if box is not None:
                    box = [value * scale for value in box]
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_labels = np.array(labels, dtype=np.int32) if labels else None
            box_array = np.array(box, dtype=np.float32) if box is not None else None
            print(f"  object {object_id} | frame {prompt['frame']}")
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt["frame"],
                obj_id=object_id,
                points=point_coords,
                labels=point_labels,
                box=box_array,
            )
            count += 1
    return count


def run_propagation(predictor, inference_state, video_info):
    masks_by_frame = defaultdict(dict)
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        frame_idx = int(frame_idx)
        for i, object_id in enumerate(object_ids):
            mask = (mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
            mask = np.squeeze(mask)
            if video_info["downscaled"]:
                mask = cv2.resize(mask, (video_info["original_width"], video_info["original_height"]),
                                  interpolation=cv2.INTER_NEAREST)
            masks_by_frame[frame_idx][int(object_id)] = mask
        if frame_idx % 100 == 0:
            print(f"  propagated frame {frame_idx}/{video_info['n_frames'] - 1}")
    return masks_by_frame


def save_outputs(segment, video_path, annotation_path, paths, video_info,
                 prompts_by_object, masks_by_frame, device):
    segment_name = segment["name"]
    object_ids = sorted(prompts_by_object)
    writer = cv2.VideoWriter(
        str(paths["overlay_video"]), cv2.VideoWriter_fourcc(*"mp4v"), video_info["fps"],
        (video_info["original_width"], video_info["original_height"])
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {paths['overlay_video']}")
    capture = cv2.VideoCapture(str(video_path))
    csv_rows = []
    origin_start = int(segment["start_frame"])
    for local_frame in range(video_info["n_frames"]):
        success, frame = capture.read()
        if not success:
            capture.release(); writer.release()
            raise RuntimeError(f"Could not read frame {local_frame} from {video_path}")
        overlay = frame.copy()
        frame_masks = masks_by_frame.get(local_frame, {})
        for object_id in object_ids:
            mask = frame_masks.get(object_id)
            mask_path_text = ""
            if mask is not None and np.any(mask > 0):
                object_dir = paths["masks_dir"] / f"obj_{object_id:02d}"
                object_dir.mkdir(parents=True, exist_ok=True)
                mask_path = object_dir / f"frame_{local_frame:06d}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                mask_path_text = str(mask_path)
                overlay = overlay_mask(overlay, mask, get_object_color(object_id))
            csv_rows.append({
                "segment_name": segment_name,
                "local_frame": local_frame,
                "global_frame": origin_start + local_frame,
                "object_id": object_id,
                **mask_statistics(mask),
                "mask_path": mask_path_text,
            })
        overlay_path = paths["overlays_dir"] / f"frame_{local_frame:06d}.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writer.write(overlay)
    capture.release(); writer.release()
    write_csv(paths["csv"], csv_rows)
    metadata = {
        "segment_name": segment_name,
        "subvideo_path": str(video_path),
        "annotation_path": str(annotation_path),
        "origin_start_frame": int(segment["start_frame"]),
        "origin_end_frame": int(segment["end_frame"]),
        "n_frames": video_info["n_frames"],
        "fps": video_info["fps"],
        "original_width": video_info["original_width"],
        "original_height": video_info["original_height"],
        "sam_width": video_info["sam_width"],
        "sam_height": video_info["sam_height"],
        "downscaled": video_info["downscaled"],
        "scale": video_info["scale"],
        "downscale_frame_threshold": DOWNSCALE_FRAME_THRESHOLD,
        "max_sam_width": MAX_SAM_WIDTH,
        "device": device,
        "model_config": MODEL_CONFIG,
        "checkpoint": str(CHECKPOINT_PATH),
        "object_ids": object_ids,
        "prompt_count": sum(len(v) for v in prompts_by_object.values()),
        "overlay_alpha": OVERLAY_ALPHA,
        "overlay_video": str(paths["overlay_video"]),
        "csv": str(paths["csv"]),
        "output_complete": True,
        "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def propagate_segment(segment_name: str, predictor, device: str, force: bool = False) -> str:
    segment = find_segment_entry(segment_name)
    video_path = resolve_video_path(segment)
    annotation_path = ANNOTATIONS_DIR / f"{segment_name}.json"
    paths = output_paths(segment_name)
    temp_dir = TEMP_ROOT / segment_name
    if SKIP_ALREADY_DONE and not force and outputs_are_complete(paths):
        print(f"[SKIP] {segment_name} already has complete outputs.")
        return "skipped"
    if paths["output_dir"].exists():
        shutil.rmtree(paths["output_dir"])
    paths["masks_dir"].mkdir(parents=True, exist_ok=True)
    paths["overlays_dir"].mkdir(parents=True, exist_ok=True)
    inference_state = None
    try:
        print("=" * 70)
        print(f"SEGMENT: {segment_name}")
        print("=" * 70)
        print(f"Subvideo   : {video_path}")
        print(f"Annotation : {annotation_path}")
        print(f"Output     : {paths['output_dir']}")
        video_info = extract_frames_to_temp(video_path, temp_dir)
        prompts_by_object = load_validated_prompts(annotation_path, video_info["n_frames"])
        print(f"Frames: {video_info['n_frames']} | FPS: {video_info['fps']:.4f}")
        print(f"SAM2 size: {video_info['sam_width']}x{video_info['sam_height']} | downscaled={video_info['downscaled']}")
        inference_state = predictor.init_state(
            video_path=str(temp_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                add_prompts(predictor, inference_state, prompts_by_object, video_info["scale"])
                masks_by_frame = run_propagation(predictor, inference_state, video_info)
        save_outputs(segment, video_path, annotation_path, paths, video_info,
                     prompts_by_object, masks_by_frame, device)
        print(f"DONE: {segment_name}")
        return "done"
    except Exception:
        if paths["metadata"].exists():
            paths["metadata"].unlink()
        raise
    finally:
        if inference_state is not None:
            reset_method = getattr(predictor, "reset_state", None)
            if callable(reset_method):
                try:
                    reset_method(inference_state)
                except Exception:
                    pass
            del inference_state
        shutil.rmtree(temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Propagate SAM2 masks through one annotated dynamic subvideo.")
    parser.add_argument("segment_name", help="Logical segment name, for example video_task3_10.")
    parser.add_argument("--force", action="store_true", help="Recompute even when complete outputs exist.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if REQUIRE_CUDA and device != "cuda":
        raise RuntimeError("CUDA is required but is not available.")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {CHECKPOINT_PATH}")
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)
    print("=" * 70)
    print("SAM2 PROPAGATION - SINGLE SUBVIDEO")
    print("=" * 70)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT_PATH), device=device)
    propagate_segment(args.segment_name, predictor, device, force=args.force)


if __name__ == "__main__":
    main()