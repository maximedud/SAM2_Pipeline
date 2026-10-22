"""
SAM2 video propagation - Chunked processing.

Purpose:
    Propagate SAM2 annotations across reference MASC subvideos
    using overlapping windows to limit GPU memory consumption.

Inputs:
    data/subvideos/*.mp4
    data/subvideos/index.json
    data/annotations/<segment_name>.json
    SAM_2/checkpoints/sam2.1_hiera_small.pt

Outputs:
    data/propagations/<segment_name>/
        masks/obj_XX/frame_XXXXXX.png
        overlays/frame_XXXXXX.jpg
        <segment_name>_overlay.mp4
        <segment_name>_masks.csv
        metadata.json

Notes:
    - Requires a CUDA-compatible GPU.
    - Already completed segments are skipped when metadata.json exists.
    - Uses overlapping windows to propagate long videos.
"""


import cv2
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from sam2.build_sam import build_sam2_video_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBVIDEOS_DIR = PROJECT_ROOT / "data" / "subvideos"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
INDEX_PATH = SUBVIDEOS_DIR / "index.json"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "propagations"
TEMP_ROOT = OUTPUT_ROOT / "_tmp_frames"

SAM2_ROOT = PROJECT_ROOT / "SAM_2"
CHECKPOINT_PATH = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

ALPHA = 0.35
REQUIRE_CUDA = True
SKIP_ALREADY_DONE = True

WINDOW_SIZE = 300
STRIDE = 250
MAX_WIDTH = 960
USE_CPU_OFFLOAD = False

OBJECT_COLORS = {
    1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255),
    7: (128, 255, 0), 8: (255, 128, 0), 9: (128, 0, 255),
    10: (0, 128, 255), 11: (255, 0, 128), 12: (0, 255, 128),
    13: (128, 128, 255), 14: (255, 128, 128), 15: (128, 255, 255),
    16: (200, 200, 0), 17: (0, 200, 200), 18: (200, 0, 200),
}


def get_color(obj_id: int):
    return OBJECT_COLORS.get(int(obj_id), (200, 200, 200))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_video_entries():
    index = load_json(INDEX_PATH)
    videos = [e for e in index if e.get("type") == "video"]
    return sorted(videos, key=lambda e: int(e["start_frame"]))


def get_prompts(annotation: dict):
    prompts_by_obj = {}
    for obj_key, obj_data in annotation.get("objects", {}).items():
        prompts_by_obj[int(obj_key)] = obj_data.get("prompts", [])
    return prompts_by_obj


def overlay_mask(frame, mask, color, alpha=ALPHA):
    out = frame.copy()
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[:, :] = color
    mask_bool = mask.astype(bool)

    if mask_bool.any():
        out[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

    return out


def mask_stats(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {
            "detected": 0, "x_min": -1, "y_min": -1, "x_max": -1, "y_max": -1,
            "centroid_x": -1, "centroid_y": -1, "area": 0,
        }
    return {
        "detected": 1,
        "x_min": int(xs.min()), "y_min": int(ys.min()),
        "x_max": int(xs.max()), "y_max": int(ys.max()),
        "centroid_x": float(xs.mean()), "centroid_y": float(ys.mean()),
        "area": int(len(xs)),
    }


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def select_last_available_masks(masks_by_frame, start_frame, end_frame):
    for frame_idx in range(end_frame, start_frame - 1, -1):
        masks = masks_by_frame.get(frame_idx, {})
        if masks:
            return masks
    return {}


def extract_window_frames(video_path: Path, temp_dir: Path, window_start: int, window_end: int):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_width > MAX_WIDTH:
        scale = MAX_WIDTH / original_width
        sam_width = int(original_width * scale)
        sam_height = int(original_height * scale)
    else:
        scale = 1.0
        sam_width = original_width
        sam_height = original_height

    cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)

    local_idx = 0
    frame_idx = window_start
    while frame_idx <= window_end:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (sam_width, sam_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(temp_dir / f"{local_idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        local_idx += 1
        frame_idx += 1

    cap.release()

    if local_idx == 0:
        raise RuntimeError(f"No frame extracted for window {window_start}-{window_end}")

    return {
        "fps": fps, "total_frames": total_frames, "window_n_frames": local_idx,
        "original_width": original_width, "original_height": original_height,
        "sam_width": sam_width, "sam_height": sam_height,
        "scale": scale, "downscaled": scale != 1.0,
    }


def add_human_prompts_to_window(predictor, inference_state, prompts_by_obj, window_start, window_end, scale):
    added = 0

    for obj_id, prompts in prompts_by_obj.items():
        for prompt in prompts:
            frame_idx = int(prompt["frame"])
            if frame_idx < window_start or frame_idx > window_end:
                continue

            local_frame = frame_idx - window_start
            points = prompt.get("points", [])
            labels = prompt.get("labels", [])
            box = prompt.get("box", None)

            if scale != 1.0:
                points = [[p[0] * scale, p[1] * scale] for p in points]
                if box is not None:
                    box = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]

            point_coords = np.array(points, dtype=np.float32) if len(points) > 0 else None
            point_labels = np.array(labels, dtype=np.int32) if len(labels) > 0 else None
            box_array = np.array(box, dtype=np.float32) if box is not None else None

            if point_coords is None and box_array is None:
                print(f"  SKIP empty human prompt obj {obj_id} | global {frame_idx}")
                continue

            print(f"  human prompt obj {obj_id} | global {frame_idx} | local {local_frame}")

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=local_frame,
                obj_id=obj_id,
                points=point_coords,
                labels=point_labels,
                box=box_array,
            )
            added += 1

    return added


def add_auto_boxes_to_window(predictor, inference_state, previous_masks, scale):
    added = 0

    for obj_id, mask in previous_masks.items():
        box = mask_to_box(mask)
        if box is None:
            continue

        if scale != 1.0:
            box = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]

        print(f"  auto box obj {obj_id} | local 0")

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=None,
            labels=None,
            box=np.array(box, dtype=np.float32),
        )
        added += 1

    return added


def store_mask(out_frame_idx, out_obj_ids, out_mask_logits, window_start, info, window_masks):
    segment_frame_idx = window_start + int(out_frame_idx)

    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
        mask = np.squeeze(mask)

        if info["downscaled"]:
            mask = cv2.resize(
                mask,
                (info["original_width"], info["original_height"]),
                interpolation=cv2.INTER_NEAREST,
            )

        window_masks[segment_frame_idx][int(obj_id)] = mask


def run_forward_and_backward(predictor, inference_state, window_start, info, window_masks):
    print("  propagating forward...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        store_mask(out_frame_idx, out_obj_ids, out_mask_logits, window_start, info, window_masks)

    print("  propagating backward...")
    try:
        iterator = predictor.propagate_in_video(inference_state, reverse=True)
    except TypeError:
        print("  WARNING: reverse=True unsupported; backward frames may stay empty.")
        return

    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
        store_mask(out_frame_idx, out_obj_ids, out_mask_logits, window_start, info, window_masks)


def propagate_one_window(
    predictor,
    video_path,
    segment_name,
    prompts_by_obj,
    window_start,
    window_end,
    keep_start,
    keep_end,
    previous_masks,
):
    temp_dir = TEMP_ROOT / segment_name / f"window_{window_start:06d}_{window_end:06d}"

    print("-" * 70)
    print(f"WINDOW {window_start} -> {window_end} | KEEP {keep_start} -> {keep_end}")
    print("-" * 70)

    info = extract_window_frames(video_path, temp_dir, window_start, window_end)

    print(f"Window frames : {info['window_n_frames']}")
    print(f"Original size : {info['original_width']}x{info['original_height']}")
    print(f"SAM2 size     : {info['sam_width']}x{info['sam_height']}")
    print(f"Downscaled    : {info['downscaled']}")

    if USE_CPU_OFFLOAD:
        inference_state = predictor.init_state(
            video_path=str(temp_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
    else:
        inference_state = predictor.init_state(video_path=str(temp_dir))

    window_masks = defaultdict(dict)

    with torch.inference_mode():
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with autocast_context:
            n_auto = add_auto_boxes_to_window(predictor, inference_state, previous_masks, info["scale"])
            n_human = add_human_prompts_to_window(
                predictor, inference_state, prompts_by_obj, window_start, window_end, info["scale"]
            )

            if n_auto == 0 and n_human == 0:
                print("  no prompt available in this window -> no masks produced")
                shutil.rmtree(temp_dir)
                return {}, {}

            run_forward_and_backward(predictor, inference_state, window_start, info, window_masks)

    kept_masks = defaultdict(dict)
    for frame_idx, obj_masks in window_masks.items():
        if keep_start <= frame_idx <= keep_end:
            for obj_id, mask in obj_masks.items():
                kept_masks[frame_idx][obj_id] = mask

    next_previous_masks = select_last_available_masks(kept_masks, keep_start, keep_end)

    shutil.rmtree(temp_dir)
    torch.cuda.empty_cache()

    return kept_masks, next_previous_masks


def propagate_one_segment(segment, predictor):
    segment_name = segment["name"]
    video_path = SUBVIDEOS_DIR / f"{segment_name}.mp4"
    annotation_path = ANNOTATIONS_DIR / f"{segment_name}.json"

    output_dir = OUTPUT_ROOT / segment_name
    masks_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    metadata_path = output_dir / "metadata.json"

    if SKIP_ALREADY_DONE and metadata_path.exists():
        print(f"[SKIP] {segment_name} already done.")
        return "skipped"

    if not video_path.exists():
        print(f"[MISSING VIDEO] {video_path}")
        return "missing_video"

    if not annotation_path.exists():
        print(f"[MISSING ANNOTATION] {annotation_path}")
        return "missing_annotation"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SEGMENT: {segment_name}")
    print("=" * 70)
    print(f"Subvideo   : {video_path}")
    print(f"Annotation : {annotation_path}")
    print(f"Output     : {output_dir}")

    annotation = load_json(annotation_path)
    prompts_by_obj = get_prompts(annotation)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Frames: {total_frames}")
    print(f"Size  : {original_width}x{original_height}")
    print(f"FPS   : {fps:.4f}")

    all_masks = defaultdict(dict)
    previous_masks = {}

    keep_start = 0
    while keep_start < total_frames:
        keep_end = min(keep_start + STRIDE - 1, total_frames - 1)
        window_start = keep_start
        window_end = min(window_start + WINDOW_SIZE - 1, total_frames - 1)

        kept_masks, previous_masks = propagate_one_window(
            predictor=predictor,
            video_path=video_path,
            segment_name=segment_name,
            prompts_by_obj=prompts_by_obj,
            window_start=window_start,
            window_end=window_end,
            keep_start=keep_start,
            keep_end=keep_end,
            previous_masks=previous_masks,
        )

        for frame_idx, obj_masks in kept_masks.items():
            for obj_id, mask in obj_masks.items():
                all_masks[frame_idx][obj_id] = mask

        keep_start += STRIDE

    print("\nSaving masks, overlays and CSV...")

    overlay_video_path = output_dir / f"{segment_name}_overlay.mp4"
    csv_path = output_dir / f"{segment_name}_masks.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps, (original_width, original_height))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    csv_rows = []
    origin_start = int(segment["start_frame"])
    local_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        for obj_id, mask in all_masks.get(local_idx, {}).items():
            obj_dir = masks_dir / f"obj_{obj_id:02d}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            mask_path = obj_dir / f"frame_{local_idx:06d}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

            overlay = overlay_mask(overlay, mask, get_color(obj_id))
            stats = mask_stats(mask)

            csv_rows.append({
                "segment_name": segment_name,
                "local_frame": local_idx,
                "global_frame": origin_start + local_idx,
                "object_id": obj_id,
                **stats,
                "mask_path": str(mask_path),
            })

        overlay_img_path = overlay_dir / f"frame_{local_idx:06d}.jpg"
        cv2.imwrite(str(overlay_img_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writer.write(overlay)
        local_idx += 1

    cap.release()
    writer.release()

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "segment_name,local_frame,global_frame,object_id,detected,"
            "x_min,y_min,x_max,y_max,centroid_x,centroid_y,area,mask_path\n"
        )
        for row in csv_rows:
            f.write(
                f"{row['segment_name']},{row['local_frame']},{row['global_frame']},"
                f"{row['object_id']},{row['detected']},{row['x_min']},{row['y_min']},"
                f"{row['x_max']},{row['y_max']},{row['centroid_x']},{row['centroid_y']},"
                f"{row['area']},{row['mask_path']}\n"
            )

    metadata = {
        "segment_name": segment_name,
        "subvideo_path": str(video_path),
        "annotation_path": str(annotation_path),
        "origin_start_frame": int(segment["start_frame"]),
        "origin_end_frame": int(segment["end_frame"]),
        "n_frames": total_frames,
        "fps": fps,
        "width": original_width,
        "height": original_height,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_width": MAX_WIDTH,
        "use_cpu_offload": USE_CPU_OFFLOAD,
        "device": "cuda",
        "overlay_video": str(overlay_video_path),
        "csv": str(csv_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"DONE: {segment_name}")
    print(f"Overlay video : {overlay_video_path}")
    print(f"Masks dir     : {masks_dir}")
    print(f"CSV           : {csv_path}")
    return "done"


def main():
    print("=" * 70)
    print("SAM2 PROPAGATION - CHUNKED V2")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if REQUIRE_CUDA and device != "cuda":
        raise RuntimeError("CUDA is not available.")

    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)

    print(f"Device: {device}")
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"STRIDE     : {STRIDE}")
    print(f"MAX_WIDTH  : {MAX_WIDTH}")

    videos = get_video_entries()

    print(f"Video segments found: {len(videos)}")
    for v in videos:
        print(f"  - {v['name']} | {v['start_frame']} -> {v['end_frame']}")

    print("\nLoading SAM2 video predictor once...")
    predictor = build_sam2_video_predictor(MODEL_CFG, str(CHECKPOINT_PATH), device=device)

    done = 0
    skipped = 0
    missing = []
    failed = []

    for i, segment in enumerate(videos, start=1):
        print("\n" + "#" * 70)
        print(f"[{i}/{len(videos)}] {segment['name']}")
        print("#" * 70)

        try:
            status = propagate_one_segment(segment, predictor)
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            elif status in ["missing_video", "missing_annotation"]:
                missing.append(segment["name"])

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {segment['name']}: {repr(e)}")
            failed.append(segment["name"])
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] {segment['name']}: {repr(e)}")
            failed.append(segment["name"])
            torch.cuda.empty_cache()

    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)

    print("=" * 70)
    print("ALL DONE")
    print(f"Done    : {done}")
    print(f"Skipped : {skipped}")
    print(f"Missing : {len(missing)}")
    print(f"Failed  : {len(failed)}")

    if missing:
        print("Missing segments:")
        for name in missing:
            print(f"  - {name}")

    if failed:
        print("Failed segments:")
        for name in failed:
            print(f"  - {name}")

    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()