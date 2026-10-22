"""
Frame-based video exploration and exact subvideo extraction tool.

This script is the first step of the segmentation pipeline. It converts the
complete participant video into a sequence of numbered JPEG frames, which then
becomes the reference used by the rest of the project. Working from numbered
frames avoids the frame-count and timestamp inconsistencies that can occur when
segments are extracted directly from a compressed video.

The interactive explorer allows the user to navigate precisely through the
recording, mark the first and last frame of a segment, assign a name and a type
("video" or "static"), rebuild the corresponding subvideo from the exact source
frames, and record all segment metadata in data/subvideos/index.json.

Inputs:
    data/input/video_entiere_participant_6.mp4

Generated data:
    data/input_frames_p6/frame_XXXXXX.jpg
    data/subvideos/<name>.mp4
    data/subvideos/index.json
    data/screenshots/*.jpg

Keyboard controls:
    q / d          : move backward / forward by 1 frame
    a / e          : move backward / forward by 10 frames
    Left / Right   : move backward / forward by 25 frames
    z / c          : move backward / forward by 100 frames
    w / x          : move backward / forward by 500 frames
    s              : mark the segment start
    p              : mark the segment end
    Space          : extract the selected segment
    v              : save a screenshot
    Esc            : close the explorer
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "video_entiere_participant_6.mp4"

FRAMES_DIR = PROJECT_ROOT / "data" / "input_frames_p6"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "data" / "subvideos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCREENSHOT_DIR = PROJECT_ROOT / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUTPUT_DIR / "index.json"

# JPEG quality used during the one-time source-frame extraction.
# Higher values preserve image quality at the expense of disk space.
JPEG_QUALITY = 95

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

current_frame_idx = 0
start_frame = None
end_frame = None
frame_display = None
fps = 0.0
frame_files = []


# ---------------------------------------------------------------------------
# Source video extraction
# ---------------------------------------------------------------------------

def extract_video_to_frames():
    """
    Extract full video sequentially into numbered JPG frames.
    This is the new source of truth.
    """
    existing = sorted(FRAMES_DIR.glob("frame_*.jpg"))

    if existing:
        print("=" * 60)
        print(f"Frames already found: {len(existing)}")
        print(f"Using existing folder: {FRAMES_DIR}")
        print("=" * 60)
        return

    print("=" * 60)
    print("EXTRACTING FULL VIDEO INTO FRAMES")
    print("This can take time, but it is done only once.")
    print("=" * 60)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = FRAMES_DIR / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if idx % 1000 == 0:
            print(f"Extracted frame {idx}")

        idx += 1

    cap.release()

    print("=" * 60)
    print(f"Done. Extracted {idx} frames.")
    print(f"Frames folder: {FRAMES_DIR}")
    print("=" * 60)


def load_frame_files():
    """Load and sort all numbered source-frame paths."""
    global frame_files

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))

    if not frame_files:
        raise RuntimeError("No frames found. Extraction failed.")

    print(f"Loaded {len(frame_files)} frames.")


def get_source_fps():
    """Read the source FPS, with a 25 FPS fallback when unavailable."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if source_fps <= 0:
        print("Warning: invalid FPS detected. Using 25 FPS.")
        return 25.0

    return source_fps


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def frames_to_timecode(frame_idx: int, fps: float) -> str:
    """Convert a frame index into a MM:SS:FF timecode."""
    total_sec = frame_idx / fps
    minutes = int(total_sec // 60)
    seconds = int(total_sec % 60)
    frames = int(round((total_sec - int(total_sec)) * fps))
    return f"{minutes:02d}:{seconds:02d}:{frames:02d}"


def read_frame(frame_idx: int):
    """Read one numbered source frame by index."""
    if frame_idx < 0 or frame_idx >= len(frame_files):
        return None

    frame = cv2.imread(str(frame_files[frame_idx]))

    if frame is None:
        print(f"Could not read frame {frame_idx}: {frame_files[frame_idx]}")

    return frame


def build_display(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    timecode = frames_to_timecode(frame_idx, fps)
    text = f"Frame {frame_idx:06d} / {len(frame_files) - 1:06d} | {timecode}"

    cv2.putText(out, text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    y = 60

    if start_frame is not None:
        label = f"START: frame {start_frame} ({frames_to_timecode(start_frame, fps)})"
        color = (0, 200, 0)

        if frame_idx == start_frame:
            cv2.rectangle(out, (0, 0), (8, h), color, -1)

        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

        y += 28

    if end_frame is not None:
        label = f"END  : frame {end_frame} ({frames_to_timecode(end_frame, fps)})"
        color = (0, 100, 255)

        if frame_idx == end_frame:
            cv2.rectangle(out, (w - 8, 0), (w, h), color, -1)

        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

        y += 28

    if start_frame is not None and end_frame is not None:
        n = abs(end_frame - start_frame) + 1
        label = f"Duration: {n} frames ({n / fps:.2f}s)"

        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 0), 1, cv2.LINE_AA)

    hint = (
        "q/d:±1  a/e:±10  LEFT/RIGHT:±25  z/c:±100  w/x:±500"
        " | s:START p:END SPACE:extract v:screenshot ESC:quit"
    )

    cv2.putText(out, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def show_frame(frame_idx: int):
    global current_frame_idx, frame_display

    frame_idx = max(0, min(frame_idx, len(frame_files) - 1))
    current_frame_idx = frame_idx

    raw = read_frame(frame_idx)

    if raw is None:
        return

    frame_display = build_display(raw, frame_idx)
    cv2.imshow("Frame Explorer", frame_display)
    update_title()


def update_title():
    s = f"{start_frame}" if start_frame is not None else "—"
    e = f"{end_frame}" if end_frame is not None else "—"

    cv2.setWindowTitle(
        "Frame Explorer",
        f"Frame Explorer | frame {current_frame_idx}/{len(frame_files)-1}"
        f" | START={s} END={e}"
    )


def mark_start():
    global start_frame

    start_frame = current_frame_idx
    print(f"START marked: frame {start_frame} ({frames_to_timecode(start_frame, fps)})")
    show_frame(current_frame_idx)


def mark_end():
    global end_frame

    end_frame = current_frame_idx
    print(f"END marked: frame {end_frame} ({frames_to_timecode(end_frame, fps)})")
    show_frame(current_frame_idx)


# ---------------------------------------------------------------------------
# Segment index
# ---------------------------------------------------------------------------

def ask_name_and_type():
    print("\n" + "=" * 50)
    print("SUB-VIDEO EXTRACTION")
    print("=" * 50)

    name = input("Name > ").strip()

    if not name:
        name = f"subvideo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    name = name.replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c in "_-")

    print("Type:")
    print("  1 → video")
    print("  2 → static")
    choice = input("Type [1/2] > ").strip()

    segment_type = "video" if choice == "1" else "static"

    return name, segment_type


def load_index():
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


def save_index(entry):
    index = load_index()
    index.append(entry)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Index updated: {INDEX_PATH}")


# ---------------------------------------------------------------------------
# Subvideo extraction
# ---------------------------------------------------------------------------

def extract_subvideo_from_frames():
    """
    Extract sub-video from numbered frames.
    Rebuild the subvideo from an exact inclusive frame interval.
    Numbered frames guarantee exact correspondence with the source video.
    """
    if start_frame is None or end_frame is None:
        print("Please mark START and END first.")
        return

    start_idx = min(start_frame, end_frame)
    end_idx = max(start_frame, end_frame)

    if start_idx == end_idx:
        print("START and END are identical.")
        return

    name, segment_type = ask_name_and_type()

    out_filename = f"{name}.mp4"
    out_path = OUTPUT_DIR / out_filename

    if out_path.exists():
        print(
            f"Output already exists: {out_path}\n"
            "Delete or rename the existing segment before extracting it again."
        )
        return

    first_img = read_frame(start_idx)

    if first_img is None:
        print("Could not read first frame. Extraction cancelled.")
        return

    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    expected_n = end_idx - start_idx + 1
    written = 0

    print("=" * 60)
    print("EXTRACTING FROM NUMBERED FRAMES")
    print(f"Frames: {start_idx} → {end_idx}")
    print(f"Expected: {expected_n} frames")
    print(f"Output: {out_path}")
    print("=" * 60)

    for idx in range(start_idx, end_idx + 1):
        frame = read_frame(idx)

        if frame is None:
            print(f"Warning: missing frame {idx}")
            break

        writer.write(frame)
        written += 1

        if written % 500 == 0:
            print(f"Written {written}/{expected_n} frames")

    writer.release()

    print("Extraction done.")
    print(f"Expected frames: {expected_n}")
    print(f"Written frames : {written}")

    if written == expected_n:
        print("Frame count OK.")
    else:
        print("WARNING: frame count mismatch.")

    entry = {
        # Canonical identifier used by annotations, propagations and reconstruction.
        "name": name,
        "type": segment_type,
        "filename": out_path.name,
        "source_video": VIDEO_PATH.name,
        "source_frames_dir": str(FRAMES_DIR),
        "start_frame": start_idx,
        "end_frame": end_idx,
        "n_frames": expected_n,
        "written_frames": written,
        "duration_sec": round(expected_n / fps, 3),
        "start_tc": frames_to_timecode(start_idx, fps),
        "end_tc": frames_to_timecode(end_idx, fps),
        "extraction_method": "numbered_frames_source_of_truth",
        "extracted_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    save_index(entry)


def save_screenshot():
    """Save the currently displayed frame as a JPEG screenshot."""
    if frame_display is None:
        return

    path = SCREENSHOT_DIR / (
        f"screenshot_frame_{current_frame_idx:06d}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )

    cv2.imwrite(str(path), frame_display)
    print(f"Screenshot saved: {path}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    """Initialize the explorer and run the interactive navigation loop."""
    global fps

    print("=" * 60)
    print("Frame-Based Video Explorer & Extractor")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    fps = get_source_fps()

    print(f"Video       : {VIDEO_PATH.name}")
    print(f"FPS         : {fps:.4f}")
    print(f"Frames dir  : {FRAMES_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print()

    extract_video_to_frames()
    load_frame_files()

    print(f"Frames loaded: {len(frame_files)}")
    print()
    print("Navigation : q/d ±1 | a/e ±10 | ←/→ ±25 | z/c ±100 | w/x ±500")
    print("Extraction : s=START | p=END | SPACE=extract")
    print("Other      : v=screenshot | ESC=quit")
    print()

    cv2.namedWindow("Frame Explorer", cv2.WINDOW_NORMAL)
    show_frame(0)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break

        elif key == ord("q"):
            show_frame(current_frame_idx - 1)

        elif key == ord("d"):
            show_frame(current_frame_idx + 1)

        elif key == ord("a"):
            show_frame(current_frame_idx - 10)

        elif key == ord("e"):
            show_frame(current_frame_idx + 10)

        elif key == 81 or key == 2:
            show_frame(current_frame_idx - 25)

        elif key == 83 or key == 3:
            show_frame(current_frame_idx + 25)

        elif key == ord("z"):
            show_frame(current_frame_idx - 100)

        elif key == ord("c"):
            show_frame(current_frame_idx + 100)

        elif key == ord("w"):
            show_frame(current_frame_idx - 500)

        elif key == ord("x"):
            show_frame(current_frame_idx + 500)

        elif key == ord("s"):
            mark_start()

        elif key == ord("p"):
            mark_end()

        elif key == 32:
            extract_subvideo_from_frames()

        elif key == ord("v"):
            save_screenshot()

    cv2.destroyAllWindows()
    print("Explorer closed.")


if __name__ == "__main__":
    main()

