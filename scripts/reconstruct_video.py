#!/usr/bin/env python3
"""
Reconstruct a complete participant video from precomputed overlays.

The timeline is read from ``data/dataset_index.ods``.

For each global frame:
- dynamic segments use:
  ``data/propagations/<source_name>/<source_name>_overlay.mp4``
- static segments use:
  ``data/static_segmentations/<source_name>/overlay.png``
- gaps use the participant's original extracted frames.

Expected project structure
--------------------------
project_root/
├── data/
│   ├── dataset_index.ods
│   ├── input_frames_p6/
│   ├── propagations/
│   ├── static_segmentations/
│   └── reconstruction/
└── SAM_2/
    └── scripts/
        └── reconstruct_video.py

Examples
--------
Reconstruct participant P6:

    python SAM_2/scripts/reconstruct_video.py P6

Overwrite an existing output:

    python SAM_2/scripts/reconstruct_video.py P6 --overwrite

Reconstruct only a frame interval:

    python SAM_2/scripts/reconstruct_video.py P6 \
        --start-frame 59000 \
        --end-frame 61250 \
        --overwrite

Fail immediately when an expected overlay is missing:

    python SAM_2/scripts/reconstruct_video.py P6 --strict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_INDEX = PROJECT_ROOT / "data" / "dataset_index.ods"
PROPAGATIONS_DIR = PROJECT_ROOT / "data" / "propagations"
STATIC_SEGMENTATIONS_DIR = PROJECT_ROOT / "data" / "static_segmentations"
RECONSTRUCTION_DIR = PROJECT_ROOT / "data" / "reconstruction"

SHEET_NAME = "segments"
FRAME_PATTERN = "frame_{:06d}.jpg"

DEFAULT_FPS = 18.976
DEFAULT_FOURCC = "mp4v"


@dataclass(frozen=True)
class ReconstructionConfig:
    """Runtime parameters for one reconstruction."""

    participant: str
    start_frame: int
    end_frame: int | None
    fps: float
    fourcc: str
    overwrite: bool
    strict: bool


def parse_args() -> ReconstructionConfig:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a participant video from dynamic and static overlays."
        )
    )
    parser.add_argument(
        "participant",
        nargs="?",
        help="Participant identifier, for example P6.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First global frame to reconstruct. Default: 0.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last global frame to reconstruct. Default: last source frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Output video FPS. Default: {DEFAULT_FPS}.",
    )
    parser.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help=f"FourCC codec used by OpenCV. Default: {DEFAULT_FOURCC}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output video and metadata file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort before reconstruction when an expected static or "
            "dynamic overlay is missing."
        ),
    )

    args = parser.parse_args()

    participant = args.participant

    if participant is None:
        participant = input("Participant (for example P6): ")

    participant = participant.strip().upper()
    if not participant:
        parser.error("Participant identifier cannot be empty.")

    if args.start_frame < 0:
        parser.error("--start-frame must be greater than or equal to 0.")

    if args.end_frame is not None and args.end_frame < args.start_frame:
        parser.error("--end-frame must be greater than or equal to --start-frame.")

    if args.fps <= 0:
        parser.error("--fps must be strictly positive.")

    if len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters.")

    return ReconstructionConfig(
        participant=participant,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        fourcc=args.fourcc,
        overwrite=args.overwrite,
        strict=args.strict,
    )


def get_source_frames_dir(participant: str) -> Path:
    """Return the directory containing the participant's original frames."""

    return PROJECT_ROOT / "data" / f"input_frames_{participant.lower()}"


def get_source_frame_path(frames_dir: Path, frame_idx: int) -> Path:
    """Return the expected path of one original frame."""

    return frames_dir / FRAME_PATTERN.format(frame_idx)


def load_participant_segments(participant: str) -> list[dict[str, Any]]:
    """
    Load and validate the participant timeline from ``dataset_index.ods``.

    Only rows whose type is ``static`` or ``video`` are retained.
    """

    if not DATASET_INDEX.exists():
        raise FileNotFoundError(f"Missing dataset index: {DATASET_INDEX}")

    dataframe = pd.read_excel(
        DATASET_INDEX,
        sheet_name=SHEET_NAME,
        engine="odf",
    )

    required_columns = {
        "participant",
        "segment_name",
        "source_name",
        "type",
        "start_frame",
        "end_frame",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise RuntimeError(
            "Missing columns in dataset_index.ods: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["participant"] = (
        dataframe["participant"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe = dataframe[
        dataframe["participant"] == participant
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            f"No timeline found for participant {participant}."
        )

    dataframe["type"] = (
        dataframe["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe["type"].isin({"static", "video"})
    ].copy()

    dataframe["segment_name"] = (
        dataframe["segment_name"].astype(str).str.strip()
    )
    dataframe["source_name"] = (
        dataframe["source_name"].astype(str).str.strip()
    )
    dataframe["start_frame"] = dataframe["start_frame"].astype(int)
    dataframe["end_frame"] = dataframe["end_frame"].astype(int)

    dataframe = dataframe.sort_values(
        ["start_frame", "end_frame"]
    )

    segments = dataframe.to_dict("records")
    validate_segments(segments)

    return segments


def validate_segments(segments: list[dict[str, Any]]) -> None:
    """Validate names, intervals and absence of timeline overlaps."""

    previous_end = -1
    previous_name: str | None = None

    for segment in segments:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        source_name = str(segment["source_name"]).strip()

        if not source_name or source_name.lower() == "nan":
            raise RuntimeError("A segment has an empty source_name.")

        if end_frame < start_frame:
            raise RuntimeError(
                f"Invalid interval for {source_name}: "
                f"{start_frame} -> {end_frame}"
            )

        if start_frame <= previous_end:
            raise RuntimeError(
                f"Overlapping segments: {previous_name} and {source_name}"
            )

        previous_end = end_frame
        previous_name = source_name


def find_last_source_frame(frames_dir: Path) -> int:
    """Return the highest frame index available in the source directory."""

    max_frame_idx = -1

    for frame_path in frames_dir.glob("frame_*.jpg"):
        try:
            frame_idx = int(frame_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue

        max_frame_idx = max(max_frame_idx, frame_idx)

    if max_frame_idx < 0:
        raise RuntimeError(f"No source frames found in {frames_dir}")

    return max_frame_idx


def read_original_frame(frames_dir: Path, frame_idx: int):
    """Read one original frame or raise a clear error."""

    frame_path = get_source_frame_path(frames_dir, frame_idx)
    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(
            f"Could not read original frame {frame_idx}: {frame_path}"
        )

    return frame


def fit_frame(frame, width: int, height: int):
    """Resize a frame only when its dimensions differ from the output size."""

    if frame.shape[:2] == (height, width):
        return frame

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


class DynamicOverlayReader:
    """Sequential reader for one propagated dynamic overlay video."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.path = (
            PROPAGATIONS_DIR
            / source_name
            / f"{source_name}_overlay.mp4"
        )
        self.capture: cv2.VideoCapture | None = None
        self.next_frame = 0

    @property
    def available(self) -> bool:
        """Return whether the expected overlay video exists."""

        return self.path.is_file()

    def read(self, local_frame: int):
        """Read one local overlay frame."""

        if not self.available:
            return None

        if self.capture is None:
            self.capture = cv2.VideoCapture(str(self.path))

            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
                return None

        if local_frame != self.next_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            self.next_frame = local_frame

        success, frame = self.capture.read()
        if not success:
            return None

        self.next_frame = local_frame + 1
        return frame

    def close(self) -> None:
        """Release the OpenCV video capture."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None


def load_static_overlay(source_name: str):
    """
    Load one static overlay image.

    Static folders must follow the canonical naming convention:

        data/static_segmentations/<source_name>/overlay.png
    """

    overlay_path = (
        STATIC_SEGMENTATIONS_DIR
        / source_name
        / "overlay.png"
    )

    return cv2.imread(str(overlay_path))


def inspect_required_overlays(
    segments: list[dict[str, Any]],
) -> tuple[
    dict[str, DynamicOverlayReader],
    dict[str, Any],
    list[str],
    list[str],
]:
    """Create overlay readers and report missing dynamic/static results."""

    dynamic_readers: dict[str, DynamicOverlayReader] = {}
    static_overlays: dict[str, Any] = {}

    for segment in segments:
        source_name = str(segment["source_name"]).strip()

        if segment["type"] == "video":
            dynamic_readers.setdefault(
                source_name,
                DynamicOverlayReader(source_name),
            )
        else:
            static_overlays.setdefault(
                source_name,
                load_static_overlay(source_name),
            )

    missing_dynamic = sorted(
        source_name
        for source_name, reader in dynamic_readers.items()
        if not reader.available
    )

    missing_static = sorted(
        source_name
        for source_name, overlay in static_overlays.items()
        if overlay is None
    )

    return (
        dynamic_readers,
        static_overlays,
        missing_dynamic,
        missing_static,
    )


def print_missing_overlays(
    missing_dynamic: list[str],
    missing_static: list[str],
) -> None:
    """Print missing overlay paths with enough information to fix them."""

    if missing_dynamic:
        print("\nMissing dynamic overlays:")
        for source_name in missing_dynamic:
            expected_path = (
                PROPAGATIONS_DIR
                / source_name
                / f"{source_name}_overlay.mp4"
            )
            print(f"  - {source_name}")
            print(f"    Expected: {expected_path}")

    if missing_static:
        print("\nMissing static overlays:")
        for source_name in missing_static:
            expected_path = (
                STATIC_SEGMENTATIONS_DIR
                / source_name
                / "overlay.png"
            )
            print(f"  - {source_name}")
            print(f"    Expected: {expected_path}")


def reconstruct(config: ReconstructionConfig) -> None:
    """Run the full reconstruction."""

    participant = config.participant
    frames_dir = get_source_frames_dir(participant)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Missing source frames directory: {frames_dir}"
        )

    segments = load_participant_segments(participant)
    last_source_frame = find_last_source_frame(frames_dir)

    start_frame = config.start_frame
    end_frame = (
        last_source_frame
        if config.end_frame is None
        else min(config.end_frame, last_source_frame)
    )

    if start_frame > end_frame:
        raise ValueError(
            f"Invalid reconstruction range: "
            f"{start_frame} -> {end_frame}"
        )

    first_frame = read_original_frame(frames_dir, start_frame)
    height, width = first_frame.shape[:2]

    RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)

    output_video = (
        RECONSTRUCTION_DIR
        / f"{participant}_full_overlay.mp4"
    )
    output_metadata = (
        RECONSTRUCTION_DIR
        / f"{participant}_full_overlay_metadata.json"
    )

    if not config.overwrite and (
        output_video.exists() or output_metadata.exists()
    ):
        raise FileExistsError(
            "Output already exists.\n"
            "Delete it first or run the script with --overwrite."
        )

    (
        dynamic_readers,
        static_overlays,
        missing_dynamic,
        missing_static,
    ) = inspect_required_overlays(segments)

    print_missing_overlays(
        missing_dynamic,
        missing_static,
    )

    if config.strict and (missing_dynamic or missing_static):
        for reader in dynamic_readers.values():
            reader.close()

        raise RuntimeError(
            "Missing overlays detected while strict mode is enabled."
        )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*config.fourcc),
        config.fps,
        (width, height),
    )

    if not writer.isOpened():
        for reader in dynamic_readers.values():
            reader.close()

        raise RuntimeError(
            f"Could not create output video: {output_video}"
        )

    total_frames = end_frame - start_frame + 1
    segment_index = 0

    dynamic_overlay_frames = 0
    static_overlay_frames = 0
    original_gap_frames = 0
    missing_dynamic_fallback_frames = 0
    missing_static_fallback_frames = 0

    current_segment_key: tuple[Any, ...] | None = None
    current_gap_start: int | None = None

    print("\n" + "=" * 72)
    print("FULL VIDEO RECONSTRUCTION")
    print("=" * 72)
    print(f"Participant : {participant}")
    print(f"Timeline    : {DATASET_INDEX}")
    print(f"Frames      : {frames_dir}")
    print(f"Range       : {start_frame} -> {end_frame}")
    print(f"FPS         : {config.fps}")
    print(f"Strict mode : {config.strict}")
    print("=" * 72)

    try:
        for count, global_frame in enumerate(
            range(start_frame, end_frame + 1),
            start=1,
        ):
            while (
                segment_index < len(segments)
                and global_frame
                > int(segments[segment_index]["end_frame"])
            ):
                segment_index += 1

            active_segment = None

            if segment_index < len(segments):
                candidate = segments[segment_index]

                if (
                    int(candidate["start_frame"])
                    <= global_frame
                    <= int(candidate["end_frame"])
                ):
                    active_segment = candidate

            output_frame = None
            fallback_reason: str | None = None

            if active_segment is not None:
                if current_gap_start is not None:
                    print(
                        f"[ORIGINAL GAP] {current_gap_start} "
                        f"-> {global_frame - 1}"
                    )
                    current_gap_start = None

                source_name = str(
                    active_segment["source_name"]
                ).strip()
                segment_type = str(
                    active_segment["type"]
                ).strip()

                segment_key = (
                    segment_type,
                    source_name,
                    int(active_segment["start_frame"]),
                    int(active_segment["end_frame"]),
                )

                if segment_key != current_segment_key:
                    print(
                        f"[{segment_type.upper()}] {source_name} | "
                        f"{active_segment['start_frame']} -> "
                        f"{active_segment['end_frame']}"
                    )
                    current_segment_key = segment_key

                if segment_type == "video":
                    local_frame = (
                        global_frame
                        - int(active_segment["start_frame"])
                    )

                    output_frame = dynamic_readers[
                        source_name
                    ].read(local_frame)

                    if output_frame is not None:
                        dynamic_overlay_frames += 1
                    else:
                        fallback_reason = "missing_dynamic"

                else:
                    output_frame = static_overlays.get(source_name)

                    if output_frame is not None:
                        static_overlay_frames += 1
                    else:
                        fallback_reason = "missing_static"

            else:
                current_segment_key = None

                if current_gap_start is None:
                    current_gap_start = global_frame

                fallback_reason = "gap"

            if output_frame is None:
                output_frame = read_original_frame(
                    frames_dir,
                    global_frame,
                )

                if fallback_reason == "missing_dynamic":
                    missing_dynamic_fallback_frames += 1
                elif fallback_reason == "missing_static":
                    missing_static_fallback_frames += 1
                else:
                    original_gap_frames += 1

            output_frame = fit_frame(
                output_frame,
                width,
                height,
            )
            writer.write(output_frame)

            if count % 1000 == 0 or count == total_frames:
                progress = 100.0 * count / total_frames
                print(
                    f"Progress: {count}/{total_frames} "
                    f"({progress:.1f}%) | "
                    f"global frame {global_frame}"
                )

        if current_gap_start is not None:
            print(
                f"[ORIGINAL GAP] {current_gap_start} "
                f"-> {end_frame}"
            )

    finally:
        writer.release()

        for reader in dynamic_readers.values():
            reader.close()

    metadata = {
        "participant": participant,
        "dataset_index": str(DATASET_INDEX),
        "source_frames_directory": str(frames_dir),
        "output_video": str(output_video),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_frames_written": total_frames,
        "dynamic_overlay_frames": dynamic_overlay_frames,
        "static_overlay_frames": static_overlay_frames,
        "original_gap_frames": original_gap_frames,
        "missing_dynamic_fallback_frames": (
            missing_dynamic_fallback_frames
        ),
        "missing_static_fallback_frames": (
            missing_static_fallback_frames
        ),
        "fps": config.fps,
        "fourcc": config.fourcc,
        "width": width,
        "height": height,
        "strict_mode": config.strict,
        "missing_dynamic_overlays": missing_dynamic,
        "missing_static_overlays": missing_static,
        "completed_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with output_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Output video               : {output_video}")
    print(f"Metadata                   : {output_metadata}")
    print(f"Dynamic overlay frames     : {dynamic_overlay_frames}")
    print(f"Static overlay frames      : {static_overlay_frames}")
    print(f"Original gap frames        : {original_gap_frames}")
    print(
        "Missing dynamic fallbacks : "
        f"{missing_dynamic_fallback_frames}"
    )
    print(
        "Missing static fallbacks  : "
        f"{missing_static_fallback_frames}"
    )
    print("=" * 72)


def main() -> None:
    """Command-line entry point."""

    reconstruct(parse_args())


if __name__ == "__main__":
    main()

