#!/usr/bin/env python3
"""
Create data/dataset_index.ods from the existing P6 index.json.

The generated file contains only the information needed later for
multi-participant reconstruction:

- participant
- segment_name
- source_name
- task
- type
- start_frame
- end_frame
- duration_frames
- notes

Requirements:
    pip install pandas odfpy
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARTICIPANT = "P6"

INDEX_PATH = PROJECT_ROOT / "data" / "subvideos" / "index.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_index.ods"

SHEET_NAME = "segments"
OVERWRITE_EXISTING_FILE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_entries() -> list[dict]:
    """Load segment entries from index.json."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")

    with INDEX_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("segments", "subvideos", "entries", "items"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unsupported index.json structure. "
        "Expected a list or a dictionary containing a segment list."
    )


def get_frame(entry: dict, main_key: str, fallback_key: str) -> int:
    """Read a frame number while supporting both key conventions."""
    value = entry.get(main_key, entry.get(fallback_key))

    if value is None:
        raise KeyError(
            f"Missing {main_key}/{fallback_key} "
            f"for segment {entry.get('name', '<unknown>')}"
        )

    return int(value)


def make_generic_name(source_name: str) -> str:
    """
    Remove a final frame range from static segment names.

    Example:
        texte_consigne_task1_001432_002791
        becomes
        texte_consigne_task1
    """
    return re.sub(r"_\d{6}_\d{6}$", "", source_name)


def extract_task(segment_name: str) -> str:
    """Extract task1, task3, task5, etc. from a segment name."""
    match = re.search(r"task\d+", segment_name, flags=re.IGNORECASE)
    return match.group(0).lower() if match else ""


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def build_rows(entries: list[dict]) -> list[dict]:
    """Convert index entries into rows for dataset_index.ods."""
    rows = []

    for entry in entries:
        segment_type = str(entry.get("type", "")).strip().lower()

        if segment_type not in {"static", "video"}:
            continue

        raw_name = str(entry.get("name", "")).strip()

        if not raw_name:
            print("Skipped one unnamed entry.")
            continue

        segment_name = make_generic_name(raw_name)
        source_name = segment_name

        try:
            start_frame = get_frame(
                entry,
                "origin_start_frame",
                "start_frame",
            )
            end_frame = get_frame(
                entry,
                "origin_end_frame",
                "end_frame",
            )
        except (KeyError, TypeError, ValueError) as error:
            print(f"Skipped {raw_name}: {error}")
            continue

        if end_frame < start_frame:
            print(
                f"Skipped {raw_name}: invalid interval "
                f"{start_frame} -> {end_frame}"
            )
            continue

        rows.append(
            {
                "participant": PARTICIPANT,
                "segment_name": segment_name,
                "source_name": source_name,
                "task": extract_task(segment_name),
                "type": segment_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "notes": "",
            }
        )

    rows.sort(key=lambda row: row["start_frame"])
    return rows


def save_ods(rows: list[dict]) -> None:
    """Save rows in a LibreOffice Calc .ods file."""
    if OUTPUT_PATH.exists() and not OVERWRITE_EXISTING_FILE:
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists.\n"
            "Rename or delete it first, or set "
            "OVERWRITE_EXISTING_FILE = True."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "participant",
        "segment_name",
        "source_name",
        "task",
        "type",
        "start_frame",
        "end_frame",
        "duration_frames",
        "notes",
    ]

    dataframe = pd.DataFrame(rows, columns=columns)

    with pd.ExcelWriter(OUTPUT_PATH, engine="odf") as writer:
        dataframe.to_excel(
            writer,
            sheet_name=SHEET_NAME,
            index=False,
        )


def main() -> None:
    entries = load_entries()
    rows = build_rows(entries)

    if not rows:
        raise RuntimeError("No usable static or video segment found.")

    save_ods(rows)

    print(f"Created: {OUTPUT_PATH}")
    print(f"Participant: {PARTICIPANT}")
    print(f"Segments: {len(rows)}")


if __name__ == "__main__":
    main()