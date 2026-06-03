# SAM2 Demo Pipeline

## Description

This project demonstrates a video segmentation pipeline based on SAM 2 (Segment Anything Model 2).

The workflow is divided into two steps:

1. Manual annotation and prompt creation
2. Video propagation and export

Outputs include:

- segmented video
- CSV file
- contour extraction
- preview images

---

## Project structure

SAM2_demo/

scripts/
- annotation_sam_preview.py
- propagation_cuda.py

checkpoints/
- sam2.1_hiera_small.pt

data/input/
- video_without_cut.mp4

annotations/

outputs/

---

## Installation

Create a Python environment:

python -m venv .venv

Activate environment:

Windows:
.venv\Scripts\activate

Linux:
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## Script 1: Annotation

Run:

python scripts/annotation_sam_preview.py

Controls:

q : previous frame
d : next frame
b : -25 frames
f : +25 frames

Left click : positive point
Right click : negative point

s : save annotations
u : remove last point
r : reset current frame annotations

ESC : quit

Output:

annotations/annotations_*.json

---

## Script 2: Propagation

Run:

python scripts/propagation_cuda.py

Outputs:

- segmented video
- CSV file
- contour JSON
- preview images

---

## Notes

This project combines official SAM 2 functionalities with custom Python developments built for experimentation and rapid prototyping.

Some components were developed with the assistance of AI in order to quickly obtain functional results and overcome development issues encountered during the project.