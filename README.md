# SAM2 Video Segmentation Pipeline

Pipeline developed during my internship at **INRIA STARS** for the segmentation and reconstruction of participant videos using **Segment Anything Model 2 (SAM2)**.

The objective of this project is to reconstruct an annotated participant video by combining static image segmentation and dynamic video propagation.

---

# Overview

The pipeline processes a participant recording through six successive stages:

```
Original participant video
            │
            ▼
1. Exploration & Extraction
            │
            ▼
2. Dataset Index Generation
            │
            ▼
3. Interactive Annotation
            │
     ┌──────┴──────┐
     ▼             ▼
4. Static      5. Dynamic
 Segmentation     Propagation
     │             │
     └──────┬──────┘
            ▼
6. Video Reconstruction
            │
            ▼
Final annotated participant video
```

---

# Repository Structure

```
SAM2_Pipeline/

├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── exploration_extraction.py
│   ├── create_dataset_index.py
│   ├── annotation_batch_subvideos.py
│   ├── segment_static_subvideos.py
│   ├── propagation_subvid.py
│   └── reconstruct_video.py
│
└── data/
    └── input_videos/
```

All remaining directories are automatically created during execution.

---

# Input Data

Before running the pipeline, place the original **Participant 6** video in:

```
data/input_videos/
```

---

# Pipeline Description

## 1. exploration_extraction.py

Extracts all static and dynamic subvideos from the original participant recording.

Outputs:

- extracted frames
- extracted subvideos
- `index.json`

---

## 2. create_dataset_index.py

Creates the experiment timeline (`dataset_index.ods`) from the extraction index.

The generated dataset contains:

- participant
- segment name
- task
- segment type
- global frame interval

---

## 3. annotation_batch_subvideos.py

Interactive annotation tool based on SAM2.

Main features:

- multi-object annotation
- positive and negative clicks
- multiple annotated frames
- annotation editing
- JSON export

Outputs:

```
annotations/
```

---

## 4. segment_static_subvideos.py

Segments all static experimental screens from the generated annotations.

Outputs:

```
static_segmentations/
```

---

## 5. propagation_subvid.py

Propagates the annotated masks through each dynamic subvideo using SAM2 Video Predictor.

Outputs:

```
propagations/
```

---

## 6. reconstruct_video.py

Reconstructs the complete participant video by combining:

- propagated dynamic overlays
- segmented static overlays
- original frames outside annotated segments

Outputs:

```
reconstruction/
```

---

# Usage

Run the scripts in the following order:

```
exploration_extraction.py

↓

create_dataset_index.py

↓

annotation_batch_subvideos.py

↓

segment_static_subvideos.py

↓

propagation_subvid.py

↓

reconstruct_video.py
```

---

# Requirements

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Pandas
- ODFPy
- SAM2

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Notes

This repository contains the complete pipeline developed during the internship.

The annotation stage is **interactive** and requires manual user input before segmentation and propagation.

Intermediate files and output directories are automatically generated during execution and are intentionally excluded from version control.

---

# Current Features

- Interactive SAM2 annotation
- Multi-object support
- Multi-frame annotations
- Static image segmentation
- Dynamic video propagation
- Automatic participant video reconstruction
- Consistent segment naming
- Automatic metadata generation

---

# Future Work

- Eye-tracking overlay integration
- Pipeline automation for multiple participants
- Improved visualization tools
- SAM 3.x integration

---

# Author

**Maxime Durand**

Master's Internship – INRIA STARS

2026