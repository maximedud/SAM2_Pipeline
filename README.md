
SAM2 Pipeline — Reference Segmentation and Tobii Gaze Mapping
A research pipeline developed during an internship with the STARS team at Inria Sophia Antipolis. It supports manual prompting and segmentation of a reference participant's screen recording using Segment Anything Model 2 (SAM 2), reconstruction of an annotated reference video, and alignment of MASC/Tobii gaze samples with the resulting segmentation masks.

Core design principle: reproducibility. Scripts, expected inputs, intermediate outputs, assumptions and quality-control information are documented so that the workflow can be reproduced and extended. This repository contains technical processing code, not the econometric analysis or private participant recordings.

Scope and status: This README describes the seven selected scripts and their interfaces as reviewed. It is not a claim that the complete workflow has been independently re-run on a clean machine. Review the compatibility notes below before interpreting gaze/object measurements.

1. Workflow
P6 source recording
  │
  └─ exploration_extraction.py
       ├─ data/input_frames_p6/frame_XXXXXX.jpg
       └─ data/subvideos/{index.json, segment videos}
              │
              ├─ create_dataset_index.py
              │    └─ data/dataset_index.ods
              │
              └─ annotation_batch_subvideos.py
                   └─ data/annotations/*.json
                         │
                         ├─ segment_static_subvideos.py
                         │    └─ data/static_segmentations/<segment>/{masks,overlay.png,metadata.json}
                         │
                         └─ propagation_subvid_full_chunked.py
                              └─ data/propagations/<segment>/{masks,overlay video,metadata.json}
                                      │
             ┌────────────────────────┴───────────────────────────┐
             │                                                    │
    reconstruct_video.py                         build_masc_tobii_sam_v3_opti.py
    (optional visualization)                     + authorized Tobii CSVs
             │                                                    │
    data/reconstruction/                         MASC/derived/tobii_sam_v3_opti/
    P6_full_overlay.mp4                          participants/PXX/*.csv
Reconstruction is optional and is not required to map Tobii gaze samples to segmentation masks.

2. Repository layout
Important: the repository is a source-code handover, not a plug-and-play standalone application. Recreate the original internship directory layout before running the scripts. In particular, the seven files from this repository's scripts/ directory must be placed under projet_SAM/SAM_2/scripts/. This preserves the existing Path(__file__).resolve().parents[2] assumptions; there is no need to rename the scripts or change those paths solely to match the GitHub repository layout.

For example, after downloading or cloning SAM2_Pipeline, copy its scripts/*.py into projet_SAM/SAM_2/scripts/, and place the checkpoint separately as shown below. Do not run the scripts directly from SAM2_Pipeline/scripts/ unless you have adapted their paths. Some scripts have additional path settings (notably the Tobii input directory), which are described below.

Required working layout:

projet_SAM/
├── SAM_2/
│   ├── scripts/
│   │   ├── exploration_extraction.py
│   │   ├── create_dataset_index.py
│   │   ├── annotation_batch_subvideos.py
│   │   ├── segment_static_subvideos.py
│   │   ├── propagation_subvid_full_chunked.py
│   │   ├── reconstruct_video.py
│   │   └── build_masc_tobii_sam_v3_opti.py
│   └── checkpoints/
│       └── sam2.1_hiera_small.pt        # obtain separately
├── data/
│   ├── input/
│   │   └── video_entiere_participant_6.mp4  # private input
│   ├── input_frames_p6/                # generated, private
│   ├── subvideos/                      # generated
│   ├── annotations/                    # generated / curated
│   ├── static_segmentations/           # generated
│   ├── propagations/                   # generated
│   ├── dataset_index.ods               # generated
│   └── reconstruction/                 # generated
└── MASC/
    └── derived/                       # generated
Tobii CSVs are expected outside the project by default:

~/Downloads/DatasetRed/P01/byTASK/MASC/P01_MASC_Tobii.csv
~/Downloads/DatasetRed/P02/byTASK/MASC/P02_MASC_Tobii.csv
...
Change DATASET_RED in build_masc_tobii_sam_v3_opti.py if the authorized dataset is stored elsewhere. Do not commit participant-level Tobii data.

3. Installation
Use a Python environment compatible with the official SAM 2 package and the available PyTorch/CUDA configuration. The original GPU processing was developed in a Linux research-computing environment; hardware and dependency versions must be checked on a new machine.

Create and activate a Python virtual environment.

Install a compatible build of PyTorch (with CUDA support for GPU propagation).

Install the official SAM 2 package following its upstream installation instructions. The scripts import sam2.build_sam and sam2.sam2_image_predictor.

Install the repository dependencies, including odfpy for the .ods timeline files:

pip install -r requirements.txt
pip install odfpy  # if not yet listed in requirements.txt
The current repository requirements.txt should be updated to include odfpy. It already lists NumPy, OpenCV, PyTorch, torchvision, pandas, matplotlib, PyYAML and tqdm. Install SAM 2 separately as noted above.

Obtain the SAM 2.1 Hiera Small checkpoint from an authorized official source and place it at:

SAM_2/checkpoints/sam2.1_hiera_small.pt
The scripts use the model configuration identifier:

configs/sam2.1/sam2.1_hiera_s.yaml
GPU requirement: propagation_subvid_full_chunked.py has REQUIRE_CUDA = True. Its execution requires a CUDA-capable GPU and a compatible PyTorch installation. Other scripts have different device behavior; see their configuration sections.

No environment lockfile is provided here; for strict reproduction, record the exact Python, PyTorch, CUDA, SAM 2, OpenCV, NumPy and pandas versions used in each run.

4. Execution order
Run commands from the projet_SAM/ working-project root after copying the scripts to SAM_2/scripts/ as described in Section 2. The original script filenames are deliberately preserved.

Step 1 — Extract P6 frames and subvideos
python SAM_2/scripts/exploration_extraction.py
The script expects:

data/input/video_entiere_participant_6.mp4
It extracts numbered source frames to data/input_frames_p6/ and supports interactive creation of subvideo segments and data/subvideos/index.json. The segment-selection/annotation of the timeline is not a fully unattended step; document or preserve the curated index for exact reproduction.

Step 2 — Create the reconstruction timeline
python SAM_2/scripts/create_dataset_index.py
Reads data/subvideos/index.json and writes data/dataset_index.ods (sheet segments) with participant, segment_name, source_name, task, type, start_frame, end_frame, duration_frames, and notes.

The script sets PARTICIPANT = "P6" and OVERWRITE_EXISTING_FILE = False. Do not overwrite an existing curated ODS file: this generator creates the initial P6 timeline, and later manual or multi-participant changes are not automatically preserved.

Dependencies: pandas, odfpy.

Step 3 — Annotate objects
python SAM_2/scripts/annotation_batch_subvideos.py
Use the interactive annotation interface to provide positive/negative point prompts and, where appropriate, bounding boxes. Annotations are saved in data/annotations/ and later consumed by the static and dynamic segmentation scripts. Keep object IDs and annotation filenames consistent across stages.

Step 4A — Segment static scenes
python SAM_2/scripts/segment_static_subvideos.py
Processes annotation JSON files whose type is static, predicts SAM 2 masks for the selected frame, and writes:

data/static_segmentations/<segment>/
├── raw.png
├── overlay.png
├── metadata.json
└── masks/
    ├── object_1.png
    └── object_2.png
It skips existing outputs when SKIP_ALREADY_DONE = True. Static masks are reused for each corresponding static Tobii segment; no video propagation is required.

Step 4B — Propagate masks over video scenes
python SAM_2/scripts/propagation_subvid_full_chunked.py
Uses annotated prompts and chunked SAM 2 propagation. Relevant script settings include:

WINDOW_SIZE = 300
STRIDE = 250
MAX_WIDTH = 960
REQUIRE_CUDA = True
SKIP_ALREADY_DONE = True
Typical outputs:

data/propagations/<segment>/
├── metadata.json
├── <segment>_overlay.mp4
├── <segment>_masks.csv
├── masks/
│   └── obj_01/
│       └── frame_000000.png
└── overlays/
    └── frame_000000.jpg
Index naming note: the reviewed propagation version references data/subvideos/index_P6.json, while the extraction script generates data/subvideos/index.json. If that line has not already been updated in your copy, change it to:

INDEX_PATH = SUBVIDEOS_DIR / "index.json"
The original environment contained both index files with identical content, but a fresh checkout should rely on one canonical name.

Step 5A — Reconstruct the reference video (optional)
python SAM_2/scripts/reconstruct_video.py P6
Reads the segments sheet in data/dataset_index.ods, the original P6 frames in data/input_frames_p6/, dynamic overlay videos, and static overlay.png files. Writes:

data/reconstruction/P6_full_overlay.mp4
data/reconstruction/P6_full_overlay_metadata.json
Options:

python SAM_2/scripts/reconstruct_video.py P6 --strict
python SAM_2/scripts/reconstruct_video.py P6 --start-frame 59000 --end-frame 61250
python SAM_2/scripts/reconstruct_video.py P6 --overwrite
--strict aborts if required overlay files are missing. Without it, missing overlays fall back to original frames, and the metadata records fallback counts. --overwrite replaces the normal full-output filename even for a short interval; back up existing outputs before using it.

Step 5B — Align Tobii samples with SAM masks
Process one participant first:

python SAM_2/scripts/build_masc_tobii_sam_v3_opti.py --participant P01
Then, after inspecting the QC outputs, process all participants:

python SAM_2/scripts/build_masc_tobii_sam_v3_opti.py --all
Other supported options:

python SAM_2/scripts/build_masc_tobii_sam_v3_opti.py --participant P01 --resume
python SAM_2/scripts/build_masc_tobii_sam_v3_opti.py --participant P01 --workers 2
python SAM_2/scripts/build_masc_tobii_sam_v3_opti.py --participant P01 --gaze-mode pixels --gaze-source-width 1920 --gaze-source-height 1080
The Tobii script reads timeSF, Event, EventValue, GazePointX, and GazePointY from semicolon-separated CSVs. It maps MASC video and question events to the corresponding reference segments; other events (including introductions) are not treated as SAM segments.

The script produces participant-level outputs under:

MASC/derived/tobii_sam_v3_opti/participants/P01/
├── segments.csv
├── unmapped_events.csv
├── gaze.csv
├── qc_report.csv
├── raw_enriched/
│   └── gaze_samples.csv
└── _segment_checkpoints/
A cross-participant summary is written to:

MASC/derived/tobii_sam_v3_opti/all_participants_summary.csv
Review qc_report.csv and unmapped_events.csv before interpreting gaze metrics. Checkpoint resume is intended for interrupted processing; if upstream masks or input data change, rerun without --resume to avoid stale intermediate results.

5. Methodological assumptions and validation
P6 is the segmentation reference. Masks from the reference video are reused for other MASC participants rather than segmenting every participant recording independently.
Temporal mapping is proportional. A Tobii event's relative time is mapped to a frame in the corresponding P6 segment. This is not a verified exact frame-to-frame synchronization between independently recorded sessions.
Static scenes use one mask frame. The same static segmentation is used throughout the corresponding Tobii event.
Gaze position and mask pixels must share the same coordinate system. The Tobii script supports normalized/pixel gaze coordinates and optional source-screen dimensions. Validate dimensions and visual alignment for representative participants.
Potential image-size mismatch: the dynamic propagation script uses MAX_WIDTH = 960 for SAM inference while recording original dimensions in its metadata. Verify that saved mask PNGs are at the dimensions assumed by the Tobii script. If not, resize masks or transform gaze coordinates consistently before interpreting gaze/object intersections. This is a methodological validation item, not merely a cosmetic issue.
Object naming: static masks use object_1.png, while dynamic mask folders use obj_01/. The Tobii reader indexes both patterns, but their labels should be normalized before comparing object identities across segment types.
Mask availability versus detection: objects_present may indicate that a readable mask file exists, even if that mask contains no positive pixels. A gaze hit is computed by testing actual mask pixels.
Segmentation errors propagate into gaze measurements. Incorrect or empty SAM masks, unrecognized MASC events, invalid gaze samples, and temporal alignment errors can affect results.
Quality control is required. Check mapped events, gaze validity, out-of-bounds samples, mask coverage and representative visual overlays. Do not interpret output CSVs as ground truth without validation.
6. Data privacy and sharing
Do not commit participant recordings, extracted source frames, Tobii CSVs, raw annotations containing sensitive source references, masks, overlays, or derived participant-level outputs unless an authorized data-sharing policy explicitly permits it.

Transfer any restricted inputs and computed results separately using an approved institutional storage mechanism. The GitHub repository is intended for source code, documentation and safe configuration examples. SAM 2 checkpoint files should be obtained separately from the authorized upstream source.

Suggested .gitignore entries for the working project (the GitHub repository already has its own .gitignore):

.venv/
__pycache__/
*.py[cod]
.idea/
.DS_Store
*.pt
*.pth
*.ckpt
*.safetensors
data/input/
data/input_frames_*/
data/input_videos/
data/subvideos/
data/annotations/
data/annotation_previews/
data/screenshots/
data/static_segmentations/
data/propagations/
data/reconstruction/
data/dataset_index*.ods
MASC/data/
MASC/derived/
DatasetRed/
*.log
These exclusions are conservative. If a curated, non-sensitive example index or annotation is approved for publication, add it explicitly rather than uploading the entire generated-data directory.

Repository publication notes
The GitHub root currently contains scripts/, checkpoints/README.md, data/input/, .gitignore, requirements.txt and this documentation. The checkpoints/README.md is a placeholder, not the model weights. Keep data/input/ free of private recordings. Rename this file to README.md at the repository root so GitHub displays it automatically.

.gitignore only prevents new untracked files from being added; it does not remove files already committed to Git history.

7. Reproducibility checklist
Before claiming a new run is reproducible, verify that:

The original input video and authorized Tobii files are available privately.
The same segment boundaries and annotation prompts are available or recreated.
data/subvideos/index.json is the index used by the propagation script.
The SAM 2.1 Small checkpoint, model config, software versions and GPU setup are recorded.
Static and dynamic masks are present and have verified dimensions.
The reconstruction timeline, if used, matches the intended participant and frame boundaries.
Tobii event mapping and gaze-coordinate scaling have been checked.
QC reports and representative overlays have been inspected.
No restricted participant data or computed sensitive outputs are included in the public repository.
8. Project context
Developed as part of a research internship in the STARS team at Inria Sophia Antipolis, using the MASC experimental material and Tobii eye-tracking data. The pipeline is intended as a reusable technical foundation for subsequent research and methodological improvements.

For reuse, consult the official SAM 2 repository and its applicable license, and follow the institution's data-access and research-ethics requirements.

