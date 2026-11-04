# Road Intersection Detection from Satellite Imagery

**Course:** Intelligent Transportation Systems (ITS) — Gati Shakti Vishwavidyalaya (GSV)
**Project:** DL-Based Road Feature Extraction from Satellite Imagery — Feature of Interest: Road Intersection Detection via Road Network Graph Extraction
**Implementation:** Pratima Chauhan
**Original proposal team:** Vedant Jadhav, Pratima Chauhan, Divy Jain

---

## What this project does

Road intersections matter for traffic signal coordination, congestion management and network planning. This project finds them automatically in satellite imagery, using two approaches:

| | Approach A — YOLO baseline | Approach B — Graph-based (main approach) |
|---|---|---|
| Idea | Detect each intersection as an object | Segment road pixels, build a road graph, read junctions from it |
| Labels needed | Intersection points / boxes | Pixel-wise road masks |
| Output | Bounding boxes | Road network graph + junction points (GeoJSON) |
| Status | Implemented (v1, known issues below) | In progress |

The two approaches use **different evaluation methods** (box-overlap mAP vs. distance-tolerance matching). Their numbers are not directly comparable unless evaluated with the same protocol.

---

## Project status

| Stage | Status |
|---|---|
| Tile download (ESRI World Imagery, Gandhinagar, zoom 18) | Done |
| OSM intersection extraction | Done — labelling rule to be revised (see known issues) |
| YOLO26n training and prediction samples | Done (v1) |
| Graph pipeline: DeepGlobe dataset preparation | Planned |
| U-Net (ResNet-34) road segmentation | Planned |
| Skeletonization → road graph → intersections | Planned |
| Georeferencing + OSM validation | Planned |
| GeoJSON export + map dashboard | Planned |

---

## Approach A — YOLO baseline (v1)

### Pipeline

Run all scripts **from the repository root**:

```
python scripts/download_tiles.py          # 272 ESRI tiles, 256x256 px, zoom 18 (~0.55 m/px)
python scripts/get_osm_intersections.py   # OSM junctions -> data/intersections.csv (1,190 points)
python scripts/generate_labels.py         # 32x32 px YOLO boxes -> data/dataset/ (221 image-label pairs)
python scripts/split_dataset.py           # 80/20 train/val -> data/dataset_split/
python scripts/train.py                   # YOLO26n, imgsz 256, up to 150 epochs, patience 50
python scripts/visualize_predictions.py   # sample predictions -> data/prediction_samples/
```

Study area: Gandhinagar sector grid, bounding box 72.63–72.65 E, 23.21–23.23 N.

### Result (validation set)

| Run | Best epoch | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|---:|
| `intersection_detector_v2-2` | 42 of 95 | 0.292 | 0.266 | 0.214 |

Values are the best-mAP@0.5 row of `results.csv`. Training curves and the confusion matrix are in `docs/results/yolo_v1/`.

### Known issues (to be fixed in a later milestone)

1. **Data leakage:** 14 of the 53 validation images also appear in the training folder, because `split_dataset.py` was run twice without emptying its output folder. The figures above are therefore likely optimistic.
2. **No held-out test set:** the validation set was also used for early stopping and selecting `best.pt`.
3. **Labelling rule:** a node is labelled an intersection if it appears in ≥2 OSM ways. A way-count is not the same as junction degree: it can mark simple road continuations (one road split into two OSM ways) as junctions. The query also includes footways and service roads.
4. **Random tile split:** neighbouring tiles share roads at their edges, so train and validation are not spatially independent.

The detector should be treated as an experimental baseline, not an accurate model.

---

## Approach B — Graph-based pipeline (planned)

```
Satellite image → preprocessing → U-Net road segmentation → road mask
  → skeletonization → road graph → intersections (degree ≥ 3, after cleaning)
  → georeferencing → OSM validation → GeoJSON → map dashboard
```

Planned training data: DeepGlobe Road Extraction dataset (stored **outside** this repository).

---

## Setup (Windows)

```
python -m venv its-env
its-env\Scripts\Activate.ps1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

Tested with Python 3.13.2, PyTorch 2.6.0 (CUDA 12.4 build), NVIDIA RTX 4050 Laptop GPU (6 GB).

---

## Repository structure

```
road-intersection-detection/
├── scripts/            YOLO baseline scripts
├── data/               OSM intersections, YOLO config, prediction samples
│                       (tiles, datasets and splits are git-ignored)
├── docs/
│   ├── PROJECT_LOG.md  milestone-by-milestone decisions, findings and results
│   └── results/        small result files copied from training runs
├── requirements.txt
└── README.md
```

Folders for the graph pipeline (`segmentation/`, `graph_extraction/`, `geospatial/`, `dashboard/`) will be added as each stage is built.

---

## Data sources and attribution

- **Imagery:** Esri World Imagery — © Esri and its data providers. Tiles are not redistributed in this repository.
- **Road data:** © OpenStreetMap contributors, available under the Open Database License (ODbL). `data/intersections.csv` is derived from OSM data.
- **Detector:** Ultralytics YOLO (AGPL-3.0).
