# Road Intersection Detection from Satellite Imagery

**Course:** Intelligent Transportation Systems (ITS), Gati Shakti Vishwavidyalaya (GSV)
**Project:** DL-Based Road Feature Extraction from Satellite Imagery. Feature of interest: Road Intersection Detection via Road Network Graph Extraction

---

## What this project does

It turns a satellite image into a **road network graph with junctions and real-world coordinates**, then checks the result against OpenStreetMap (OSM).

```
Satellite tiles -> U-Net road segmentation -> road mask -> skeleton -> road graph
  -> junctions (graph nodes with degree >= 3, after cleaning)
  -> georeferenced GeoJSON -> OSM validation -> interactive map dashboard
```

The dashboard (`docs/dashboard/dashboard.html`) shows the satellite image, the road mask, the extracted roads, the junctions (coloured by 3-way / 4-way / complex), the OSM check (matched / extra / missed junctions) and a shortest-route tool that runs on the **extracted** graph (not on Google or OSM routing).

A YOLO intersection detector (Approach A, below) was built first as a baseline.

**This is a research prototype tested on one study area (Gandhinagar). It is not a production mapping tool.**

---

## Results (Approach B, graph pipeline)

### 1. Road segmentation (U-Net, ResNet-34 encoder, DeepGlobe)

Trained 30 epochs (best epoch 28). Image-level split 4980 / 623 / 623 made **before** cropping, so no tile appears in two splits. Threshold 0.45 chosen on validation.

| Split | Images | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 623 | 0.631 | 0.764 | 0.783 | 0.774 |
| **Test (unseen)** | 623 | **0.634** | 0.771 | 0.781 | 0.776 |

### 2. Road network on Gandhinagar (5.37 km2, zoom 18, 0.55 m/pixel)

The model never saw Gandhinagar during training.

| Item | Value |
|---|---|
| Road length | 72.7 km (13.5 km/km2) |
| Graph | 746 nodes, 852 edges, 29 connected components (largest: 91.4% of nodes) |
| Junctions (degree >= 3) | 441: 369 three-way, 68 four-way, 4 five-way or more |
| Roundabouts collapsed to one node | 14 |
| Dead ends | 305 |

### 3. Validation against OpenStreetMap

Reference: OSM drivable roads, junction = node of degree >= 3 (1,007 reference junctions after merging points within 12 m). Greedy one-to-one matching.

| Tolerance | Precision | Recall | F1 |
|---|---:|---:|---:|
| 10 m | 0.753 | 0.322 | 0.451 |
| 15 m | 0.788 | 0.337 | 0.472 |
| 20 m | 0.809 | 0.346 | 0.484 |

Mean location error of matched junctions: 4.2 to 4.9 m.

Road length overlap (5 m buffer): completeness 0.585, correctness 0.894, quality 0.547.
Completeness by OSM class: residential 0.560 (62.5 km), tertiary 0.711 (22.3 km), secondary 0.754 (5.9 km), service 0.409 (13.9 km).

### How to read these numbers

- **Precision is much higher than recall.** About 79% of predicted junctions match an OSM junction, but only about 34% of OSM junctions are found. The pipeline misses narrow or tree-covered lanes (only 56% of residential road length is found), and every missed road removes its junctions.
- **OSM is not perfect ground truth.** It is incomplete and has its own mapping choices, so some "false positives" may be real junctions that OSM lacks, and some "misses" may not be visible in the image.
- **The settings were tuned on this same area** (7-configuration sweep: threshold, gap bridging, spur pruning, merge distance, kept in `docs/results/graph_v1/sweep_table.json`). The default settings gave 355 junctions, junction F1 0.404 and road quality 0.515. The final settings (threshold 0.20, gap 35 m, spur 8 m) are therefore optimistic for this area. Behaviour on a new city is untested.
- **Segmentation numbers (DeepGlobe test) and Gandhinagar numbers are different tests** and must not be mixed.

### Known limitations

- Low junction recall (about 34%); narrow and tree-covered roads are missed.
- Medians on divided roads can create small loops and extra junctions.
- Overpasses are treated as flat (2D), so a bridge crossing can appear as a junction.
- Only one OSM-checked area; no cross-city test.
- Not done from the original proposal: D-LinkNet, clDice loss, APLS metric, Massachusetts cross-dataset test, FastAPI service.
- YOLO baseline (Approach A) has data leakage and no test set, and its mAP is **not comparable** with the graph F1 scores above. No claim is made that either approach is better. A same-protocol comparison script (`geospatial/compare_yolo.py`) exists but has not been run.

---

## Reproduce Approach B

Datasets and weights are **not** in this repository. Download the DeepGlobe Road Extraction dataset yourself (Kaggle; research licence) and unzip it outside the repo (we used `C:\datasets\deepglobe`). Run all commands from the repository root.

```
python segmentation/prepare_split.py --data-dir <deepglobe train folder>
python segmentation/train_unet.py --data-dir <deepglobe train folder> --run-name unet_r34_v1 --epochs 30
python segmentation/evaluate.py --data-dir <deepglobe train folder> --split val
python segmentation/evaluate.py --data-dir <deepglobe train folder> --split test --threshold 0.45
python segmentation/predict_mosaic.py --tta                  # needs data/tiles from download_tiles.py
python graph_extraction/build_network.py --thr 0.20 --gap-m 35 --spur-m 8
python geospatial/validate_osm.py                            # fetches OSM (needs internet)
python geospatial/validate_roads.py
python geospatial/make_dashboard.py                          # writes outputs/gandhinagar/dashboard.html
```

Check each script's `--help` for exact argument names. Training took about 2.7 min per epoch on an RTX 4050 Laptop GPU (6 GB), peak 1.2 GB. The laptop ran hot (85 to 89 C); use a cooling pad or a better machine.

---

## Approach A â€” YOLO baseline (v1, baseline)

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

Study area: Gandhinagar sector grid, bounding box 72.63â€“72.65 E, 23.21â€“23.23 N.

### Result (validation set)

| Run | Best epoch | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|---:|
| `intersection_detector_v2-2` | 42 of 95 | 0.292 | 0.266 | 0.214 |

Values are the best-mAP@0.5 row of `results.csv`. Training curves and the confusion matrix are in `docs/results/yolo_v1/`.

### Known issues (to be fixed in a later milestone)

1. **Data leakage:** 14 of the 53 validation images also appear in the training folder, because `split_dataset.py` was run twice without emptying its output folder. The figures above are therefore likely optimistic.
2. **No held-out test set:** the validation set was also used for early stopping and selecting `best.pt`.
3. **Labelling rule:** a node is labelled an intersection if it appears in â‰¥2 OSM ways. A way-count is not the same as junction degree: it can mark simple road continuations (one road split into two OSM ways) as junctions. The query also includes footways and service roads.
4. **Random tile split:** neighbouring tiles share roads at their edges, so train and validation are not spatially independent.

The detector should be treated as an experimental baseline, not an accurate model.

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
â”œâ”€â”€ scripts/            YOLO baseline scripts (Approach A)
â”œâ”€â”€ segmentation/       U-Net data split, training, evaluation, mosaic prediction
â”œâ”€â”€ graph_extraction/   skeleton -> graph, cleaning, junctions
â”œâ”€â”€ geospatial/         georeferencing, OSM validation, dashboard builder, sweep, YOLO comparison
â”œâ”€â”€ data/               OSM intersections, YOLO config, prediction samples
â”‚                       (tiles, datasets and splits are git-ignored)
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ PROJECT_LOG.md  decisions, findings, errors and results
â”‚   â”œâ”€â”€ BUSINESS_SLIDE.md
â”‚   â”œâ”€â”€ dashboard/      the interactive map (open dashboard.html in Chrome or Edge)
â”‚   â””â”€â”€ results/        small result files (metrics, GeoJSON, sweep table, YOLO v1 curves)
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## Data sources and attribution

- **Imagery:** Esri World Imagery, (c) Esri and its data providers. Tiles are not redistributed here. Terms of use apply.
- **Road data:** (c) OpenStreetMap contributors, Open Database License (ODbL). Used as the validation reference and for YOLO labels.
- **Training data:** DeepGlobe Road Extraction dataset (Demir et al., CVPRW 2018). Research-use licence; not redistributed here.
- **Model code:** segmentation-models-pytorch (U-Net, ResNet-34 ImageNet encoder); Ultralytics YOLO (AGPL-3.0) for the baseline.
- **Dashboard:** Leaflet 1.9.4 (BSD-2-Clause), bundled in `geospatial/vendor/`.

Because DeepGlobe and Esri imagery have research or restricted terms, any commercial use would need commercially licensed imagery and training data.
