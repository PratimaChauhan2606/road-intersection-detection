# Project Log — Road Intersection Detection (ITS)

This log records decisions, findings, errors and results, milestone by milestone.
Status words used: **Done** (implemented and checked), **In progress**, **Planned** (not started).

---

## Environment (verified 2026-11-04)

| Item | Value |
|---|---|
| OS / editor | Windows, VS Code, PowerShell |
| Python | 3.13.2 (venv `its-env`, inside the project folder) |
| PyTorch | 2.6.0+cu124 — `torch.cuda.is_available()` = True |
| torchvision | 0.21.0+cu124 |
| Ultralytics | 8.4.153 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM, driver 592.82 |

Note: `nvidia-smi` shows "CUDA 13.1" = the maximum CUDA version the driver supports. PyTorch ships its own CUDA 12.4 runtime, so no separate CUDA Toolkit is needed.

---

## Milestone 1 — Inspect existing repository (Done, 2026-11-04)

### Repository state found
- 4 commits on `main`, latest `2dfb9c0`. No README.
- Root `.gitignore` was empty; ignore rules were in `scripts/.gitignore`, where they only apply inside `scripts/`. Result: venv, `runs/`, datasets and `*.pt` were not ignored.
- `requirements.txt` was saved as UTF-16 (PowerShell `>` redirect), so GitHub showed it as binary.
- Uncommitted local change in `scripts/get_osm_intersections.py`: threshold `count >= 3` → `count >= 2`. The committed `data/intersections.csv` (1,190 points) was produced after this change.

### Data counts (verified)
| Item | Count |
|---|---:|
| Downloaded tiles (`data/tiles`) | 272 |
| OSM intersection points (`intersections.csv`) | 1,190 |
| Image-label pairs (`data/dataset`) | 221 |
| `dataset_split` train / val | 182 / 53 |
| Images present in BOTH train and val | **14** |

### YOLO runs (from each run's `results.csv`, best mAP@0.5 row)
| Run | Epochs | Best epoch | P | R | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| intersection_detector | 50 | 50 | 0.002 | 0.50 | 0.004 |
| intersection_detector_v2 | 130 | 114 | 0.003 | 0.80 | 0.010 |
| **intersection_detector_v2-2** (baseline) | 95 | 42 | 0.292 | 0.266 | 0.214 |

Interpretation: runs 1 and v2 effectively failed. Their recall values of exactly 0.50 and 0.80 suggest very few labelled intersections in validation, consistent with the old `>= 3` rule (not directly verified). v2-2 is the baseline.

### Findings / known issues
1. **Leakage:** 14 of 53 val images also in train — `split_dataset.py` was run twice without clearing its output folder. Baseline metrics are likely optimistic.
2. **No test set:** val was also used for early stopping / `best.pt` selection.
3. **Labelling rule counts OSM ways, not junction degree.**
   - `>= 3` ways missed crossings of through-roads and T-junctions on a through-road.
   - `>= 2` ways (current) catches those, but also marks road continuations (one road stored as two ways) as junctions.
   - Correct rule: degree = number of road segments touching the node (interior node of a way +2, end node +1); junction if degree ≥ 3. Also filter non-drivable `highway` types.
4. Random tile split is not spatially independent.

### Decisions
- Keep YOLO code and data paths exactly as they are (scripts hard-code `data/...` paths).
- Report YOLO v1 honestly with the caveats above.
- Fix the OSM labelling rule and re-run YOLO on a clean, spatially separated split in Milestone 10, so the YOLO vs. graph comparison is fair.

---

## Milestone 2 — Git hygiene + environment record (In progress)

- [x] 2.1 Root `.gitignore` written; `git check-ignore` confirms venv, `runs/`, datasets, tiles and `*.pt` are ignored.
- [x] 2.2 `requirements.txt` rebuilt by hand (UTF-8, pinned, only directly used packages; `overpy` removed as unused). `pip install -r requirements.txt --dry-run` → all already satisfied.
- [x] 2.3 README v1, this log, YOLO v1 result files copied to `docs/results/yolo_v1/`.
- [x] 2.4 Reviewed and committed (90bdcaa).

---

## Milestones 3 to 9: Graph pipeline (Done, 2026-11-16)

- **Data:** DeepGlobe Road Extraction (Kaggle), 6,226 image/mask pairs with masks, 1024x1024. Kept outside the repo. Split by image ID (4980 / 623 / 623, seed 42) before cropping, to avoid leakage.
- **Model:** U-Net, ResNet-34 ImageNet encoder, BCE + Dice loss, AdamW 3e-4, cosine schedule, AMP, 512x512 random crops, batch 4, flips / rot90 / brightness augmentation. 30 epochs, best epoch 28, about 2.7 min per epoch, peak GPU memory 1.24 GB.
- **Segmentation result:** val (623) threshold 0.45: IoU 0.631, P 0.764, R 0.783, F1 0.774. Test (623, unseen): IoU 0.634, P 0.771, R 0.781, F1 0.776. Pixel accuracy was not used (background dominates).
- **Georeferencing:** Web-Mercator tile maths, Gandhinagar mosaic 16x17 tiles at zoom 18 (4096x4352 px, 0.549 m/px, 5.37 km2).
- **Graph extraction:** clean mask, skeleton, own skeleton-to-graph code, spur pruning, degree-2 contraction, gap bridging, junction merging, roundabout collapse, simplification. Junction = node of degree >= 3.
- **OSM validation:** degree rule on OSM, drivable classes only, parking aisles / driveways / drive-throughs / areas skipped, reference junctions merged within 12 m, one-to-one matching at 10 / 15 / 20 m. Road-length overlap with 5 m buffer.

## Milestone 10: Findings and decisions (Done, 2026-11-16)

1. **Old YOLO labels:** 1,102 of the 1,190 points (93%) are real junctions by the degree rule (81% excluding service roads). So the `>= 2` way rule was mostly fine for this area. The YOLO model was not re-run on a clean split.
2. **Refuted hypothesis:** I first thought service lanes explained the low junction recall. Excluding service and living_street roads left recall almost unchanged (about 0.35), so that was not the main cause. Missing narrow and tree-covered roads is.
3. **Parameter sweep (7 configurations, tuned on the evaluation area itself, so optimistic):**

| Config | Junctions | J precision | J recall | J F1 | Road quality |
|---|---:|---:|---:|---:|---:|
| 1 default (thr 0.45) | 355 | 0.785 | 0.272 | 0.404 | 0.515 |
| 2 thr 0.30 | 387 | 0.792 | 0.299 | 0.434 | 0.528 |
| 4 thr 0.30, gap 35, spur 8 | 403 | 0.788 | 0.307 | 0.442 | 0.529 |
| 5 as 4, merge 6 | 418 | 0.762 | 0.308 | 0.438 | 0.528 |
| **6 thr 0.20, gap 35, spur 8 (chosen)** | 441 | 0.788 | 0.337 | 0.472 | 0.547 |
| 7 thr 0.30, close 2 | 379 | 0.836 | 0.308 | 0.450 | 0.513 |

(Config 3 matched config 2 in junction scores.) Metrics at 15 m tolerance.
4. **Final settings:** thr 0.20, gap 35 m, spur 8 m, merge 10 m, roundabout perimeter 120 m, close 1.1 m, min area 120 m2, simplify 1.1 m, flip TTA.
5. **Final network:** 72.658 km, 746 nodes, 852 edges, 441 junctions (369 / 68 / 4), 14 roundabouts, 305 dead ends, 29 components.
6. **Final validation:** junction P / R / F1 at 15 m = 0.788 / 0.337 / 0.472; road completeness 0.585, correctness 0.894, quality 0.547.
7. **Dashboard:** Leaflet inlined, shortest route over the extracted graph. Bug found and fixed: road and junction layers swallowed map clicks, so a second click never set the destination. Popups became hover tooltips; verified with real mouse clicks.
8. **Laptop overheating** (GPU 85 to 89 C, burning smell) during training. Training finished; not repeated on this laptop.
9. **YOLO vs graph comparison:** script written (`geospatial/compare_yolo.py`), **not run** (time). No claim that either approach is better; YOLO mAP and graph F1 are different metrics.

## Milestone 11: Release (In progress)

- [ ] Copy small results into `docs/results/graph_v1/` and the dashboard into `docs/dashboard/`
- [ ] README with results, commit and push
- [ ] Presentation slides

## Open issues (carry forward)
- [ ] Re-run YOLO on a clean spatial split with a held-out test area (not done)
- [ ] Run the same-protocol YOLO vs graph comparison (optional, not done)
- [ ] Test on a second city (not done); settings were tuned on Gandhinagar
- [ ] Not done from the proposal: D-LinkNet, clDice, APLS, Massachusetts cross-dataset test, FastAPI
- [ ] DeepGlobe is research-licence only; Esri imagery has its own terms; commercial use would need other data
- [ ] Proposal states RTX 3050; actual GPU is RTX 4050. Correct in the final report
