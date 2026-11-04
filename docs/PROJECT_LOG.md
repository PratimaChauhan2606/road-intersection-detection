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
- [ ] 2.3 README v1, this log, YOLO v1 result files copied to `docs/results/yolo_v1/`.
- [ ] 2.4 Review and commit.

---

## Open issues (carry forward)
- [ ] Fix OSM junction rule (degree-based, drivable roads only) — Milestone 10
- [ ] Re-run YOLO on clean spatial split with a held-out test area — Milestone 10
- [ ] Check DeepGlobe licence / permitted use — Milestone 3
- [ ] Check Esri World Imagery terms for downloading tiles in an academic project — Milestone 3
- [ ] Proposal states RTX 3050; actual GPU is RTX 4050 — correct in final report
- [ ] Proposal lists D-LinkNet + clDice ablation as objectives — plan a reduced version or explain in report
