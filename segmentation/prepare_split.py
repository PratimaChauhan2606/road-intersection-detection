"""
Step 1 of segmentation: check the DeepGlobe dataset and split it BY IMAGE.

- Finds the training folder (only DeepGlobe 'train' has masks).
- Checks that every *_sat.jpg has a *_mask.png.
- Splits image IDs 80/10/10 into train/val/test (fixed seed), BEFORE any
  cropping, so crops of one image can never land in two splits.
- Writes the ID lists to data/segmentation/splits/ (small, committed to Git).
- Prints dataset statistics and saves a few image/mask examples to look at.

Run from the repository root:
    python segmentation/prepare_split.py --data-dir "C:/datasets/deepglobe"
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

from common import find_train_dir, list_pair_ids, read_image, read_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder where DeepGlobe was extracted")
    ap.add_argument("--out", default="data/segmentation/splits")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-sample", type=int, default=300, help="How many masks to read for statistics")
    ap.add_argument("--examples-dir", default="outputs/segmentation/data_check")
    args = ap.parse_args()

    train_dir = find_train_dir(args.data_dir)
    ids, sat_only, mask_only = list_pair_ids(train_dir)
    print(f"Training folder found : {train_dir}")
    print(f"Complete image/mask pairs: {len(ids)}")
    print(f"Images without mask      : {len(sat_only)}")
    print(f"Masks without image      : {len(mask_only)}")
    if not ids:
        raise SystemExit("No pairs found - stopping.")

    # ---- split by image ID
    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * args.test))
    n_val = int(round(n * args.val))
    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test:n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val:])
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) and not (set(val_ids) & set(test_ids))

    os.makedirs(args.out, exist_ok=True)
    for name, lst in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
    print(f"Split (seed {args.seed}): train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
    print("Overlap between splits: 0 (checked)")

    # ---- statistics on a sample of masks
    sample = rng.sample(ids, min(args.stats_sample, len(ids)))
    sizes, road_frac, raw_values = set(), [], set()
    for id_ in sample:
        img = read_image(os.path.join(train_dir, f"{id_}_sat.jpg"))
        raw = cv2.imread(os.path.join(train_dir, f"{id_}_mask.png"), cv2.IMREAD_GRAYSCALE)
        sizes.add(img.shape[:2])
        raw_values.update(np.unique(raw)[:20].tolist())
        road_frac.append(float((raw > 127).mean()))
    road_frac = np.array(road_frac)
    print(f"Image sizes seen         : {sorted(sizes)}")
    print(f"Raw mask values (sample) : {sorted(raw_values)[:20]}")
    print(f"Road pixels per image    : mean {road_frac.mean()*100:.2f}% | "
          f"median {np.median(road_frac)*100:.2f}% | min {road_frac.min()*100:.2f}% | max {road_frac.max()*100:.2f}%")

    info = {
        "train_dir": train_dir,
        "pairs": len(ids),
        "split_seed": args.seed,
        "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
        "image_sizes": [list(s) for s in sorted(sizes)],
        "road_fraction_mean": float(road_frac.mean()),
        "road_fraction_median": float(np.median(road_frac)),
        "stats_sample": len(sample),
    }
    with open(os.path.join(args.out, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---- save 4 examples: image | mask overlay
    os.makedirs(args.examples_dir, exist_ok=True)
    for id_ in sample[:4]:
        img = read_image(os.path.join(train_dir, f"{id_}_sat.jpg"))
        mask = read_mask(os.path.join(train_dir, f"{id_}_mask.png"))
        overlay = img.copy()
        overlay[mask == 1] = (0.4 * overlay[mask == 1] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        panel = np.concatenate([img, overlay], axis=1)
        panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
        cv2.imwrite(os.path.join(args.examples_dir, f"{id_}_check.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"Saved example images to  : {args.examples_dir}  (left: image, right: road mask in red)")


if __name__ == "__main__":
    main()
