"""
Step 3 of segmentation: evaluate a trained model.

1) On VAL: try several thresholds and pick the one with the best IoU.
       python segmentation/evaluate.py --data-dir "C:/datasets/deepglobe" --split val
2) On TEST: report final numbers using the threshold chosen on val.
       python segmentation/evaluate.py --data-dir "C:/datasets/deepglobe" --split test --threshold 0.5

Saves metrics JSON, example panels (image | ground truth | prediction) and,
optionally, the probability masks used later by the graph stage.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from common import (find_train_dir, load_model, metrics_from_counts, predict_probs,
                    read_ids, read_image, read_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", default="outputs/segmentation/unet_r34_v1/best.pt")
    ap.add_argument("--splits-dir", default="data/segmentation/splits")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold (use on test). If omitted, sweeps 0.30-0.70.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N images (0 = all)")
    ap.add_argument("--examples", type=int, default=8, help="How many example panels to save")
    ap.add_argument("--save-probs", action="store_true", help="Save probability masks as PNG (0-255)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    image_dir = find_train_dir(args.data_dir)
    ids = read_ids(os.path.join(args.splits_dir, f"{args.split}.txt"))
    if args.limit:
        ids = ids[: args.limit]

    thresholds = [args.threshold] if args.threshold is not None else [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    counts = {t: [0, 0, 0] for t in thresholds}

    run_dir = os.path.dirname(args.checkpoint)
    out_dir = os.path.join(run_dir, f"eval_{args.split}")
    ex_dir = os.path.join(out_dir, "examples")
    prob_dir = os.path.join(out_dir, "probs")
    os.makedirs(ex_dir, exist_ok=True)
    if args.save_probs:
        os.makedirs(prob_dir, exist_ok=True)

    t0 = time.time()
    for i, id_ in enumerate(ids, 1):
        img = read_image(os.path.join(image_dir, f"{id_}_sat.jpg"))
        gt = read_mask(os.path.join(image_dir, f"{id_}_mask.png")).astype(bool)
        prob = predict_probs(model, img, device)
        for t in thresholds:
            pred = prob >= t
            counts[t][0] += int((pred & gt).sum())
            counts[t][1] += int((pred & ~gt).sum())
            counts[t][2] += int((~pred & gt).sum())

        if args.save_probs:
            cv2.imwrite(os.path.join(prob_dir, f"{id_}_prob.png"), (prob * 255).astype(np.uint8))
        if i <= args.examples:
            t_show = thresholds[0] if len(thresholds) == 1 else 0.5
            g = np.repeat((gt * 255).astype(np.uint8)[..., None], 3, 2)
            p = np.repeat(((prob >= t_show) * 255).astype(np.uint8)[..., None], 3, 2)
            panel = np.concatenate([img, g, p], axis=1)
            panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
            cv2.imwrite(os.path.join(ex_dir, f"{id_}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        if i % 50 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)} images ({(time.time()-t0)/60:.1f} min)", flush=True)

    results = {str(t): metrics_from_counts(*counts[t]) for t in thresholds}
    best_t = max(thresholds, key=lambda t: results[str(t)]["iou"])
    print(f"\nSplit: {args.split} | images: {len(ids)}")
    print(f"{'thr':>5} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    for t in thresholds:
        r = results[str(t)]
        mark = "  <- best IoU" if (len(thresholds) > 1 and t == best_t) else ""
        print(f"{t:>5.2f} {r['iou']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}{mark}")

    summary = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"), "split": args.split,
               "n_images": len(ids), "results_by_threshold": results, "best_threshold": best_t}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics + examples to {out_dir}")
    if len(thresholds) > 1:
        print(f"Use --threshold {best_t} when you evaluate on the test split.")


if __name__ == "__main__":
    main()
