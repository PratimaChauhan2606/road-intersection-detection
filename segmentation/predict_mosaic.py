"""
Stitch the Gandhinagar tiles into one big image and predict road probabilities.

Why a mosaic? A road crossing a tile edge is cut in two if each 256 px tile is
predicted alone. Stitching first keeps roads continuous and lets the graph
stage see real junctions.

Outputs (in --out):
    mosaic.jpg   the stitched satellite image
    prob.png     road probability, 0-255 (255 = certain road)
    meta.json    tile index of top-left tile, zoom, size, missing tiles

Run from the repository root:
    python segmentation/predict_mosaic.py --tta
"""
import argparse
import glob
import json
import os
import re
import time

import cv2
import numpy as np
import torch

from common import load_model, predict_probs

TILE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", default="data/tiles")
    ap.add_argument("--checkpoint", default="outputs/segmentation/unet_r34_v1/best.pt")
    ap.add_argument("--out", default="outputs/gandhinagar")
    ap.add_argument("--tta", action="store_true", help="Average predictions over flips (4x slower, usually smoother)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pat = re.compile(r"^(\d+)_(\d+)_(\d+)\.png$")
    tiles = {}
    for p in glob.glob(os.path.join(args.tiles_dir, "*.png")):
        m = pat.match(os.path.basename(p))
        if m:
            z, x, y = map(int, m.groups())
            tiles[(z, x, y)] = p
    if not tiles:
        raise SystemExit(f"No tiles named <z>_<x>_<y>.png in {args.tiles_dir}")
    zooms = {k[0] for k in tiles}
    if len(zooms) != 1:
        raise SystemExit(f"Mixed zoom levels found: {zooms}")
    z = zooms.pop()
    xs = [k[1] for k in tiles]
    ys = [k[2] for k in tiles]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx_, ny_ = x1 - x0 + 1, y1 - y0 + 1
    print(f"Tiles: {len(tiles)} | zoom {z} | grid {nx_} x {ny_} = {nx_ * ny_} slots")

    mosaic = np.zeros((ny_ * TILE, nx_ * TILE, 3), dtype=np.uint8)
    missing = []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            p = tiles.get((z, tx, ty))
            if p is None:
                missing.append([tx, ty])
                continue
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (TILE, TILE):
                img = cv2.resize(img, (TILE, TILE))
            mosaic[(ty - y0) * TILE:(ty - y0 + 1) * TILE, (tx - x0) * TILE:(tx - x0 + 1) * TILE] = img
    print(f"Mosaic size: {mosaic.shape[1]} x {mosaic.shape[0]} px | missing tiles (left black): {len(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Model epoch {ckpt.get('epoch')} on {device}")

    t0 = time.time()
    variants = [(False, False)] + ([(True, False), (False, True), (True, True)] if args.tta else [])
    acc = np.zeros(mosaic.shape[:2], dtype=np.float32)
    for fl_h, fl_v in variants:
        im = mosaic
        if fl_h:
            im = im[:, ::-1]
        if fl_v:
            im = im[::-1]
        pr = predict_probs(model, np.ascontiguousarray(im), device)
        if fl_v:
            pr = pr[::-1]
        if fl_h:
            pr = pr[:, ::-1]
        acc += pr
    prob = acc / len(variants)
    print(f"Prediction done in {(time.time() - t0) / 60:.1f} min ({len(variants)} pass(es))")

    cv2.imwrite(os.path.join(args.out, "mosaic.jpg"), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(args.out, "prob.png"), (prob * 255).astype(np.uint8))
    meta = {"z": z, "x0": x0, "y0": y0, "width": int(mosaic.shape[1]), "height": int(mosaic.shape[0]),
            "n_tiles": len(tiles), "missing_tiles": missing, "checkpoint": args.checkpoint,
            "checkpoint_epoch": ckpt.get("epoch"), "tta": bool(args.tta)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved mosaic.jpg, prob.png, meta.json to {args.out}")
    print(f"Mean road probability: {prob.mean():.3f} | pixels >= 0.5: {(prob >= 0.5).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
