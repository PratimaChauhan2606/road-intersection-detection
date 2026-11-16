"""
Road-LENGTH validation against OpenStreetMap (complements the junction check).

Junction matching is harsh: one missed alley changes the count. This script asks a
simpler question - how much of the road network did we find, and how much of what we
found is really there? It draws both networks as 1-pixel lines on the mosaic grid and
checks overlap within a tolerance buffer:

    completeness = share of OSM road length that lies within T metres of a predicted road
    correctness  = share of predicted road length that lies within T metres of an OSM road
    quality      = TP / (TP + FP + FN)   (= 1 / (1/completeness + 1/correctness - 1))

Completeness is also reported per OSM road class, so you can see whether misses are the
main roads or small lanes. OSM is not perfect ground truth: unmapped roads count against
"correctness", so it is a lower bound.

Run from the repository root (after validate_osm.py has cached the OSM data):
    python geospatial/validate_roads.py
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from geo import lonlat_to_px, mosaic_mpp  # noqa: E402
from osm_junctions import DRIVABLE, SKIP_SERVICE  # noqa: E402


def draw_lines(shape, polylines):
    img = np.zeros(shape, dtype=np.uint8)
    for pts in polylines:
        arr = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], False, 1, 1)
    return img.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/gandhinagar")
    ap.add_argument("--osm-cache", default="outputs/osm_gandhinagar.json")
    ap.add_argument("--tol-m", type=float, default=5.0)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)
    mpp = mosaic_mpp(meta)
    shape = (meta["height"], meta["width"])
    r = max(1, int(math.ceil(args.tol_m / mpp)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    with open(args.osm_cache) as f:
        osm = json.load(f)
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm["elements"] if e["type"] == "node"}
    by_class = {}
    for e in osm["elements"]:
        if e["type"] != "way":
            continue
        t = e.get("tags", {})
        hw = t.get("highway")
        if hw not in DRIVABLE or t.get("service") in SKIP_SERVICE or t.get("area") == "yes":
            continue
        pts = [lonlat_to_px(*nodes[n], meta["x0"], meta["y0"], meta["z"]) for n in e["nodes"] if n in nodes]
        if len(pts) >= 2:
            by_class.setdefault(hw, []).append(pts)

    with open(os.path.join(args.dir, "roads.geojson")) as f:
        gj = json.load(f)
    pred_lines = [[lonlat_to_px(lo, la, meta["x0"], meta["y0"], meta["z"]) for lo, la in ft["geometry"]["coordinates"]]
                  for ft in gj["features"]]
    pred = draw_lines(shape, pred_lines)
    pred_buf = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)

    rasters = {hw: draw_lines(shape, pl) for hw, pl in by_class.items()}
    small = {"service", "living_street"}

    def score(osm_r):
        osm_buf = cv2.dilate(osm_r.astype(np.uint8), kernel).astype(bool)
        n_o, n_p = int(osm_r.sum()), int(pred.sum())
        comp = (osm_r & pred_buf).sum() / n_o if n_o else float("nan")
        corr = (pred & osm_buf).sum() / n_p if n_p else float("nan")
        qual = 1.0 / (1.0 / comp + 1.0 / corr - 1.0) if comp > 0 and corr > 0 else 0.0
        return comp, corr, qual, n_o * mpp / 1000.0

    groups = {
        "all drivable roads": np.logical_or.reduce(list(rasters.values())),
        "excluding service lanes": np.logical_or.reduce([v for k, v in rasters.items() if k not in small]),
    }
    print(f"Tolerance {args.tol_m:.0f} m ({r} px) | predicted road length {pred.sum() * mpp / 1000:.1f} km (pixel-count estimate)")
    print(f"\n{'OSM reference':28s} {'OSM km':>8} {'Complete':>9} {'Correct':>8} {'Quality':>8}")
    res = {"tolerance_m": args.tol_m, "groups": {}, "per_class_completeness": {}}
    for name, rr in groups.items():
        c, k, q, km = score(rr)
        res["groups"][name] = {"osm_km": km, "completeness": c, "correctness": k, "quality": q}
        print(f"{name:28s} {km:8.1f} {c:9.3f} {k:8.3f} {q:8.3f}")
    print(f"\n{'Completeness by OSM class':28s} {'OSM km':>8} {'found':>8}")
    for hw, rr in sorted(rasters.items(), key=lambda kv: -kv[1].sum()):
        n = int(rr.sum())
        c = (rr & pred_buf).sum() / n if n else float("nan")
        res["per_class_completeness"][hw] = {"osm_km": n * mpp / 1000.0, "completeness": float(c)}
        print(f"{hw:28s} {n * mpp / 1000.0:8.1f} {c:8.3f}")
    with open(os.path.join(args.dir, "road_overlap.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved road_overlap.json to {args.dir}")


if __name__ == "__main__":
    main()
