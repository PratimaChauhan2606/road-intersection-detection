"""
Validate predicted intersections against OpenStreetMap junctions.

Matching rule (distance tolerance): a predicted junction is a true positive if an
unused OSM junction lies within T metres; each OSM junction can match only once.
    precision = TP / predicted     recall = TP / OSM junctions     F1 = harmonic mean
We report several tolerances and the systematic offset between imagery and OSM.

IMPORTANT - how to read the numbers:
  * OSM is not perfect ground truth. Missing roads in OSM make correct predictions
    look like false positives; shifted geometry lowers every score.
  * Junctions within one tolerance of the study-area border are ignored on both sides
    (a junction cut by the border cannot be judged fairly).
  * The "offset-corrected" row removes the median shift using OSM itself, so it is a
    diagnostic, not a headline result.

Run from the repository root (needs internet the first time; result is cached):
    python geospatial/validate_osm.py
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from geo import mosaic_bounds, to_local_metres  # noqa: E402
from osm_junctions import cluster_points, fetch_osm, junctions_from_osm, old_rule_points  # noqa: E402


def match(pred, ref, tol):
    """Greedy one-to-one matching by increasing distance. Returns list of (i, j, dist)."""
    if len(pred) == 0 or len(ref) == 0:
        return []
    tree = cKDTree(ref)
    cand = []
    for i, p in enumerate(pred):
        for j in tree.query_ball_point(p, tol):
            cand.append((float(np.hypot(*(p - ref[j]))), i, j))
    cand.sort()
    up, ur, out = set(), set(), []
    for d, i, j in cand:
        if i not in up and j not in ur:
            up.add(i)
            ur.add(j)
            out.append((i, j, d))
    return out


def prf(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/gandhinagar")
    ap.add_argument("--osm-cache", default="outputs/osm_gandhinagar.json")
    ap.add_argument("--tols", default="10,15,20", help="Tolerances in metres, comma separated")
    ap.add_argument("--primary-tol", type=float, default=15.0)
    ap.add_argument("--merge-m", type=float, default=12.0, help="Merge OSM junctions closer than this (dual carriageways)")
    ap.add_argument("--exclude", default="", help="Highway classes to leave out of the reference, e.g. service,living_street")
    ap.add_argument("--tag", default="", help="Suffix for output files, e.g. main -> validation_main.json")
    ap.add_argument("--old-csv", default="data/intersections.csv", help="Old YOLO label points (optional)")
    args = ap.parse_args()

    tols = [float(t) for t in args.tols.split(",")]
    suffix = f"_{args.tag}" if args.tag else ""
    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)
    west, south, east, north = mosaic_bounds(meta)
    lon0, lat0 = (west + east) / 2, (south + north) / 2
    border = max(tols)

    data = fetch_osm((west, south, east, north), args.osm_cache)
    exclude = {e.strip() for e in args.exclude.split(',') if e.strip()}
    ref_all, n_ways = junctions_from_osm(data, exclude)
    if exclude:
        print(f"Reference excludes highway classes: {sorted(exclude)}")
    print(f"OSM: {n_ways} drivable ways, {len(ref_all)} junction nodes with degree >= 3")

    def inside(lon, lat):
        x, y = to_local_metres(lon, lat, lon0, lat0)
        xw, yw = to_local_metres(west, south, lon0, lat0)
        xe, yn = to_local_metres(east, north, lon0, lat0)
        return xw + border <= x <= xe - border and yw + border <= y <= yn - border

    ids = [n for n, (lo, la, d) in ref_all.items() if inside(lo, la)]
    ref_xy = np.array([to_local_metres(ref_all[n][0], ref_all[n][1], lon0, lat0) for n in ids]).reshape(-1, 2)
    n_raw = len(ref_xy)
    ref_xy, _ = cluster_points(ref_xy, ids, args.merge_m)
    print(f"Reference junctions inside study area (border {border:.0f} m removed): {n_raw} -> {len(ref_xy)} after merging within {args.merge_m:.0f} m")

    with open(os.path.join(args.dir, "intersections.geojson")) as f:
        gj = json.load(f)
    pred_ll = [f["geometry"]["coordinates"] for f in gj["features"]]
    keep = [i for i, (lo, la) in enumerate(pred_ll) if inside(lo, la)]
    pred_xy = np.array([to_local_metres(pred_ll[i][0], pred_ll[i][1], lon0, lat0) for i in keep]).reshape(-1, 2)
    print(f"Predicted junctions inside study area: {len(pred_xy)} (of {len(pred_ll)})")

    results = {"excluded_classes": sorted(exclude), "n_reference": int(len(ref_xy)), "n_predicted": int(len(pred_xy)), "tolerances": {}}
    print(f"\n{'tol(m)':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'mean err(m)':>12}")
    for t in tols:
        m = match(pred_xy, ref_xy, t)
        tp = len(m)
        p, r, f = prf(tp, len(pred_xy), len(ref_xy))
        err = float(np.mean([d for _, _, d in m])) if m else float("nan")
        results["tolerances"][str(t)] = {"tp": tp, "fp": len(pred_xy) - tp, "fn": len(ref_xy) - tp,
                                         "precision": p, "recall": r, "f1": f, "mean_error_m": err}
        print(f"{t:>7.0f} {tp:>5} {len(pred_xy) - tp:>5} {len(ref_xy) - tp:>5} {p:7.3f} {r:7.3f} {f:7.3f} {err:12.2f}")

    # systematic offset between imagery-derived and OSM geometry
    m_big = match(pred_xy, ref_xy, max(tols))
    if len(m_big) >= 5:
        off = np.median([ref_xy[j] - pred_xy[i] for i, j, _ in m_big], axis=0)
        results["median_offset_m"] = {"east": float(off[0]), "north": float(off[1]), "magnitude": float(np.hypot(*off))}
        print(f"\nMedian offset OSM - prediction: east {off[0]:+.1f} m, north {off[1]:+.1f} m (|offset| {np.hypot(*off):.1f} m) from {len(m_big)} matches")
        shifted = pred_xy + off
        results["offset_corrected"] = {}
        print("Offset-corrected (diagnostic only):")
        for t in tols:
            mm = match(shifted, ref_xy, t)
            p, r, f = prf(len(mm), len(pred_xy), len(ref_xy))
            results["offset_corrected"][str(t)] = {"precision": p, "recall": r, "f1": f}
            print(f"{t:>7.0f}  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    else:
        print("\nToo few matches to estimate an offset.")

    # how good was the ORIGINAL way-count labelling rule?
    if os.path.exists(args.old_csv):
        import csv
        old_ids = [int(r["node_id"]) for r in csv.DictReader(open(args.old_csv))]
        true_set = set(ref_all)
        ok = sum(1 for n in old_ids if n in true_set)
        results["old_yolo_labels"] = {"points": len(old_ids), "with_degree_ge3": ok}
        print(f"\nOriginal YOLO label points: {len(old_ids)} | of these, real junctions by the degree rule: {ok} ({ok / max(len(old_ids), 1) * 100:.0f}%)")
        print(f"Real junctions (degree>=3) in the whole OSM download: {len(ref_all)}  -> labels missed or mislabelled by the old rule")

    # map layer for the dashboard: TP / FP / FN at the primary tolerance
    m = match(pred_xy, ref_xy, args.primary_tol)
    mp, mr = {i for i, _, _ in m}, {j for _, j, _ in m}

    def ll(x, y):
        return [round(lon0 + x / (111320.0 * np.cos(np.radians(lat0))), 7), round(lat0 + y / 110574.0, 7)]

    feats = []
    for i, xy in enumerate(pred_xy):
        feats.append({"type": "Feature", "properties": {"status": "TP" if i in mp else "FP"},
                      "geometry": {"type": "Point", "coordinates": ll(*xy)}})
    for j, xy in enumerate(ref_xy):
        if j not in mr:
            feats.append({"type": "Feature", "properties": {"status": "FN"}, "geometry": {"type": "Point", "coordinates": ll(*xy)}})
    with open(os.path.join(args.dir, f"validation{suffix}.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    with open(os.path.join(args.dir, f"validation_results{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation{suffix}.geojson (TP/FP/FN at {args.primary_tol:.0f} m) and validation_results{suffix}.json to {args.dir}")


if __name__ == "__main__":
    main()
