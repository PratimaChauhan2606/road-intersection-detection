"""
Turn the road probability map into a georeferenced road network.

Input  (from segmentation/predict_mosaic.py): prob.png, meta.json, mosaic.jpg
Output (in the same folder):
    roads.geojson          road segments (LineString) with length_m and width_m
    intersections.geojson  junction points with degree and type
    stats.json             network statistics + the cleaning log (for your report)
    mask_overlay.png       transparent red road mask (for the map)
    network_preview.jpg    mosaic with graph drawn on it (for slides)

Run from the repository root:
    python graph_extraction/build_network.py
"""
import argparse
import json
import math
import os
import sys

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "geospatial"))

from geo import mosaic_bounds, mosaic_mpp, px_to_lonlat  # noqa: E402
from network import extract_network, junction_type  # noqa: E402


def lonlat(meta, y, x):
    lon, lat = px_to_lonlat(x, y, meta["x0"], meta["y0"], meta["z"])
    return [round(lon, 7), round(lat, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/gandhinagar", help="Folder with prob.png and meta.json")
    ap.add_argument("--thr", type=float, default=0.5, help="Probability threshold (use the best one from evaluate.py)")
    ap.add_argument("--close-m", type=float, default=1.1, help="Gap-closing radius in metres")
    ap.add_argument("--min-area-m2", type=float, default=120.0, help="Drop road blobs smaller than this")
    ap.add_argument("--spur-m", type=float, default=15.0, help="Prune dead-end branches shorter than this")
    ap.add_argument("--gap-m", type=float, default=20.0, help="Join facing dead ends closer than this (0 = off)")
    ap.add_argument("--merge-m", type=float, default=10.0, help="Merge junctions closer than this")
    ap.add_argument("--roundabout-m", type=float, default=120.0, help="Max perimeter of a cycle treated as a roundabout")
    ap.add_argument("--simplify-m", type=float, default=1.1, help="Douglas-Peucker tolerance")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)
    prob = cv2.imread(os.path.join(args.dir, "prob.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mpp = mosaic_mpp(meta)
    west, south, east, north = mosaic_bounds(meta)
    print(f"Mosaic {meta['width']}x{meta['height']} px | {mpp:.3f} m/pixel | "
          f"area {(meta['width'] * mpp) * (meta['height'] * mpp) / 1e6:.2f} km2")

    px = lambda m: max(1, int(round(m / mpp)))  # metres -> pixels
    mask, skel, G, log = extract_network(
        prob, thr=args.thr, close_radius=px(args.close_m), min_area_px=int(args.min_area_m2 / mpp ** 2),
        fill_hole_px=int(40 / mpp ** 2), junction_radius=3, spur_px=px(args.spur_m),
        merge_px=px(args.merge_m), roundabout_px=px(args.roundabout_m), simplify_px=args.simplify_m / mpp,
        gap_px=(px(args.gap_m) if args.gap_m > 0 else 0))
    print("Cleaning log:", json.dumps(log))

    # road width from the distance transform of the mask
    edt = ndi.distance_transform_edt(mask)
    H, W = mask.shape

    road_feats, total_len_m = [], 0.0
    for i, (u, v, k, d) in enumerate(G.edges(keys=True, data=True)):
        coords = d["coords"]
        length_m = d["length_px"] * mpp
        ys = np.clip(np.round([c[0] for c in coords]).astype(int), 0, H - 1)
        xs = np.clip(np.round([c[1] for c in coords]).astype(int), 0, W - 1)
        width_m = float(2.0 * np.median(edt[ys, xs]) * mpp)
        total_len_m += length_m
        road_feats.append({"type": "Feature", "properties": {
            "id": i, "u": int(d["u"]), "v": int(d["v"]), "length_m": round(length_m, 1), "width_m": round(width_m, 1)},
            "geometry": {"type": "LineString", "coordinates": [lonlat(meta, y, x) for y, x in coords]}})

    inter_feats, by_type = [], {}
    for n, data in G.nodes(data=True):
        deg = G.degree(n)
        if deg >= 3:
            t = junction_type(deg)
            by_type[t] = by_type.get(t, 0) + 1
            y, x = data["pos"]
            inter_feats.append({"type": "Feature", "properties": {
                "id": int(n), "degree": int(deg), "type": t, "roundabout": bool(data.get("roundabout", False))},
                "geometry": {"type": "Point", "coordinates": lonlat(meta, y, x)}})

    degs = [dd for _, dd in G.degree()]
    simple = nx.Graph(G)
    area_km2 = (meta["width"] * mpp) * (meta["height"] * mpp) / 1e6
    comps = [len(c) for c in nx.connected_components(simple)] if G.number_of_nodes() else []
    stats = {
        "mosaic_area_km2": round(area_km2, 3),
        "metres_per_pixel": round(mpp, 4),
        "probability_threshold": args.thr,
        "total_road_length_km": round(total_len_m / 1000, 3),
        "road_density_km_per_km2": round(total_len_m / 1000 / area_km2, 2),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "intersections_degree_ge3": len(inter_feats),
        "intersections_by_type": by_type,
        "roundabouts_collapsed": log["roundabouts_collapsed"],
        "dead_ends": int(sum(1 for dd in degs if dd == 1)),
        "mean_node_degree": round(float(np.mean(degs)), 2) if degs else 0,
        "connected_components": len(comps),
        "largest_component_share_of_nodes": round(max(comps) / G.number_of_nodes(), 3) if comps else 0,
        "cleaning_log": log,
        "parameters": vars(args),
        "bounds_west_south_east_north": [west, south, east, north],
    }

    out = args.dir
    with open(os.path.join(out, "roads.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": road_feats}, f)
    with open(os.path.join(out, "intersections.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": inter_feats}, f)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    ov = np.zeros((H, W, 4), dtype=np.uint8)
    ov[mask] = (255, 40, 40, 150)  # BGRA
    cv2.imwrite(os.path.join(out, "mask_overlay.png"), ov)

    mos_path = os.path.join(out, "mosaic.jpg")
    if os.path.exists(mos_path):
        img = cv2.imread(mos_path)
        for u, v, k, d in G.edges(keys=True, data=True):
            pts = np.array([[c[1], c[0]] for c in d["coords"]], dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
        for n, data in G.nodes(data=True):
            y, x = data["pos"]
            deg = G.degree(n)
            if deg >= 3:
                cv2.circle(img, (int(x), int(y)), 7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out, "network_preview.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("\n=== NETWORK SUMMARY ===")
    for key in ["total_road_length_km", "road_density_km_per_km2", "nodes", "edges", "intersections_degree_ge3",
                "intersections_by_type", "dead_ends", "mean_node_degree", "connected_components",
                "largest_component_share_of_nodes"]:
        print(f"{key:36s}: {stats[key]}")
    print(f"\nSaved roads.geojson, intersections.geojson, stats.json, mask_overlay.png, network_preview.jpg to {out}")


if __name__ == "__main__":
    main()
