"""
Try several network-extraction settings and print ONE comparison table.

For each setting it runs build_network.py, validate_osm.py and validate_roads.py into
its own folder (outputs/sweep/<name>), so your main outputs are not touched.

HONESTY NOTE: Gandhinagar is the only area with an OSM check. Choosing the best row here
and then reporting its score on the same area makes that score optimistic. In your report,
show the WHOLE table (including the default row) and say the settings were tuned on the
same area.

Run from the repository root:
    python geospatial/tune_sweep.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

CONFIGS = [
    ("1_default",            ["--thr", "0.45", "--gap-m", "20", "--spur-m", "15", "--merge-m", "10"]),
    ("2_thr030",             ["--thr", "0.30", "--gap-m", "20", "--spur-m", "15", "--merge-m", "10"]),
    ("3_thr030_gap35",       ["--thr", "0.30", "--gap-m", "35", "--spur-m", "15", "--merge-m", "10"]),
    ("4_thr030_gap35_spur8", ["--thr", "0.30", "--gap-m", "35", "--spur-m", "8", "--merge-m", "10"]),
    ("5_..._merge6",         ["--thr", "0.30", "--gap-m", "35", "--spur-m", "8", "--merge-m", "6"]),
    ("6_thr020_gap35_spur8", ["--thr", "0.20", "--gap-m", "35", "--spur-m", "8", "--merge-m", "10"]),
    ("7_thr030_close2",      ["--thr", "0.30", "--gap-m", "35", "--spur-m", "8", "--merge-m", "10", "--close-m", "2.2"]),
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(r.stdout[-800:], r.stderr[-800:])
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/gandhinagar")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--osm-cache", default="outputs/osm_gandhinagar.json")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    py = sys.executable
    rows = []
    for name, flags in CONFIGS:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for fn in ("prob.png", "meta.json"):
            shutil.copy(os.path.join(args.base, fn), d)
        print(f"Running {name} ...", flush=True)
        run([py, os.path.join(root, "graph_extraction", "build_network.py"), "--dir", d] + flags)
        run([py, os.path.join(here, "validate_osm.py"), "--dir", d, "--osm-cache", args.osm_cache, "--old-csv", "none"])
        run([py, os.path.join(here, "validate_roads.py"), "--dir", d, "--osm-cache", args.osm_cache])
        st = json.load(open(os.path.join(d, "stats.json")))
        vj = json.load(open(os.path.join(d, "validation_results.json")))["tolerances"]["15.0"]
        ro = json.load(open(os.path.join(d, "road_overlap.json")))["groups"]["all drivable roads"]
        rows.append((name, st["intersections_degree_ge3"], st["connected_components"], st["total_road_length_km"],
                     vj["precision"], vj["recall"], vj["f1"], ro["completeness"], ro["correctness"], ro["quality"]))

    print("\nJunctions matched within 15 m; road overlap within 5 m; reference = all drivable OSM roads")
    print(f"{'setting':22s} {'junct':>6} {'parts':>6} {'km':>6} | {'J-prec':>6} {'J-rec':>6} {'J-F1':>6} | {'R-comp':>6} {'R-corr':>6} {'R-qual':>6}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:6d} {r[2]:6d} {r[3]:6.1f} | {r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f} | {r[7]:6.3f} {r[8]:6.3f} {r[9]:6.3f}")
    with open(os.path.join(args.out, "sweep_table.json"), "w") as f:
        json.dump([dict(zip(["setting", "junctions", "components", "km", "j_prec", "j_rec", "j_f1", "r_comp", "r_corr", "r_qual"], r)) for r in rows], f, indent=2)
    print(f"\nSaved {os.path.join(args.out, 'sweep_table.json')}")


if __name__ == "__main__":
    main()
