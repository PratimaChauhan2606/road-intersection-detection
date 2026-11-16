"""
OpenStreetMap reference junctions, computed with the DEGREE rule.

Why not "number of ways >= 3"?  One physical road is often stored as several OSM
ways (a new way starts when the name or speed limit changes), and a through road
that is crossed by another is ONE way. So counting ways misses real crossings and
can invent junctions. Instead we count road PIECES leaving each node:
    a node in the middle of a way contributes 2, a node at either end contributes 1.
A node with total degree >= 3 is a real junction. This is the same definition the
predicted road graph uses, so both sides of the comparison mean the same thing.

OpenStreetMap data (c) OpenStreetMap contributors, ODbL.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import requests
from scipy.spatial import cKDTree

DRIVABLE = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service", "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link"]
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through"}
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "ITS-student-project/1.0 (GSV road network extraction)"}


def fetch_osm(bbox, cache_path):
    """bbox = (west, south, east, north). Downloads once, then re-uses the cache file."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    west, south, east, north = bbox
    types = "|".join(DRIVABLE)
    q = f'[out:json][timeout:90];(way["highway"~"^({types})$"]({south},{west},{north},{east}););(._;>;);out body;'
    last = None
    for url in ENDPOINTS:
        try:
            print(f"Querying {url} ...")
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=150)
            r.raise_for_status()
            data = r.json()
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  failed: {e}")
            time.sleep(3)
    raise SystemExit(f"Could not reach Overpass: {last}")


def junctions_from_osm(data, exclude=()):
    """Return dict {node_id: (lon, lat, degree)} for nodes with degree >= 3 on drivable roads.

    exclude: highway classes to ignore (e.g. {'service', 'living_street'}) for a sensitivity analysis.
    """
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
    deg = defaultdict(int)
    n_ways = 0
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        if tags.get("highway") not in DRIVABLE or tags.get("highway") in exclude:
            continue
        if tags.get("service") in SKIP_SERVICE or tags.get("area") == "yes":
            continue
        refs = el["nodes"]
        if len(refs) < 2:
            continue
        n_ways += 1
        for i, r in enumerate(refs):
            deg[r] += 1 if i in (0, len(refs) - 1) else 2
        if refs[0] == refs[-1]:  # closed way (e.g. roundabout): start node touched by both ends
            pass
    out = {n: (nodes[n][0], nodes[n][1], d) for n, d in deg.items() if d >= 3 and n in nodes}
    return out, n_ways


def old_rule_points(data, min_ways=2):
    """The original (way-count) rule from scripts/get_osm_intersections.py, for comparison."""
    cnt = defaultdict(set)
    for el in data["elements"]:
        if el["type"] == "way":
            for r in el["nodes"]:
                cnt[r].add(el["id"])
    return {n for n, w in cnt.items() if len(w) >= min_ways}


def cluster_points(xy, ids, radius):
    """Merge points closer than `radius` (e.g. the two junctions of a dual carriageway). Returns centroids."""
    if len(xy) == 0:
        return np.zeros((0, 2)), []
    parent = list(range(len(xy)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in cKDTree(xy).query_pairs(radius):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = defaultdict(list)
    for i in range(len(xy)):
        groups[find(i)].append(i)
    cents = np.array([xy[g].mean(axis=0) for g in groups.values()])
    members = [[ids[i] for i in g] for g in groups.values()]
    return cents, members
