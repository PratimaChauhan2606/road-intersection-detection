"""
Road mask -> skeleton -> road graph -> cleaned graph -> intersections.

Pipeline (each step is a function):
    clean_mask        threshold, close small gaps, drop tiny blobs, fill tiny holes
    skeletonise       reduce thick road regions to 1-pixel centrelines
    skeleton_to_graph pixels -> nodes (junctions + dead ends) and edges (road pieces)
    prune_spurs       remove short dead-end branches (skeleton noise)
    contract_degree2  nodes with exactly 2 edges are just curves, not junctions
    merge_junctions   merge junction nodes that are very close; collapse roundabouts
    finalize          simplify edge geometry (Douglas-Peucker)

Node degree = number of road pieces meeting at the node.
Intersection = node with degree >= 3 (after cleaning).
"""
import math
from collections import defaultdict

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

S8 = np.ones((3, 3), dtype=bool)
K8 = np.ones((3, 3), dtype=np.uint8)
K8[1, 1] = 0
OFFS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ------------------------------------------------------------ mask cleaning
def clean_mask(prob, thr=0.5, close_radius=2, min_area_px=400, fill_hole_px=150):
    m = (prob >= thr).astype(np.uint8)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = m.astype(bool)
    # remove small isolated blobs
    lab, n = ndi.label(m, structure=S8)
    if n:
        areas = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = areas >= min_area_px
        m = keep[lab]
    # fill tiny holes (otherwise the skeleton draws a small useless loop around each)
    inv = ~m
    lab, n = ndi.label(inv)
    if n:
        areas = ndi.sum(inv, lab, index=np.arange(1, n + 1))
        small = np.zeros(n + 1, dtype=bool)
        small[1:] = areas < fill_hole_px
        m = m | small[lab]
    return m


def skeletonise(mask):
    return skeletonize(mask.astype(bool))


# ---------------------------------------------------------- graph building
def _shift_pairs(NL, dy, dx):
    H, W = NL.shape
    if dx >= 0:
        a, b = NL[0:H - dy, 0:W - dx], NL[dy:H, dx:W]
    else:
        a, b = NL[0:H - dy, -dx:W], NL[dy:H, 0:W + dx]
    sel = (a > 0) & (b > 0) & (a != b)
    return a[sel], b[sel]


def _order_pixels(pix, start_yx):
    """Walk the pixels of one road piece in order, starting from the end nearest start_yx."""
    P = {tuple(p) for p in pix}
    if len(P) == 1:
        return [next(iter(P))]

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in OFFS8 if (y + dy, x + dx) in P]

    ends = [p for p in P if len(nbrs(p)) == 1]
    cand = ends if ends else list(P)
    sy, sx = start_yx
    start = min(cand, key=lambda p: (p[0] - sy) ** 2 + (p[1] - sx) ** 2)
    path, seen, cur = [start], {start}, start
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))  # prefer straight steps
        cur = nxt[0]
        path.append(cur)
        seen.add(cur)
    return path


def _path_length(coords):
    return float(sum(math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
                     for i in range(len(coords) - 1)))


def skeleton_to_graph(skel, junction_radius=3):
    """Return a networkx MultiGraph. Node attr: pos=(y, x). Edge attrs: u, v, coords (u->v), length_px."""
    skel = skel.astype(bool)
    H, W = skel.shape
    nb = ndi.convolve(skel.astype(np.uint8), K8, mode="constant") * skel
    junc = skel & (nb >= 3)
    endp = skel & (nb == 1)

    # Junction pixels that are close together form ONE node (handles "duplicate junction pixels").
    dil = ndi.binary_dilation(junc, structure=S8, iterations=junction_radius)
    jl, nj = ndi.label(dil, structure=S8)
    jl = jl * skel
    ep = endp & (jl == 0)
    el, ne = ndi.label(ep, structure=S8)
    NL = jl.copy()
    NL[el > 0] = el[el > 0] + nj
    n_nodes = nj + ne

    G = nx.MultiGraph()
    if n_nodes:
        cms = ndi.center_of_mass(np.ones_like(NL, dtype=np.uint8), NL, index=np.arange(1, n_nodes + 1))
        for i, (cy, cx) in enumerate(cms, start=1):
            G.add_node(i, pos=(float(cy), float(cx)))

    # Road pieces = skeleton without node pixels
    seg = skel & (NL == 0)
    sl, ns = ndi.label(seg, structure=S8)
    ys, xs = np.nonzero(seg)
    seg_of = sl[ys, xs]

    # which nodes touch which piece
    pa, pb = [], []
    for dy, dx in OFFS8:
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        nn = NL[yy[ok], xx[ok]]
        sel = nn > 0
        pa.append(seg_of[ok][sel])
        pb.append(nn[sel])
    seg_nodes = defaultdict(set)
    if pa:
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        for s, nd in set(zip(pa.tolist(), pb.tolist())):
            seg_nodes[s].add(nd)

    order = np.argsort(seg_of, kind="stable")
    ys_s, xs_s, seg_s = ys[order], xs[order], seg_of[order]
    bounds = np.flatnonzero(np.diff(seg_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(seg_s)]])
    pix_of = {int(seg_s[a]): np.stack([ys_s[a:b], xs_s[a:b]], axis=1) for a, b in zip(starts, ends)}

    for s, nodes in seg_nodes.items():
        nodes = sorted(nodes)
        if len(nodes) == 1:
            u = v = nodes[0]
        elif len(nodes) == 2:
            u, v = nodes
        else:  # rare: piece touches 3+ node blobs -> connect the two farthest apart
            best, u, v = -1, nodes[0], nodes[1]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pi, pj = G.nodes[nodes[i]]["pos"], G.nodes[nodes[j]]["pos"]
                    d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                    if d > best:
                        best, u, v = d, nodes[i], nodes[j]
        path = _order_pixels(pix_of[s], G.nodes[u]["pos"])
        coords = [G.nodes[u]["pos"]] + [(float(y), float(x)) for y, x in path] + [G.nodes[v]["pos"]]
        G.add_edge(u, v, u=u, v=v, coords=coords, length_px=_path_length(coords))

    # nodes that touch directly (no pixels between them)
    direct = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a, b = _shift_pairs(NL, dy, dx)
        direct.update(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))
    for u, v in direct:
        if not G.has_edge(u, v):
            coords = [G.nodes[u]["pos"], G.nodes[v]["pos"]]
            G.add_edge(u, v, u=u, v=v, coords=coords, length_px=_path_length(coords))
    return G


# ------------------------------------------------------------ graph cleaning
def _orient(d, start):
    return d["coords"] if d["u"] == start else d["coords"][::-1]


def prune_spurs(G, min_len_px):
    """Remove short dead-end branches that hang off a junction (skeleton noise)."""
    removed = 0
    while True:
        todo = []
        for u, v, k, d in G.edges(keys=True, data=True):
            if u == v or d["length_px"] >= min_len_px:
                continue
            du, dv = G.degree(u), G.degree(v)
            if (du == 1 and dv >= 3) or (dv == 1 and du >= 3) or (du == 1 and dv == 1):
                todo.append((u, v, k))
        if not todo:
            break
        for u, v, k in todo:
            if G.has_edge(u, v, k):
                G.remove_edge(u, v, k)
                removed += 1
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return removed


def contract_degree2(G):
    """A node with exactly two road pieces is a bend, not a junction: join the pieces."""
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n not in G or G.degree(n) != 2:
                continue
            inc = list(G.edges(n, keys=True, data=True))
            if len(inc) != 2:
                continue
            (_, o1, k1, d1), (_, o2, k2, d2) = inc
            if o1 == n or o2 == n:
                continue
            c1 = _orient(d1, o1)  # o1 -> n
            c2 = _orient(d2, n)   # n -> o2
            coords = list(c1) + list(c2[1:])
            G.remove_node(n)
            G.add_edge(o1, o2, u=o1, v=o2, coords=coords, length_px=d1["length_px"] + d2["length_px"])
            changed = True
    return G



def bridge_gaps(G, gap_px, min_cos=0.8):
    """Join two dead ends that face each other across a short gap (broken road in the mask).

    Only leaf-to-leaf joins are made, and both ends must point toward each other,
    so a dead end is never glued to an unrelated road beside it.
    """
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    info = {}
    for n in leaves:
        (_, _, _, d), = list(G.edges(n, keys=True, data=True))
        if d["u"] == d["v"]:
            continue
        c = _orient(d, n)  # n -> other end of its road piece
        k = min(len(c) - 1, 3)
        # use a point ~10 px along the road to get a stable direction
        acc, j = 0.0, 0
        while j < len(c) - 1 and acc < 10:
            acc += math.hypot(c[j + 1][0] - c[j][0], c[j + 1][1] - c[j][1])
            j += 1
        vy, vx = c[0][0] - c[j][0], c[0][1] - c[j][1]
        nv = math.hypot(vy, vx)
        if nv > 0:
            info[n] = (G.nodes[n]["pos"], (vy / nv, vx / nv))
    ids = list(info)
    if len(ids) < 2:
        return 0
    pts = np.array([info[n][0] for n in ids])
    cands = []
    for i, j in cKDTree(pts).query_pairs(gap_px):
        a, b = ids[i], ids[j]
        (pa, da), (pb, db) = info[a], info[b]
        ab = (pb[0] - pa[0], pb[1] - pa[1])
        dist = math.hypot(*ab)
        if dist == 0:
            continue
        uy, ux = ab[0] / dist, ab[1] / dist
        if da[0] * uy + da[1] * ux >= min_cos and db[0] * -uy + db[1] * -ux >= min_cos:
            cands.append((dist, a, b))
    cands.sort()
    used, n_br = set(), 0
    for dist, a, b in cands:
        if a in used or b in used:
            continue
        used.update([a, b])
        coords = [G.nodes[a]["pos"], G.nodes[b]["pos"]]
        G.add_edge(a, b, u=a, v=b, coords=coords, length_px=_path_length(coords))
        n_br += 1
    return n_br


def merge_junctions(G, merge_px, roundabout_max_perimeter_px, roundabout_max_nodes=8):
    """Merge junction nodes closer than merge_px; collapse small cycles (roundabouts) to one node."""
    parent = {n: n for n in G.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    junc = [n for n in G.nodes if G.degree(n) >= 3]
    if len(junc) > 1 and merge_px > 0:
        pts = np.array([G.nodes[n]["pos"] for n in junc])
        for i, j in cKDTree(pts).query_pairs(merge_px):
            union(junc[i], junc[j])

    # roundabouts = short cycles with several arms
    S = nx.Graph()
    for u, v, d in G.edges(data=True):
        if u != v and (not S.has_edge(u, v) or S[u][v]["length_px"] > d["length_px"]):
            S.add_edge(u, v, length_px=d["length_px"])
    rb_members = set()
    for cyc in nx.cycle_basis(S):
        if len(cyc) < 3 or len(cyc) > roundabout_max_nodes:
            continue
        per = sum(S[cyc[i]][cyc[(i + 1) % len(cyc)]]["length_px"] for i in range(len(cyc)))
        arms = sum(1 for n in cyc if G.degree(n) >= 3)
        if per <= roundabout_max_perimeter_px and arms >= 3:
            for n in cyc[1:]:
                union(cyc[0], n)
            rb_members.update(cyc)

    groups = defaultdict(list)
    for n in G.nodes:
        groups[find(n)].append(n)
    H = nx.MultiGraph()
    for root, members in groups.items():
        pts = np.array([G.nodes[m]["pos"] for m in members])
        H.add_node(root, pos=(float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                   roundabout=bool(len(members) > 1 and any(m in rb_members for m in members)))
    for u, v, k, d in G.edges(keys=True, data=True):
        du, dv = d["u"], d["v"]
        mu, mv = find(du), find(dv)
        if mu == mv and du != dv:
            continue  # connector inside a merged group
        coords = [H.nodes[mu]["pos"]] + list(d["coords"][1:-1]) + [H.nodes[mv]["pos"]]
        H.add_edge(mu, mv, u=mu, v=mv, coords=coords, length_px=_path_length(coords))
    return H


def finalize(G, simplify_px=2.0):
    """Pin edge ends to node positions and simplify with Douglas-Peucker."""
    for u, v, k, d in G.edges(keys=True, data=True):
        c = list(d["coords"])
        c[0] = G.nodes[d["u"]]["pos"]
        c[-1] = G.nodes[d["v"]]["pos"]
        arr = np.array(c, dtype=float)
        if len(arr) > 2 and simplify_px > 0:
            arr = approximate_polygon(arr, tolerance=simplify_px)
        d["coords"] = [(float(y), float(x)) for y, x in arr]
        d["length_px"] = _path_length(d["coords"])
    return G


# ------------------------------------------------------------ top-level API
def extract_network(prob, thr=0.5, close_radius=2, min_area_px=400, fill_hole_px=150,
                    junction_radius=3, spur_px=27, merge_px=18, roundabout_px=220, simplify_px=2.0, gap_px=36):
    """Run the whole pipeline. Returns (mask, skeleton, graph, log dict)."""
    log = {}
    mask = clean_mask(prob, thr, close_radius, min_area_px, fill_hole_px)
    log["mask_road_percent"] = float(mask.mean() * 100)
    skel = skeletonise(mask)
    log["skeleton_pixels"] = int(skel.sum())
    G = skeleton_to_graph(skel, junction_radius)
    log["raw_nodes"], log["raw_edges"] = G.number_of_nodes(), G.number_of_edges()
    log["raw_junctions_deg3plus"] = sum(1 for n in G if G.degree(n) >= 3)
    log["spurs_removed"] = prune_spurs(G, spur_px)
    contract_degree2(G)
    log["gaps_bridged"] = bridge_gaps(G, gap_px) if gap_px > 0 else 0
    contract_degree2(G)
    G = merge_junctions(G, merge_px, roundabout_px)
    log["roundabouts_collapsed"] = sum(1 for n in G if G.nodes[n].get("roundabout"))
    prune_spurs(G, spur_px)
    contract_degree2(G)
    G = finalize(G, simplify_px)
    # fresh node ids 0..N-1
    G = nx.convert_node_labels_to_integers(G, first_label=0, label_attribute="old_id")
    mapping_u = {}
    for n, data in G.nodes(data=True):
        mapping_u[data["old_id"]] = n
    for u, v, k, d in G.edges(keys=True, data=True):
        d["u"], d["v"] = mapping_u[d["u"]], mapping_u[d["v"]]
    log["final_nodes"], log["final_edges"] = G.number_of_nodes(), G.number_of_edges()
    return mask, skel, G, log


def junction_type(degree: int) -> str:
    if degree == 3:
        return "T/Y-junction (degree 3)"
    if degree == 4:
        return "four-way (degree 4)"
    return f"complex (degree {degree})"
