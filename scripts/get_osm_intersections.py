import requests
import math
from PIL import Image, ImageDraw
from collections import defaultdict

# Same bounding box as Step 3 (Gandhinagar sector grid)
SOUTH, WEST, NORTH, EAST = 23.2100, 72.6300, 23.2300, 72.6500

query = f"""
[out:json][timeout:60];
(
  way["highway"]({SOUTH},{WEST},{NORTH},{EAST});
);
(._;>;);
out body;
"""

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "ITS-project-mideval/1.0 (student project)"}

print("Querying Overpass API... (can take 10-30 seconds)")
r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90)
r.raise_for_status()   # raises a clear error if the request failed
data = r.json()

nodes = {el["id"]: el for el in data["elements"] if el["type"] == "node"}
ways = [el for el in data["elements"] if el["type"] == "way"]
print(f"Got {len(ways)} road ways, {len(nodes)} nodes")

# Count how many road ways touch each node.
# A point on a single straight road belongs to 1 way.
# A point where 3+ roads meet is an intersection.
node_way_count = defaultdict(int)
for way in ways:
    for node_id in set(way["nodes"]):   # set() avoids double-counting within one way
        node_way_count[node_id] += 1

intersections = [
    (node_id, nodes[node_id]["lat"], nodes[node_id]["lon"])
    for node_id, count in node_way_count.items()
    if count >= 3 and node_id in nodes
]
# Figure out which downloaded tile contains the most intersections
def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

ZOOM = 18
tile_counts = defaultdict(int)
for node_id, lat, lon in intersections:
    tx, ty = latlon_to_tile(lat, lon, ZOOM)
    tile_counts[(tx, ty)] += 1

best_tile = max(tile_counts, key=tile_counts.get)
print(f"Best tile: {ZOOM}_{best_tile[0]}_{best_tile[1]} with {tile_counts[best_tile]} intersections")   

print(f"Found {len(intersections)} intersections (nodes shared by 3+ roads)")

# Save raw list for Step 5 (label generation) to use later
with open("data/intersections.csv", "w") as f:
    f.write("node_id,lat,lon\n")
    for node_id, lat, lon in intersections:
        f.write(f"{node_id},{lat},{lon}\n")
print("Saved data/intersections.csv")


# --- Overlay demo: draw intersections onto ONE downloaded tile ---
def latlon_to_pixel(lat, lon, tile_x, tile_y, zoom, tile_size=256):
    n = 2 ** zoom
    x_tile_f = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_tile_f = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    px = (x_tile_f - tile_x) * tile_size
    py = (y_tile_f - tile_y) * tile_size
    return px, py


Z, X, Y = ZOOM, best_tile[0], best_tile[1]  
tile_path = f"data/tiles/{Z}_{X}_{Y}.png"
img = Image.open(tile_path).convert("RGB")
draw = ImageDraw.Draw(img)

drawn = 0
for node_id, lat, lon in intersections:
    px, py = latlon_to_pixel(lat, lon, X, Y, Z)
    if 0 <= px <= 256 and 0 <= py <= 256:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline="red", width=2)
        drawn += 1

img.save("data/demo_intersections_overlay.png")
print(f"Drew {drawn} intersections on tile {Z}_{X}_{Y} -> data/demo_intersections_overlay.png")