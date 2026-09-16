import csv
import math
import os
import shutil

ZOOM = 18
TILE_SIZE = 256
TILES_DIR = "data/tiles"
BOX_SIZE_PX = 32   # size of the bounding box drawn around each intersection point

# YOLO dataset output structure
IMAGES_DIR = "data/dataset/images"
LABELS_DIR = "data/dataset/labels"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


def latlon_to_tile_and_pixel(lat, lon, zoom, tile_size=256):
    n = 2 ** zoom
    x_tile_f = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_tile_f = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    tile_x, tile_y = int(x_tile_f), int(y_tile_f)
    px = (x_tile_f - tile_x) * tile_size
    py = (y_tile_f - tile_y) * tile_size
    return tile_x, tile_y, px, py


# Load intersections
intersections = []
with open("data/intersections.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        intersections.append((row["node_id"], float(row["lat"]), float(row["lon"])))

# Group intersections by which tile they fall into
from collections import defaultdict
tile_to_points = defaultdict(list)
for node_id, lat, lon in intersections:
    tx, ty, px, py = latlon_to_tile_and_pixel(lat, lon, ZOOM, TILE_SIZE)
    tile_to_points[(tx, ty)].append((px, py))

print(f"{len(tile_to_points)} tiles contain at least one intersection")

count_labeled = 0
for (tx, ty), points in tile_to_points.items():
    src_tile = f"{TILES_DIR}/{ZOOM}_{tx}_{ty}.png"
    if not os.path.exists(src_tile):
        continue  # this tile wasn't downloaded (outside our 272-tile grid)

    # Copy the image into the dataset folder
    dst_image = f"{IMAGES_DIR}/{ZOOM}_{tx}_{ty}.png"
    shutil.copy(src_tile, dst_image)

    # Write YOLO label file: one line per box
    # Format: class x_center y_center width height   (all normalized 0-1)
    label_lines = []
    for px, py in points:
        x_center = px / TILE_SIZE
        y_center = py / TILE_SIZE
        w = BOX_SIZE_PX / TILE_SIZE
        h = BOX_SIZE_PX / TILE_SIZE
        # clip to stay inside 0-1 range in case a point is right at the tile edge
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    label_path = f"{LABELS_DIR}/{ZOOM}_{tx}_{ty}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

    count_labeled += 1

print(f"Created {count_labeled} labeled image/label pairs in data/dataset/")