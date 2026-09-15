import mercantile, requests, os

BBOX = (72.6300, 23.2100, 72.6500, 23.2300)   # west, south, east, north — Gandhinagar sector grid
ZOOM = 18
OUT_DIR = "data/tiles"
os.makedirs(OUT_DIR, exist_ok=True)

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

tiles = list(mercantile.tiles(*BBOX, zooms=[ZOOM]))
print(f"Found {len(tiles)} tiles to download")

for i, t in enumerate(tiles):
    out_path = f"{OUT_DIR}/{t.z}_{t.x}_{t.y}.png"
    if os.path.exists(out_path):
        continue  # already downloaded, skip it

    url = ESRI_URL.format(z=t.z, x=t.x, y=t.y)
    try:
        r = requests.get(url, headers={"User-Agent": "ITS-project"}, timeout=10)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"[{i+1}/{len(tiles)}] Saved {t.z} {t.x} {t.y}")
        else:
            print(f"[{i+1}/{len(tiles)}] Failed (status {r.status_code}): {t}")
    except requests.exceptions.RequestException as e:
        print(f"[{i+1}/{len(tiles)}] Error: {t} -> {e}")

print("Done.")