import os
import shutil
import random

random.seed(42)  # reproducible split

IMAGES_DIR = "data/dataset/images"
LABELS_DIR = "data/dataset/labels"

OUT_BASE = "data/dataset_split"
for split in ["train", "val"]:
    os.makedirs(f"{OUT_BASE}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT_BASE}/labels/{split}", exist_ok=True)

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"Total: {len(image_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

for split, files in [("train", train_files), ("val", val_files)]:
    for fname in files:
        label_name = fname.replace(".png", ".txt")
        shutil.copy(f"{IMAGES_DIR}/{fname}", f"{OUT_BASE}/images/{split}/{fname}")
        shutil.copy(f"{LABELS_DIR}/{label_name}", f"{OUT_BASE}/labels/{split}/{label_name}")

print("Done. Dataset ready at data/dataset_split/")