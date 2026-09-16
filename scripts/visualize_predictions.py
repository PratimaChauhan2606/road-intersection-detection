from ultralytics import YOLO
import os

model = YOLO("runs/detect/runs/intersection_detector_v2-2/weights/best.pt")

VAL_DIR = "data/dataset_split/images/val"
OUT_DIR = "data/prediction_samples"
os.makedirs(OUT_DIR, exist_ok=True)

val_images = [f for f in os.listdir(VAL_DIR) if f.endswith(".png")][:8]  # sample 8

for img_name in val_images:
    img_path = f"{VAL_DIR}/{img_name}"
    results = model(img_path, conf=0.15)  # lower confidence threshold to actually see boxes with this small dataset
    results[0].save(filename=f"{OUT_DIR}/pred_{img_name}")

print(f"Saved {len(val_images)} prediction images to {OUT_DIR}")