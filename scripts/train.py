from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data/data.yaml",
    epochs=150,        # more epochs — small dataset means each epoch is only seconds
    imgsz=256,
    batch=16,
    project="runs",
    name="intersection_detector_v2",
    patience=50,
    mosaic=0.0,        # mosaic augmentation hurts on very small datasets
    degrees=90,        # rotation augmentation — intersections look the same rotated
    fliplr=0.5,
    flipud=0.5         # vertical flip is valid for aerial imagery (unlike normal photos)
)