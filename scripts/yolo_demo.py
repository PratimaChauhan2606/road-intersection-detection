from ultralytics import YOLO

# YOLO26n — smallest/fastest variant, good fit for small-object detection
# in aerial imagery (which is exactly our use case: road intersections)
model = YOLO("yolo26n.pt")   # auto-downloads weights on first run

results = model("data/tiles/18_183968_113687.png")

results[0].save(filename="data/yolo_demo_output.png")
print("Inference ran successfully. Output saved to data/yolo_demo_output.png")