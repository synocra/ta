from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # ✅ langsung panggil pretrained YOLO11m
model.train(
    data="/content/ta/dataset.yaml",
    epochs=100,
    imgsz=480,
    batch=8,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    pretrained=True,
    project="/content/ta/runs",
    name="yolo11m_ripeness_colab"
)
