from ultralytics import YOLO

# ✅ Load pretrained YOLO11n
model = YOLO("yolo11n.pt")

# ✅ Latih model dengan augmentasi "buah tampak kecil / jauh"
model.train(
    data="dataset.yaml",
    epochs=120,
    imgsz=480,
    batch=32,
    device=0,
    workers=8,
    optimizer="AdamW",
    lr0=0.001,
    pretrained=False,
    project="runs",
    name="yolo11s_ripeness_farview",
    
    scale=0.40,           # mengecilkan objek (0.3–0.6 ideal)
    translate=0.20,       # geser posisi objek, menambah variasi posisi
    mosaic=1.0,           # aktifkan mosaic untuk memperkecil buah relatif
    perspective=0.0005,   # efek distorsi kamera ringan
    shear=1.0,            # sedikit kemiringan untuk variasi perspektif
    hsv_h=0.015,          # variasi hue kecil
    hsv_v=0.4,            # variasi brightness
    erasing=0.0,          # nonaktifkan agar buah kecil tidak terhapus
    copy_paste=0.0,       # nonaktif agar tidak tempel objek berlebihan
)
