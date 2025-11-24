from ultralytics import YOLO

# Load model hasil training kamu
model = YOLO("D:/TATATA/halo/ultralytics/best.pt")

# Jalankan kamera default (index 0)
model.predict(
    source=0,  # 0 = kamera default, 1 atau 2 jika kamera eksternal
    show=True,  # tampilkan window real-time
    conf=0.5,  # confidence threshold
    device=0,  # GPU (jika CUDA aktif)
    stream=False,  # False = akan buka jendela real-time
)
