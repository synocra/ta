import os
from ultralytics import YOLO

# ✅ Inisialisasi path utama
ROOT_DIR = "/content/ta"
DATA_PATH = os.path.join(ROOT_DIR, "dataset.yaml")
MODEL_CFG = os.path.join(ROOT_DIR, "ultralytics", "cfg", "models", "11", "yolo11_ca_p2.yaml")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# ✅ Validasi path dataset.yaml
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset YAML tidak ditemukan di: {DATA_PATH}")

print("📁 Path aktif:")
print(" - Model :", MODEL_CFG)
print(" - Data  :", DATA_PATH)
print(" - Runs  :", RUNS_DIR)

model = YOLO(MODEL_CFG)

# ✅ Latih model di GPU T4 dengan augmentasi “jarak jauh”
model.train(
    data=DATA_PATH,
    epochs=120,
    imgsz=480,
    batch=32,
    workers=8,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    pretrained=False,
    project=RUNS_DIR,
    name="yolo11mod_nano_ripeness_farview",
    verbose=True,

    # 💡 Augmentasi untuk simulasi objek kecil
    scale=0.40,          # mengecilkan objek (0.4–0.6 cocok)
    translate=0.2,       # geser posisi objek agar tidak selalu di tengah
    mosaic=1.0,          # aktifkan mosaic (4 gambar jadi satu)
    perspective=0.0005,  # distorsi ringan agar seperti jarak optik
    shear=1.0,           # sedikit rotasi & kemiringan
    hsv_h=0.015,
    hsv_v=0.4,
    erasing=0.0,         # matikan random erasing agar objek kecil tidak hilang
    copy_paste=0.0,      # hindari salin-tempel berlebihan
)

print("\n🎉 Training selesai! Model tersimpan di folder:")
print(f"📂 {os.path.join(RUNS_DIR, 'yolo11mod_nano_ripeness_farview')}")
