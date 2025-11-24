import os

from ultralytics import YOLO

# ✅ Inisialisasi path utama
ROOT_DIR = "/content/ta"
DATA_PATH = os.path.join(ROOT_DIR, "dataset.yaml")
MODEL_CFG = os.path.join(ROOT_DIR, "ultralytics", "cfg", "models", "11", "yolo11mod.yaml")  # custom nano
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# ✅ Validasi path dataset.yaml
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset YAML tidak ditemukan di: {DATA_PATH}")

print("📁 Path aktif:")
print(" - Model :", MODEL_CFG)
print(" - Data  :", DATA_PATH)
print(" - Runs  :", RUNS_DIR)

# ✅ Load model YOLO11mod dengan skala nano
model = YOLO(MODEL_CFG, scale="n")  # ✅ paksa pakai skala nano

# ✅ Latih model di GPU T4
model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=480,
    batch=8,
    workers=2,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    momentum=0.9,
    warmup_epochs=3,
    amp=True,
    deterministic=False,
    pretrained=True,  # gunakan backbone pretrained
    project=RUNS_DIR,
    name="yolo11mod_nano_ripeness_colab",  # ✅ nama run disesuaikan
    verbose=True,
)

print("\n🎉 Training selesai! Model tersimpan di folder:")
print(f"📂 {os.path.join(RUNS_DIR, 'yolo11mod_nano_ripeness_colab')}")
