import os
from ultralytics import YOLO

# ✅ Inisialisasi path utama
ROOT_DIR = "/content/ta"
DATA_PATH = os.path.join(ROOT_DIR, "dataset.yaml")
MODEL_CFG = os.path.join(ROOT_DIR, "ultralytics", "cfg", "models", "11", "yolo11.yaml")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# ✅ Validasi path dataset.yaml
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset YAML tidak ditemukan di: {DATA_PATH}")

print("📁 Path aktif:")
print(" - Model :", MODEL_CFG)
print(" - Data  :", DATA_PATH)
print(" - Runs  :", RUNS_DIR)

# ✅ Load model custom
model = YOLO(MODEL_CFG)  # ❌ tanpa pretrained di sini

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
    pretrained=True,
    project=RUNS_DIR,
    name="yolo11biasa_ripeness_colab",
    verbose=True
)

print("\n🎉 Training selesai! Model tersimpan di folder:")
print(f"📂 {os.path.join(RUNS_DIR, 'yolo11mod_ripeness_colab')}")
