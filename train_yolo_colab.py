# ==============================================================================
# HERBALSCAN — YOLOV8 CLASSIFICATION TRAINING (Google Colab)
# ==============================================================================
#
# Script ini menggunakan YOLOv8-cls (classification mode) dari Ultralytics
# untuk training model klasifikasi daun herbal.
#
# KEUNTUNGAN vs MobileNetV3 manual:
#   ✅ Preprocessing otomatis (augmentation built-in)
#   ✅ Training pipeline yang sudah teruji
#   ✅ Export ke banyak format (ONNX, TorchScript, dll)
#   ✅ Inference cepat
#
# CARA PAKAI:
#   1. Buka Google Colab → Runtime → GPU (T4)
#   2. Upload dataset_herbal.zip
#   3. Jalankan semua sel berurutan
#   4. Download model best.pt → rename ke model_herbalscan_yolo.pt
#   5. Taruh di folder models/ project kamu
#
# Struktur folder yang diharapkan (sama dengan MobileNetV3):
#   dataset_extracted/
#     ├── belimbing_wuluh/
#     ├── jambu_biji/
#     ├── ...
#     ├── sirih/
#     └── bukan_herbal/
# ==============================================================================


# =============================================================================
# SEL 1: INSTALL & SETUP
# =============================================================================

import subprocess
import sys

# Install ultralytics
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.0"])

import os
import zipfile
import shutil

# --- Ekstrak dataset ---
zip_path = 'dataset_herbal.zip'
data_dir = 'dataset_extracted'

if os.path.exists(zip_path):
    print(f"Mengekstrak {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    print("Ekstraksi selesai!")
else:
    print(f"WARNING: {zip_path} tidak ditemukan!")

# --- Auto-Fix folder structure ---
if os.path.exists(data_dir):
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) == 1 and subdirs[0] != "bukan_herbal":
        parent_sub = os.path.join(data_dir, subdirs[0])
        print(f"Merapikan: '{subdirs[0]}' -> '{data_dir}/'")
        for item in os.listdir(parent_sub):
            src = os.path.join(parent_sub, item)
            dst = os.path.join(data_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        if not os.listdir(parent_sub):
            os.rmdir(parent_sub)

    # Normalize folder names
    for d in os.listdir(data_dir):
        old_p = os.path.join(data_dir, d)
        if os.path.isdir(old_p):
            new_name = d.strip().lower().replace(" ", "_")
            new_p = os.path.join(data_dir, new_name)
            if old_p != new_p:
                os.rename(old_p, new_p)

# --- Tampilkan dataset ---
print("\nStruktur dataset:")
for d in sorted(os.listdir(data_dir)):
    p = os.path.join(data_dir, d)
    if os.path.isdir(p):
        n = 0
        for root, dirs, files in os.walk(p):
            n += len([f for f in files if f.lower().endswith(('.jpg','.png','.jpeg','.webp'))])
        print(f"  {d:25s} {n:5d} gambar")


# =============================================================================
# SEL 2: BUAT STRUKTUR DATASET YOLO-CLS
# =============================================================================
# YOLOv8-cls butuh struktur: dataset/train/class_name/ dan dataset/val/class_name/

import numpy as np
from pathlib import Path

data_dir = 'dataset_extracted'
yolo_dir = 'dataset_yolo_cls'

# Buat folder train/ dan val/
for split in ['train', 'val']:
    os.makedirs(os.path.join(yolo_dir, split), exist_ok=True)

# Split 80/20 per kelas
np.random.seed(42)
total_train, total_val = 0, 0

for class_name in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Kumpulkan semua gambar (termasuk subfolder)
    images = []
    for root, dirs, files in os.walk(class_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append(os.path.join(root, f))

    np.random.shuffle(images)
    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Copy ke struktur YOLO
    for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs)]:
        dest_dir = os.path.join(yolo_dir, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for img_path in split_imgs:
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dest_path)

    total_train += len(train_imgs)
    total_val += len(val_imgs)
    print(f"  {class_name:25s} train: {len(train_imgs):4d} | val: {len(val_imgs):4d}")

print(f"\nTotal: train={total_train}, val={total_val}")
print(f"Dataset YOLO siap di: {yolo_dir}/")


# =============================================================================
# SEL 3: TRAINING YOLOV8-CLS
# =============================================================================

from ultralytics import YOLO

# Load pretrained YOLOv8-small classification model
# Opsi: yolov8n-cls (nano), yolov8s-cls (small), yolov8m-cls (medium)
model = YOLO('yolov8s-cls.pt')

# Train!
results = model.train(
    data=yolo_dir,
    epochs=50,
    imgsz=224,
    batch=32,
    patience=15,          # Early stopping
    optimizer='AdamW',
    lr0=1e-3,
    lrf=0.01,             # Final LR = lr0 * lrf
    weight_decay=0.01,
    cos_lr=True,          # Cosine annealing
    hsv_h=0.015,          # Augmentation: hue
    hsv_s=0.5,            # Augmentation: saturation
    hsv_v=0.4,            # Augmentation: value
    degrees=30,           # Rotation
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    flipud=0.1,
    erasing=0.2,          # Random erasing
    crop_fraction=0.8,    # Random crop
    pretrained=True,
    device=0 if __import__('torch').cuda.is_available() else 'cpu',
    project='herbalscan_yolo',
    name='train_v1',
    verbose=True,
)

print("\nTraining selesai!")
print(f"Best model: {results.save_dir}/weights/best.pt")


# =============================================================================
# SEL 4: EVALUASI
# =============================================================================

# Validasi pada val set
metrics = model.val()
print(f"\nVal Accuracy Top-1: {metrics.top1:.4f}")
print(f"Val Accuracy Top-5: {metrics.top5:.4f}")


# =============================================================================
# SEL 5: EXPORT & SIMPAN
# =============================================================================

import json

# Copy best model
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
shutil.copy2(best_model_path, 'model_herbalscan_yolo.pt')
print(f"\nModel disimpan: model_herbalscan_yolo.pt")

# Export ke TorchScript (opsional)
try:
    model.export(format='torchscript', imgsz=224)
    print("Export TorchScript berhasil!")
except Exception as e:
    print(f"Export TorchScript gagal: {e}")

# Simpan class names
class_names = sorted(os.listdir(os.path.join(yolo_dir, 'train')))
class_names_clean = [name.strip().lower().replace(" ", "_") for name in class_names]
with open('class_names.json', 'w') as f:
    json.dump(class_names_clean, f, indent=2)
print(f"Class names: {class_names_clean}")

# Info
print(f"\n{'='*60}")
print(f"Download files:")
print(f"  - model_herbalscan_yolo.pt  (WAJIB)")
print(f"  - class_names.json          (WAJIB)")
print(f"\nTaruh di folder models/ project kamu.")
print(f"ml_service.py akan otomatis deteksi dan pakai model YOLO.")
