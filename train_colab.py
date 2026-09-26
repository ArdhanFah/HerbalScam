# ==============================================================================
# HERBALSCAN — GOOGLE COLAB TRAINING SCRIPT (PyTorch Transfer Learning v2)
# ==============================================================================
#
# PERUBAHAN UTAMA dari v1:
#   ✅ Hapus CIFAR-10 — pakai dataset bukan_herbal yang proper
#   ✅ Preserve aspect ratio (padding, bukan stretch)
#   ✅ Augmentation jauh lebih agresif
#   ✅ Classifier head multi-layer (576→256→128→N)
#   ✅ Freeze/Unfreeze strategy (2 fase training)
#   ✅ 40 epoch + Mixed Precision Training
#
# CARA PAKAI:
#   1. Buka Google Colab → Runtime → Change runtime type → GPU (T4)
#   2. Upload dataset_herbal.zip ke Colab
#   3. Pastikan folder bukan_herbal/ sudah ada di dalam zip
#      (jalankan prepare_bukan_herbal.py lokal dulu, lalu masukkan ke zip)
#   4. Jalankan SEL 1 s/d SEL 7 satu per satu
#   5. Download model_herbalscan.pt dari Colab → pindah ke models/
#
# Struktur folder yang diharapkan:
#   dataset_extracted/
#     ├── belimbing_wuluh/
#     ├── jambu_biji/
#     ├── jeruk_nipis/
#     ├── kemangi/
#     ├── lidah_buaya/
#     ├── nangka/
#     ├── pandan/
#     ├── pepaya/
#     ├── seledri/
#     ├── sirih/
#     └── bukan_herbal/        <-- dari prepare_bukan_herbal.py + foto real
# ==============================================================================


# =============================================================================
# SEL 1: SETUP — Mount Google Drive & Ekstrak Dataset
# =============================================================================
# Jalankan sel ini dulu! Mendukung upload manual ke Colab ATAU dari Google Drive.

import os
import zipfile
import shutil

# --- 1a. Mount Google Drive (Opsional jika simpan di Drive) ---
# Un-comment 2 baris di bawah jika file dataset ada di Google Drive:
from google.colab import drive
drive.mount('/content/drive')

# Path ke file dataset di Google Drive atau root Colab:
# Ganti path ini sesuai lokasi file zip kamu di Drive, contoh:
# zip_path = '/content/drive/MyDrive/dataset_herbal.zip'
zip_path = 'dataset_herbal.zip' 
data_dir = 'dataset_extracted'

if not os.path.exists(zip_path) and os.path.exists('/content/drive/MyDrive/dataset_herbal.zip'):
    zip_path = '/content/drive/MyDrive/dataset_herbal.zip'

if os.path.exists(zip_path):
    print(f"Mengekstrak {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    print("Ekstraksi selesai!")
elif os.path.exists('/content/drive/MyDrive/dataset_extracted'):
    print("Menggunakan folder dataset langsung dari Drive...")
    data_dir = '/content/drive/MyDrive/dataset_extracted'
else:
    print(f"WARNING: {zip_path} tidak ditemukan!")
    print("Upload file zip dataset ke Colab / Google Drive terlebih dahulu.")

# --- Auto-Fix Struktur Folder (Flatten & Normalisasi nama folder) ---
if os.path.exists(data_dir):
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # Jika hasil ekstrak ada di dalam 1 folder utama (misal: "Indonesian Herb Leaf Dataset...")
    if len(subdirs) == 1 and subdirs[0] != "bukan_herbal":
        parent_sub = os.path.join(data_dir, subdirs[0])
        print(f"Merapikan subfolder sub-level: '{subdirs[0]}' -> '{data_dir}/'")
        for item in os.listdir(parent_sub):
            src = os.path.join(parent_sub, item)
            dst = os.path.join(data_dir, item)
            if os.path.exists(dst):
                # Merge jika folder sudah ada
                if os.path.isdir(src) and os.path.isdir(dst):
                    for f in os.listdir(src):
                        shutil.move(os.path.join(src, f), os.path.join(dst, f))
                else:
                    shutil.move(src, dst)
            else:
                shutil.move(src, dst)
        if os.path.exists(parent_sub) and not os.listdir(parent_sub):
            os.rmdir(parent_sub)

    # Rename folder ke format lowercase & snake_case
    for d in os.listdir(data_dir):
        old_p = os.path.join(data_dir, d)
        if os.path.isdir(old_p):
            new_name = d.strip().lower().replace(" ", "_")
            new_p = os.path.join(data_dir, new_name)
            if old_p != new_p:
                os.rename(old_p, new_p)
                print(f"Rename folder: '{d}' -> '{new_name}'")

# --- 1b. Tampilkan struktur folder ---
print("\nStruktur dataset:")
total_images = 0
if os.path.exists(data_dir):
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            # Hitung semua gambar secara rekursif (termasuk subfolder)
            n = 0
            for root, dirs, files in os.walk(p):
                n += len([f for f in files if f.lower().endswith(('.jpg','.png','.jpeg','.webp'))])
            total_images += n
            print(f"  {d:25s} {n:5d} gambar")

    print(f"\n  {'TOTAL':25s} {total_images:5d} gambar")

# Cek apakah bukan_herbal ada
bukan_herbal_dir = os.path.join(data_dir, "bukan_herbal")
if not os.path.exists(bukan_herbal_dir):
    print("\n⚠️  WARNING: Folder 'bukan_herbal/' tidak ditemukan!")
    print("   Jalankan prepare_bukan_herbal.py dulu, lalu masukkan hasilnya ke dataset zip.")
    print("   Training tetap bisa jalan tapi model tidak akan bisa menolak non-herbal.")


# =============================================================================
# SEL 2: DATA LOADING & AUGMENTATION
# =============================================================================
# Augmentasi agresif + preserve aspect ratio.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np

# --- 2a. Custom Transform: Resize with Padding (preserve aspect ratio) ---
class ResizeWithPadding:
    """
    Resize gambar agar sisi terpanjang = target_size,
    lalu pad sisi terpendek dengan warna fill_color.
    Ini menjaga aspect ratio asli daun agar tidak terdistorsi.
    """
    def __init__(self, target_size=224, fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad ke target_size x target_size
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom],
                      fill=self.fill_color)
        return img

# --- 2b. Transformasi ---
# Augmentation JAUH lebih agresif dari v1
data_transforms = {
    'train': transforms.Compose([
        ResizeWithPadding(target_size=224, fill_color=(255, 255, 255)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ]),
    'val': transforms.Compose([
        ResizeWithPadding(target_size=224, fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# --- 2c. Load dataset ---
data_dir = 'dataset_extracted'
full_dataset = datasets.ImageFolder(root=data_dir)
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"Kelas ({num_classes}): {class_names}")
print(f"Total gambar: {len(full_dataset)}")

# --- 2d. Split 80/20 stratified ---
targets = np.array(full_dataset.targets)
train_idx, val_idx = [], []

for c in range(num_classes):
    indices = np.where(targets == c)[0]
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx.extend(indices[:split])
    val_idx.extend(indices[split:])

print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

# --- 2e. Buat dataset dengan transform ---
class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)

train_dataset = SubsetWithTransform(full_dataset, train_idx, data_transforms['train'])
val_dataset   = SubsetWithTransform(full_dataset, val_idx,   data_transforms['val'])

# --- 2f. Class-balanced sampling ---
targets_train = [full_dataset.targets[i] for i in train_idx]
class_counts = np.bincount(targets_train, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1e-6)  # avoid division by zero
sample_weights = [class_weights[t] for t in targets_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Loader siap! Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"Distribusi kelas train: {dict(zip(class_names, class_counts))}")


# =============================================================================
# SEL 3: MODEL — MobileNetV3-Small + Custom Multi-Layer Head
# =============================================================================
# Head multi-layer yang lebih expressive + freeze/unfreeze strategy.

import torch.nn as nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load MobileNetV3-Small pretrained
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# --- Ganti classifier head (multi-layer, lebih expressive) ---
# MobileNetV3-Small features output: 576 dimensions
in_features = model.classifier[0].in_features  # 576
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(256, 128),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(128, num_classes),
)

model = model.to(device)

# --- FASE 1: Freeze backbone, train hanya classifier head ---
# Ini mencegah catastrophic forgetting pada pretrained features
for param in model.features.parameters():
    param.requires_grad = False

# Loss: Label Smoothing biar ga overconfident
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer: hanya train classifier dulu
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=0.01
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# Tampilkan jumlah parameter
total_params = sum(p.numel() for p in model.parameters())
trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen      = total_params - trainable
print(f"Total params: {total_params:,}")
print(f"Trainable (classifier only): {trainable:,}")
print(f"Frozen (backbone): {frozen:,}")
print(f"\nModel Classifier Architecture:")
print(model.classifier)


# =============================================================================
# SEL 4: TRAINING — 2-Phase Freeze/Unfreeze + Mixed Precision
# =============================================================================
# Fase 1 (epoch 1-10): Train classifier head only (backbone frozen)
# Fase 2 (epoch 11-40): Unfreeze backbone, train everything with lower LR
# + Mixed Precision Training untuk kecepatan 2x di GPU

import time
import copy

# --- Konfigurasi Training ---
PHASE1_EPOCHS = 10       # Head-only training
PHASE2_EPOCHS = 30       # Full fine-tuning
NUM_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS  # Total: 40 epochs
MAX_PATIENCE = 10        # Early stopping patience

best_acc = 0.0
best_loss = float('inf')
patience = 0
best_model_wts = copy.deepcopy(model.state_dict())

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

# Mixed Precision setup
use_amp = device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

print(f"Training {NUM_EPOCHS} epochs (Phase 1: {PHASE1_EPOCHS} head-only, Phase 2: {PHASE2_EPOCHS} fine-tune)")
print(f"Mixed Precision: {'ON' if use_amp else 'OFF'}")
print("=" * 60)
start = time.time()

for epoch in range(NUM_EPOCHS):
    # --- Transisi ke Fase 2: Unfreeze backbone ---
    if epoch == PHASE1_EPOCHS:
        print(f"\n{'='*60}")
        print(f"🔓 FASE 2: Unfreeze backbone! Fine-tuning seluruh model.")
        print(f"{'='*60}")

        # Unfreeze semua parameter
        for param in model.features.parameters():
            param.requires_grad = True

        # Reset optimizer dengan learning rate 10x lebih kecil
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,  # 10x lebih kecil dari Fase 1
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params sekarang: {trainable:,}")

        # Reset patience untuk fase baru
        patience = 0

    current_lr = optimizer.param_groups[0]['lr']
    phase_label = "HEAD" if epoch < PHASE1_EPOCHS else "FULL"
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} [{phase_label}] (lr={current_lr:.2e})")
    print("-" * 40)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision forward pass
            with torch.amp.autocast('cuda', enabled=use_amp):
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc  = running_corrects.double() / total_samples

        tag = "Train" if phase == "train" else "Val  "
        print(f"  {tag}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
            history['lr'].append(current_lr)
        else:
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc.item())

            scheduler.step(epoch_loss)

            if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience = 0
                print(f"    >> Model terbaik disimpan! (Acc: {best_acc:.4f}, Loss: {best_loss:.4f})")
            else:
                patience += 1

    if patience >= MAX_PATIENCE:
        print(f"\nEarly stopping! Tidak ada perbaikan selama {MAX_PATIENCE} epoch.")
        break

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"Training selesai dalam {elapsed//60:.0f}m {elapsed%60:.0f}s")
print(f"Best Val Accuracy: {best_acc:.4f}")
print(f"Best Val Loss: {best_loss:.4f}")


# =============================================================================
# SEL 5: PLOT Training History
# =============================================================================
# Grafik loss, accuracy, dan learning rate.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=2)
axes[0].plot(history['val_loss'],   label='Val Loss',   marker='s', markersize=2)
axes[0].axvline(x=PHASE1_EPOCHS-0.5, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=2)
axes[1].plot(history['val_acc'],   label='Val Acc',   marker='s', markersize=2)
axes[1].axvline(x=PHASE1_EPOCHS-0.5, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(history['lr'], label='Learning Rate', color='green', marker='.', markersize=2)
axes[2].axvline(x=PHASE1_EPOCHS-0.5, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('LR')
axes[2].set_yscale('log')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Grafik disimpan: training_history.png")


# =============================================================================
# SEL 6: CONFUSION MATRIX — Cek per kelas mana yang sering salah
# =============================================================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model.load_state_dict(best_model_wts)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


# =============================================================================
# SEL 7: SIMPAN MODEL — .pt + class_names.json
# =============================================================================

import json

model.load_state_dict(best_model_wts)
model.eval()

# --- 7a. Simpan State Dict ---
torch.save(model.state_dict(), 'model_herbalscan.pt')
print("Disimpan: model_herbalscan.pt")

# --- 7b. Simpan TorchScript JIT ---
try:
    example = torch.rand(1, 3, 224, 224).to(device)
    traced = torch.jit.trace(model, example)
    traced.save('model_herbalscan_jit.pt')
    print("Disimpan: model_herbalscan_jit.pt")
except Exception as e:
    print(f"Gagal buat JIT: {e}")

# --- 7c. Simpan class names (lowercase snake_case, konsisten) ---
# PENTING: urutan harus SAMA PERSIS dengan ImageFolder output
class_names_clean = [name.strip().lower().replace(" ", "_") for name in class_names]
with open('class_names.json', 'w') as f:
    json.dump(class_names_clean, f, indent=2)
print(f"Disimpan: class_names.json -> {class_names_clean}")

# --- 7d. Simpan info training ---
training_info = {
    "num_classes": num_classes,
    "class_names": class_names_clean,
    "best_val_accuracy": float(best_acc),
    "best_val_loss": float(best_loss),
    "epochs_trained": len(history['train_loss']),
    "total_epochs": NUM_EPOCHS,
    "phase1_epochs": PHASE1_EPOCHS,
    "phase2_epochs": PHASE2_EPOCHS,
    "model_architecture": "MobileNetV3-Small",
    "classifier_head": "576->256->128->N",
    "input_size": 224,
    "preprocessing": "ResizeWithPadding(224, black)",
    "dataset_size": len(full_dataset),
}
with open('training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"Disimpan: training_info.json")

# --- 7e. Tampilkan info ---
print(f"\n{'=' * 60}")
print(f"Dataset: {len(full_dataset)} gambar, {num_classes} kelas")
print(f"Best Val Accuracy: {best_acc:.4f}")
print(f"Best Val Loss: {best_loss:.4f}")
print(f"Epochs trained: {len(history['train_loss'])}/{NUM_EPOCHS}")
print(f"  Phase 1 (head-only): {PHASE1_EPOCHS} epochs")
print(f"  Phase 2 (fine-tune): {len(history['train_loss']) - PHASE1_EPOCHS} epochs")
print(f"\nDownload files:")
print(f"  - model_herbalscan.pt       (WAJIB)")
print(f"  - model_herbalscan_jit.pt   (opsional)")
print(f"  - class_names.json          (WAJIB)")
print(f"  - training_info.json        (opsional, untuk referensi)")
print(f"  - training_history.png")
print(f"  - confusion_matrix.png")
print(f"\nSetelah download, taruh file .pt dan .json di folder models/ project kamu.")
