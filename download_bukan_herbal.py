"""
Download dataset foto real untuk kelas 'bukan_herbal'.

Sumber: Kaggle "Natural Images" dataset
Berisi 8 kelas: airplane, car, cat, dog, flower, fruit, motorbike, person
Semua resolusi tinggi (bukan 32x32 seperti CIFAR-10).

CARA PAKAI:
  Opsi A (Kaggle API — recommended):
    1. pip install kaggle
    2. Taruh kaggle.json di ~/.kaggle/
    3. python download_bukan_herbal.py --kaggle

  Opsi B (Manual):
    1. Download dari: https://www.kaggle.com/datasets/prasunroy/natural-images
    2. Ekstrak, lalu jalankan: python download_bukan_herbal.py --local <path_to_extracted>

  Opsi C (Tanpa Kaggle — pakai gambar sintetis + foto HP sendiri):
    1. python prepare_bukan_herbal.py
    2. Tambahkan foto real dari HP ke folder bukan_herbal_dataset/real/
"""
import os
import sys
import shutil
import argparse
import random
from pathlib import Path

OUTPUT_DIR = "bukan_herbal_dataset"
# Kelas dari Natural Images yang akan dipakai sebagai "bukan herbal"
# Skip 'flower' karena bisa mirip tanaman herbal
SKIP_CLASSES = {"flower"}
MAX_PER_SOURCE_CLASS = 200  # Max gambar per kelas sumber


def download_kagglehub():
    """Download Natural Images dataset via kagglehub (no credentials needed)."""
    try:
        import kagglehub
    except ImportError:
        print("❌ Library kagglehub belum terinstall!")
        print("   Jalankan: pip install kagglehub")
        sys.exit(1)

    print("Downloading Natural Images via kagglehub...")
    try:
        path = kagglehub.dataset_download("prasunroy/natural-images")
        print(f"✅ Download selesai!")
        print(f"   Path: {path}")
        return path
    except Exception as e:
        print(f"❌ Gagal download: {e}")
        print("   Download manual dari:")
        print("   https://www.kaggle.com/datasets/prasunroy/natural-images")
        sys.exit(1)


def find_image_root(base_dir):
    """
    Cari folder yang berisi subfolder kelas gambar.
    Natural Images biasanya punya struktur:
      natural_images/natural_images/airplane/, car/, cat/, ...
    atau
      natural_images/airplane/, car/, cat/, ...
    """
    # Cek apakah langsung ada subfolder kelas
    subdirs = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('_')]

    # Cek apakah ada gambar di subfolder
    for sd in subdirs:
        sd_path = os.path.join(base_dir, sd)
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                        for f in os.listdir(sd_path) if os.path.isfile(os.path.join(sd_path, f)))
        if has_images:
            return base_dir

    # Coba masuk 1 level lebih dalam
    for sd in subdirs:
        nested = os.path.join(base_dir, sd)
        nested_subdirs = [d for d in os.listdir(nested)
                         if os.path.isdir(os.path.join(nested, d))]
        for nsd in nested_subdirs:
            nsd_path = os.path.join(nested, nsd)
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                            for f in os.listdir(nsd_path) if os.path.isfile(os.path.join(nsd_path, f)))
            if has_images:
                return nested

    return base_dir


def process_images(source_dir, output_dir, max_per_class=MAX_PER_SOURCE_CLASS):
    """
    Proses gambar dari dataset source ke folder output bukan_herbal.
    Gabungkan semua kelas non-plant menjadi satu folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_root = find_image_root(source_dir)
    print(f"\nImage root: {image_root}")

    total_copied = 0
    class_dirs = sorted([d for d in os.listdir(image_root)
                        if os.path.isdir(os.path.join(image_root, d)) and not d.startswith('_')])

    print(f"Kelas ditemukan: {class_dirs}")

    for class_name in class_dirs:
        # Skip kelas yang mirip tanaman
        if class_name.lower() in SKIP_CLASSES:
            print(f"  ⏭️  Skip '{class_name}' (mirip tanaman)")
            continue

        class_path = os.path.join(image_root, class_name)

        # Kumpulkan semua gambar
        images = []
        for root, dirs, files in os.walk(class_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    images.append(os.path.join(root, f))

        if not images:
            continue

        # Random sample jika terlalu banyak
        random.seed(42)
        if len(images) > max_per_class:
            images = random.sample(images, max_per_class)

        # Copy ke output
        for img_path in images:
            ext = os.path.splitext(img_path)[1].lower()
            dest_name = f"{class_name}_{total_copied:05d}{ext}"
            dest_path = os.path.join(output_dir, dest_name)
            shutil.copy2(img_path, dest_path)
            total_copied += 1

        print(f"  ✅ {class_name:15s} → {len(images)} gambar")

    return total_copied


def merge_with_synthetic(output_dir):
    """Gabungkan dengan gambar sintetis dari prepare_bukan_herbal.py jika ada."""
    synthetic_dir = "bukan_herbal_dataset"
    if not os.path.exists(synthetic_dir) or synthetic_dir == output_dir:
        return 0

    count = 0
    for f in os.listdir(synthetic_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and f.startswith('synth_'):
            src = os.path.join(synthetic_dir, f)
            dst = os.path.join(output_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                count += 1

    # Juga cek subfolder real/
    real_dir = os.path.join(synthetic_dir, "real")
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                src = os.path.join(real_dir, f)
                dst = os.path.join(output_dir, f"real_{count:05d}{os.path.splitext(f)[1]}")
                shutil.copy2(src, dst)
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Download foto real untuk kelas bukan_herbal")
    parser.add_argument("--local", type=str, default=None,
                       help="Path ke folder dataset yang sudah di-download manual")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                       help=f"Folder output (default: {OUTPUT_DIR})")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_SOURCE_CLASS,
                       help=f"Max gambar per kelas sumber (default: {MAX_PER_SOURCE_CLASS})")

    args = parser.parse_args()

    print("=" * 60)
    print("HERBALSCAN — Download Foto Real 'Bukan Herbal'")
    print("=" * 60)

    if args.local:
        # Pakai dataset yang sudah ada di lokal
        if not os.path.exists(args.local):
            print(f"❌ Folder tidak ditemukan: {args.local}")
            sys.exit(1)
        source_dir = args.local
        print(f"Menggunakan dataset lokal: {source_dir}")
    else:
        # Default: download otomatis via kagglehub
        source_dir = download_kagglehub()

    # Proses gambar → masuk ke subfolder real/
    real_dir = os.path.join(args.output, "real")
    total_real = process_images(source_dir, real_dir, args.max_per_class)

    # Hitung sintetis (dari prepare_bukan_herbal.py)
    total_synth = len([f for f in os.listdir(args.output)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

    # Hitung total
    total_all = total_real + total_synth

    print(f"\n{'=' * 60}")
    print(f"✅ Dataset bukan_herbal siap!")
    print(f"   Foto real    : {total_real} gambar  ({os.path.abspath(real_dir)}/)")
    print(f"   Sintetis     : {total_synth} gambar  ({os.path.abspath(args.output)}/)")
    print(f"   TOTAL        : {total_all} gambar")
    print(f"\nLangkah selanjutnya:")
    print(f"  Pindahkan/copy folder ini ke dataset_herbal.zip sebagai 'bukan_herbal/'")
    print(f"  Atau copy langsung ke dataset_extracted/bukan_herbal/ di Colab")


if __name__ == "__main__":
    main()
