import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Konfigurasi
OUTPUT_DIR = "bukan_herbal_dataset"
SYNTHETIC_COUNT = 300       # Jumlah gambar sintetis yang di-generate
IMG_SIZE = (224, 224)       # Ukuran output gambar
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def generate_random_noise(size):
    """Gambar noise acak (TV static)."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def generate_solid_color(size):
    """Gambar solid warna acak (non-hijau dominant)."""
    # Hindari warna hijau murni agar tidak overlap dengan daun
    r = random.randint(0, 255)
    g = random.randint(0, 180)  # batasi hijau
    b = random.randint(0, 255)
    arr = np.full((*size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr)


def generate_gradient(size):
    """Gambar gradient linear acak."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Pilih 2 warna acak
    c1 = np.array([random.randint(0, 255) for _ in range(3)])
    c2 = np.array([random.randint(0, 255) for _ in range(3)])
    # Gradient horizontal atau vertikal
    if random.random() < 0.5:
        for x in range(w):
            t = x / max(w - 1, 1)
            arr[:, x] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    else:
        for y in range(h):
            t = y / max(h - 1, 1)
            arr[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    return Image.fromarray(arr)


def generate_geometric_shapes(size):
    """Gambar dengan bentuk geometris acak (lingkaran, kotak, garis)."""
    img = Image.new("RGB", size, color=(
        random.randint(180, 255),
        random.randint(180, 255),
        random.randint(180, 255)
    ))
    draw = ImageDraw.Draw(img)
    num_shapes = random.randint(3, 12)
    for _ in range(num_shapes):
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape_type = random.choice(["rect", "ellipse", "line"])
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
        # Pillow requires ordered coords for rect/ellipse
        lx, rx = min(x1, x2), max(x1, x2)
        ly, ry = min(y1, y2), max(y1, y2)
        if shape_type == "rect":
            draw.rectangle([lx, ly, rx, ry], fill=color)
        elif shape_type == "ellipse":
            draw.ellipse([lx, ly, rx, ry], fill=color)
        elif shape_type == "line":
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 8))
    return img


def generate_texture(size):
    """Gambar tekstur (noise yang di-blur — simulasi permukaan meja/lantai/kain)."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    # Blur kuat untuk efek tekstur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 12)))
    # Tint warna random
    tint = np.array([random.randint(50, 200) for _ in range(3)])
    arr = np.array(img).astype(float)
    arr = (arr * 0.5 + tint * 0.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_dark_image(size):
    """Gambar gelap/hampir hitam (simulasi foto gagal/blur gelap)."""
    brightness = random.randint(5, 40)
    arr = np.random.randint(0, brightness, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def generate_bright_overexposed(size):
    """Gambar terang/overexposed (simulasi foto terlalu terang)."""
    brightness = random.randint(200, 255)
    arr = np.random.randint(brightness, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# Registry semua generator
GENERATORS = [
    ("noise",      generate_random_noise,       0.15),
    ("solid",      generate_solid_color,         0.10),
    ("gradient",   generate_gradient,            0.15),
    ("shapes",     generate_geometric_shapes,    0.20),
    ("texture",    generate_texture,             0.20),
    ("dark",       generate_dark_image,          0.10),
    ("bright",     generate_bright_overexposed,  0.10),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Subfolder untuk foto real dari user
    real_dir = os.path.join(OUTPUT_DIR, "real")
    os.makedirs(real_dir, exist_ok=True)

    print("=" * 60)
    print("HERBALSCAN — Persiapan Dataset 'Bukan Herbal'")
    print("=" * 60)

    # --- Generate gambar sintetis ---
    print(f"\nMembuat {SYNTHETIC_COUNT} gambar sintetis...")

    names = [name for name, _, _ in GENERATORS]
    weights = [w for _, _, w in GENERATORS]
    generators = [gen for _, gen, _ in GENERATORS]

    saved = 0
    for i in range(SYNTHETIC_COUNT):
        # Pilih generator berdasarkan bobot
        gen_idx = random.choices(range(len(GENERATORS)), weights=weights, k=1)[0]
        gen_func = generators[gen_idx]
        gen_name = names[gen_idx]

        img = gen_func(IMG_SIZE)
        filename = f"synth_{gen_name}_{i:04d}.jpg"
        img.save(os.path.join(OUTPUT_DIR, filename), quality=85)
        saved += 1

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{SYNTHETIC_COUNT} gambar di-generate")

    print(f"\n✅ {saved} gambar sintetis berhasil dibuat!")

    # --- Instruksi untuk user ---
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, "w") as f:
        f.write("""
=== INSTRUKSI MENAMBAHKAN FOTO REAL ===

Untuk meningkatkan akurasi penolakan non-herbal, TAMBAHKAN 100-300 foto real
dari HP kamu ke folder 'real/' di sini.

Contoh foto yang bagus untuk ditambahkan:
  - Meja / lantai / tembok
  - Baju / kain / handuk
  - Kucing / anjing / hewan peliharaan
  - Makanan (nasi, roti, kue)
  - Tangan / wajah manusia
  - Keyboard / monitor / gadget
  - Rumput liar / tanaman hias non-herbal
  - Foto blur / gelap / overexposed

Tips:
  - Variasikan pencahayaan (siang, malam, lampu, outdoor)
  - Variasikan sudut pengambilan foto
  - Ukuran file tidak penting, akan di-resize otomatis saat training

Setelah menambahkan foto, GABUNGKAN semua isi folder ini ke:
  dataset_extracted/bukan_herbal/
""")

    print(f"\nLokasi output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  - {saved} gambar sintetis")
    print(f"  - Folder 'real/' untuk foto HP kamu")
    print(f"  - README.txt berisi instruksi")
    print(f"\n⚠️  PENTING: Tambahkan 100-300 foto real dari HP untuk hasil terbaik!")
    print(f"    Lihat {readme_path} untuk panduan lengkap.")


if __name__ == "__main__":
    main()
