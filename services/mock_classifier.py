import random


# Database mock tanaman herbal Indonesia
HERBAL_DATABASE = [
    {
        "name": "Jahe",
        "latin_name": "Zingiber officinale",
        "family": "Zingiberaceae",
        "is_herbal": True,
        "description": "Jahe adalah tanaman rimpang yang sangat populer sebagai rempah-rempah dan bahan obat tradisional Indonesia.",
        "benefits": [
            "Meredakan mual dan masuk angin",
            "Menghangatkan tubuh",
            "Anti-inflamasi alami",
            "Membantu pencernaan",
            "Meningkatkan imunitas tubuh",
        ],
        "usage": "Diseduh sebagai wedang jahe, dijadikan jamu, atau ditambahkan dalam masakan. Bisa juga dioleskan sebagai minyak jahe untuk pijat.",
        "image_hint": "Daun panjang berbentuk lanset, tersusun berseling pada batang.",
    },
    {
        "name": "Kunyit",
        "latin_name": "Curcuma longa",
        "family": "Zingiberaceae",
        "is_herbal": True,
        "description": "Kunyit dikenal dengan warna kuning khasnya, telah digunakan selama ribuan tahun dalam pengobatan tradisional.",
        "benefits": [
            "Anti-oksidan kuat (kurkumin)",
            "Anti-inflamasi",
            "Menjaga kesehatan hati",
            "Memperlancar pencernaan",
            "Meningkatkan daya tahan tubuh",
        ],
        "usage": "Dibuat jamu kunyit asam, ditambahkan dalam masakan sebagai bumbu, atau diminum sebagai suplemen kurkumin.",
        "image_hint": "Daun lebar lonjong, mirip daun pisang kecil, berwarna hijau tua.",
    },
    {
        "name": "Temulawak",
        "latin_name": "Curcuma zanthorrhiza",
        "family": "Zingiberaceae",
        "is_herbal": True,
        "description": "Temulawak adalah tanaman asli Indonesia yang telah lama digunakan sebagai jamu dan obat tradisional.",
        "benefits": [
            "Meningkatkan nafsu makan",
            "Menjaga kesehatan hati dan empedu",
            "Anti-inflamasi",
            "Melancarkan pencernaan",
            "Mengatasi masalah lambung",
        ],
        "usage": "Dibuat jamu temulawak, bisa juga diseduh sebagai teh herbal atau diolah menjadi kapsul suplemen.",
        "image_hint": "Daun besar lonjong dengan tangkai panjang, mirip kunyit tapi lebih besar.",
    },
    {
        "name": "Lidah Buaya",
        "latin_name": "Aloe vera",
        "family": "Asphodelaceae",
        "is_herbal": True,
        "description": "Lidah buaya adalah tanaman sukulen yang kaya akan gel bening dengan berbagai manfaat kesehatan dan kecantikan.",
        "benefits": [
            "Melembapkan dan menyehatkan kulit",
            "Mempercepat penyembuhan luka",
            "Menyehatkan rambut",
            "Membantu pencernaan",
            "Mengandung antioksidan",
        ],
        "usage": "Gel lidah buaya dioleskan langsung ke kulit, dicampurkan dalam minuman, atau dijadikan masker rambut alami.",
        "image_hint": "Daun tebal berdaging, berduri di tepi, berbentuk tombak.",
    },
    {
        "name": "Sirih",
        "latin_name": "Piper betle",
        "family": "Piperaceae",
        "is_herbal": True,
        "description": "Sirih adalah tanaman merambat yang daunnya telah digunakan dalam tradisi dan pengobatan Nusantara sejak lama.",
        "benefits": [
            "Antiseptik alami",
            "Menjaga kesehatan mulut dan gigi",
            "Menghentikan mimisan",
            "Mengurangi bau badan",
            "Anti-bakteri",
        ],
        "usage": "Daun direbus untuk air kumur, ditempelkan pada luka, atau dikunyah untuk kesehatan mulut. Air rebusan sirih juga bisa diminum.",
        "image_hint": "Daun berbentuk jantung/hati, mengkilap, dengan tulang daun yang menonjol.",
    },
    {
        "name": "Kencur",
        "latin_name": "Kaempferia galanga",
        "family": "Zingiberaceae",
        "is_herbal": True,
        "description": "Kencur adalah rempah-rempah yang banyak digunakan dalam jamu dan masakan Indonesia, terutama sebagai beras kencur.",
        "benefits": [
            "Mengatasi batuk dan pilek",
            "Meningkatkan nafsu makan",
            "Meredakan masuk angin",
            "Mengurangi pegal-pegal",
            "Menyegarkan badan",
        ],
        "usage": "Dibuat jamu beras kencur, ditumbuk untuk obat batuk tradisional, atau digunakan sebagai bumbu masakan.",
        "image_hint": "Daun bulat-lonjong yang tumbuh mendatar di atas tanah.",
    },
    {
        "name": "Daun Mint",
        "latin_name": "Mentha piperita",
        "family": "Lamiaceae",
        "is_herbal": True,
        "description": "Mint adalah tanaman herbal dengan aroma segar yang khas, banyak digunakan sebagai penyegar dan obat alami.",
        "benefits": [
            "Menyegarkan nafas",
            "Meredakan sakit kepala",
            "Membantu pencernaan",
            "Mengurangi mual",
            "Efek menenangkan",
        ],
        "usage": "Diseduh sebagai teh mint, ditambahkan ke minuman sebagai penyegar, atau dihirup aromanya untuk relaksasi.",
        "image_hint": "Daun kecil bergerigi, berselang-seling, dengan aroma mentol khas.",
    },
    {
        "name": "Kumis Kucing",
        "latin_name": "Orthosiphon aristatus",
        "family": "Lamiaceae",
        "is_herbal": True,
        "description": "Kumis kucing dinamai dari bunganya yang menyerupai kumis kucing, tanaman ini populer sebagai obat herbal ginjal.",
        "benefits": [
            "Melancarkan buang air kecil",
            "Membantu kesehatan ginjal",
            "Menurunkan kadar asam urat",
            "Anti-inflamasi",
            "Menurunkan tekanan darah",
        ],
        "usage": "Daun direbus dan airnya diminum sebagai teh herbal. Biasanya diminum 2-3 kali sehari.",
        "image_hint": "Daun bergerigi kasar, berhadapan, dengan bunga putih menyerupai kumis.",
    },
]

# Response untuk daun yang bukan herbal
NON_HERBAL_RESPONSES = [
    {
        "name": "Tidak Dikenali",
        "latin_name": "-",
        "family": "-",
        "is_herbal": False,
        "description": "Daun ini tidak teridentifikasi sebagai tanaman herbal dalam database kami. Bisa jadi ini adalah tanaman hias, tanaman liar, atau jenis tanaman lain yang belum terdaftar.",
        "benefits": [],
        "usage": "Tidak ada data penggunaan herbal untuk tanaman ini. Jangan mengonsumsi tanaman yang tidak dikenali tanpa konsultasi ahli.",
        "image_hint": "",
    },
    {
        "name": "Bukan Tanaman Herbal",
        "latin_name": "-",
        "family": "-",
        "is_herbal": False,
        "description": "Sistem kami mendeteksi bahwa ini bukan merupakan tanaman herbal yang terdaftar. Gambar mungkin bukan daun, atau merupakan tanaman non-herbal.",
        "benefits": [],
        "usage": "Pastikan Anda memfoto daun dengan jelas dan pencahayaan yang cukup untuk hasil yang lebih akurat.",
        "image_hint": "",
    },
]


def classify_leaf(image_path: str) -> dict:
    """
    Mock classifier untuk daun herbal.
    Mengembalikan hasil klasifikasi berupa dict.
    
    Pada implementasi nyata, fungsi ini akan diganti dengan
    model ML yang sudah di-train (TensorFlow/PyTorch).
    
    Args:
        image_path: Path ke gambar yang akan diklasifikasi
        
    Returns:
        dict dengan keys: name, latin_name, family, is_herbal,
              description, benefits, usage, confidence
    """
    # 80% chance terdeteksi sebagai herbal, 20% bukan
    is_herbal = random.random() < 0.8

    if is_herbal:
        plant = random.choice(HERBAL_DATABASE)
        confidence = round(random.uniform(0.75, 0.98), 2)
    else:
        plant = random.choice(NON_HERBAL_RESPONSES)
        confidence = round(random.uniform(0.60, 0.85), 2)

    return {
        "name": plant["name"],
        "latin_name": plant["latin_name"],
        "family": plant["family"],
        "is_herbal": plant["is_herbal"],
        "description": plant["description"],
        "benefits": plant["benefits"],
        "usage": plant["usage"],
        "confidence": confidence,
        "confidence_percent": int(confidence * 100),
    }
