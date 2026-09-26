"""
YOLOv8 Classification Service untuk HerbalScan.

Digunakan oleh ml_service.py sebagai alternatif model inference
ketika model_herbalscan_yolo.pt tersedia.
"""

import os
import json
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_YOLO_PATH = os.path.join(BASE_DIR, "models", "model_herbalscan_yolo.pt")
CLASS_NAMES_JSON = os.path.join(BASE_DIR, "models", "class_names.json")
METADATA_PATH = os.path.join(BASE_DIR, "data", "herbal_metadata.json")

_YOLO_MODEL = None
_YOLO_LOAD_ATTEMPTED = False


def is_yolo_available():
    """Cek apakah model YOLO tersedia."""
    return os.path.exists(MODEL_YOLO_PATH)


def load_yolo_model():
    """
    Memuat model YOLOv8 classification.
    Returns model object atau None jika gagal.
    """
    global _YOLO_MODEL, _YOLO_LOAD_ATTEMPTED
    _YOLO_LOAD_ATTEMPTED = True

    if not os.path.exists(MODEL_YOLO_PATH):
        print("[YOLO Service] Model YOLO tidak ditemukan, skip.")
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_YOLO_PATH)
        print(f"[YOLO Service SUCCESS] Model YOLO dimuat ({MODEL_YOLO_PATH}).")
        _YOLO_MODEL = model
        return model
    except ImportError:
        print("[YOLO Service WARNING] ultralytics tidak terinstall.")
        return None
    except Exception as e:
        print(f"[YOLO Service ERROR] Gagal memuat model YOLO: {e}")
        return None


def get_yolo_model():
    global _YOLO_MODEL, _YOLO_LOAD_ATTEMPTED
    if _YOLO_MODEL is None and not _YOLO_LOAD_ATTEMPTED:
        load_yolo_model()
    return _YOLO_MODEL


def predict_with_yolo(image_path: str) -> dict:
    """
    Prediksi gambar menggunakan model YOLOv8 classification.

    Returns dict dengan format yang sama seperti predict_leaf() di ml_service.py,
    atau None jika gagal.
    """
    model = get_yolo_model()
    if model is None:
        return None

    try:
        # Load metadata & class names
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            herbal_metadata = json.load(f)

        if os.path.exists(CLASS_NAMES_JSON):
            with open(CLASS_NAMES_JSON, "r") as f:
                class_names = json.load(f)
        else:
            class_names = [
                "belimbing_wuluh", "jambu_biji", "jeruk_nipis", "kemangi",
                "lidah_buaya", "nangka", "pandan", "pepaya", "seledri", "sirih",
                "bukan_herbal"
            ]

        # Inferensi
        results = model.predict(
            source=image_path,
            imgsz=224,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None

        result = results[0]
        probs = result.probs

        # Top-1 prediction
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()

        # YOLO class names mapping
        # YOLO mungkin menggunakan indeks sendiri, cek dari model.names
        if hasattr(model, 'names') and model.names:
            yolo_class_name = model.names.get(top1_idx, f"class_{top1_idx}")
        elif top1_idx < len(class_names):
            yolo_class_name = class_names[top1_idx]
        else:
            yolo_class_name = "bukan_herbal"

        # Normalisasi
        clean_class_id = yolo_class_name.strip().lower().replace(" ", "_")

        # Confidence threshold
        confidence_level = "high"
        if top1_conf < 0.25:
            clean_class_id = "bukan_herbal"
            confidence_level = "very_low"
        elif top1_conf < 0.50:
            confidence_level = "low"

        # Top-3 debug
        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()
        top3_info = []
        for i in range(min(3, len(top5_indices))):
            idx = top5_indices[i]
            name = model.names.get(idx, f"class_{idx}") if hasattr(model, 'names') else f"class_{idx}"
            top3_info.append((name, round(top5_confs[i] * 100, 1)))
        print(f"[YOLO Service] Top-3: {top3_info} | Level: {confidence_level}")

        # Ambil metadata
        plant_info = herbal_metadata.get(clean_class_id, herbal_metadata.get("bukan_herbal"))

        return {
            "name": plant_info["name"],
            "latin_name": plant_info["latin_name"],
            "family": plant_info["family"],
            "is_herbal": plant_info["is_herbal"],
            "description": plant_info["description"],
            "benefits": plant_info["benefits"],
            "usage": plant_info["usage"],
            "confidence": round(top1_conf, 4),
            "confidence_percent": int(top1_conf * 100),
            "confidence_level": confidence_level,
            "class_id": clean_class_id,
            "is_real_model": True,
            "model_type": "yolov8"
        }

    except Exception as e:
        print(f"[YOLO Service ERROR] Prediksi gagal: {e}")
        return None
