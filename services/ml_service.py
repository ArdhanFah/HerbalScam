import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration & Metadata Loading
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data", "herbal_metadata.json")
MODEL_PT_PATH = os.path.join(BASE_DIR, "models", "model_herbalscan.pt")
MODEL_JIT_PATH = os.path.join(BASE_DIR, "models", "model_herbalscan_jit.pt")
CLASS_NAMES_JSON = os.path.join(BASE_DIR, "models", "class_names.json")

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    HERBAL_METADATA: Dict[str, Any] = json.load(f)

from services.firestore_service import get_plant_from_firestore, save_scan_history

# Load class names (falling back to default classes if not generated yet)
if os.path.exists(CLASS_NAMES_JSON):
    with open(CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
        CLASS_NAMES: List[str] = json.load(f)
else:
    CLASS_NAMES: List[str] = [
        "belimbing_wuluh", "jambu_biji", "jeruk_nipis", "kemangi",
        "lidah_buaya", "nangka", "pandan", "pepaya", "seledri", "sirih",
        "bukan_herbal"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Transforms
# ─────────────────────────────────────────────────────────────────────────────

class ResizeWithPadding:
    """
    Resizes image maintaining aspect ratio and pads to target_size.
    Identical to training pipeline transform.
    """

    def __init__(self, target_size: int = 224, fill_color: Tuple[int, int, int] = (255, 255, 255)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        import torchvision.transforms.functional as TF

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill_color)


# ─────────────────────────────────────────────────────────────────────────────
# Background Removal Utilities
# ─────────────────────────────────────────────────────────────────────────────

_REMBG_SESSION = None
_REMBG_LOADED = False


def _get_rembg_session():
    """Lazily loads and caches rembg session for background removal."""
    global _REMBG_SESSION, _REMBG_LOADED
    if _REMBG_LOADED:
        return _REMBG_SESSION
    _REMBG_LOADED = True
    try:
        from rembg import new_session
        _REMBG_SESSION = new_session(model_name="u2netp")
        print("[ML Service] rembg session initialized (u2netp) ✅")
    except ImportError:
        print("[ML Service] rembg not installed, fallback segmentation will be used.")
    except Exception as e:
        print(f"[ML Service] Error initializing rembg: {e}")
    return _REMBG_SESSION


def remove_background(img: Image.Image) -> Image.Image:
    """
    Removes image background using rembg (u2netp) and composites onto a white canvas.
    """
    session = _get_rembg_session()
    if session is not None:
        try:
            from rembg import remove
            nobg = remove(img, session=session)
            if nobg.mode == "RGBA":
                bg = Image.new("RGB", nobg.size, (255, 255, 255))
                bg.paste(nobg, mask=nobg.split()[3])
                print("[ML Service] Background removed via rembg (u2netp) ✅")
                return bg
        except Exception as e:
            print(f"[ML Service] rembg removal failed ({e}), using raw image.")

    print("[ML Service] rembg not available, using raw original image.")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & Initialization
# ─────────────────────────────────────────────────────────────────────────────

_ML_MODEL_BUNDLE: Optional[Tuple[Any, Any, Any]] = None
_MODEL_LOAD_ATTEMPTED: bool = False


def _build_mobilenet_model(state_dict: Dict[str, Any], num_classes: int) -> Any:
    """Builds MobileNetV3 (Small or Large) matching saved state_dict architecture."""
    import torch.nn as nn
    from torchvision import models

    in_features = 576
    if "classifier.0.weight" in state_dict:
        in_features = state_dict["classifier.0.weight"].shape[1]

    try:
        if in_features == 960:
            model = models.mobilenet_v3_large(weights=None)
            print("[ML Service] Auto-detected MobileNetV3-Large architecture (960 features) ✅")
        else:
            model = models.mobilenet_v3_small(weights=None)
            print(f"[ML Service] Auto-detected MobileNetV3-Small architecture ({in_features} features) ✅")
    except TypeError:
        if in_features == 960:
            model = models.mobilenet_v3_large(pretrained=False)
        else:
            model = models.mobilenet_v3_small(pretrained=False)

    classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
    max_layer_idx = max((int(k.split(".")[1]) for k in classifier_keys if k.split(".")[1].isdigit()), default=0)

    if max_layer_idx >= 6:
        hidden_1 = state_dict["classifier.0.weight"].shape[0]
        hidden_2 = state_dict["classifier.3.weight"].shape[0]
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_1, hidden_2),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_2, num_classes),
        )
    else:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(state_dict)
    return model


def load_ml_model() -> Optional[Tuple[Any, Any, Any]]:
    """Loads PyTorch MobileNetV3 or TorchScript JIT model into memory."""
    global _ML_MODEL_BUNDLE, _MODEL_LOAD_ATTEMPTED
    _MODEL_LOAD_ATTEMPTED = True

    try:
        import torch
        import torchvision.transforms as transforms
    except ImportError as e:
        print(f"[ML Service WARNING] PyTorch not installed ({e}). Using Mock Classifier.")
        return None

    device = torch.device("cpu")
    transform = transforms.Compose([
        ResizeWithPadding(target_size=224, fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Try PyTorch state_dict (.pt)
    if os.path.exists(MODEL_PT_PATH):
        try:
            loaded = torch.load(MODEL_PT_PATH, map_location=device, weights_only=False)
            if isinstance(loaded, dict):
                num_classes = len(CLASS_NAMES)
                for key in ["classifier.6.weight", "classifier.3.weight", "fc.weight"]:
                    if key in loaded:
                        num_classes = loaded[key].shape[0]
                        break
                model = _build_mobilenet_model(loaded, num_classes)
            else:
                model = loaded

            model.eval()
            print(f"[ML Service SUCCESS] Loaded PyTorch model ({MODEL_PT_PATH}).")
            _ML_MODEL_BUNDLE = (model, transform, device)
            return _ML_MODEL_BUNDLE
        except Exception as e:
            print(f"[ML Service ERROR] Failed loading PyTorch model: {e}")

    # 2. Try TorchScript JIT
    if os.path.exists(MODEL_JIT_PATH):
        try:
            model = torch.jit.load(MODEL_JIT_PATH, map_location=device)
            model.eval()
            print(f"[ML Service SUCCESS] Loaded TorchScript JIT model ({MODEL_JIT_PATH}).")
            _ML_MODEL_BUNDLE = (model, transform, device)
            return _ML_MODEL_BUNDLE
        except Exception as e:
            print(f"[ML Service ERROR] Failed loading JIT model: {e}")

    print("[ML Service INFO] No valid PyTorch model found in models/.")
    return None


def get_ml_model_bundle() -> Optional[Tuple[Any, Any, Any]]:
    """Returns cached model bundle or loads it if not attempted yet."""
    global _ML_MODEL_BUNDLE, _MODEL_LOAD_ATTEMPTED
    if _ML_MODEL_BUNDLE is None and not _MODEL_LOAD_ATTEMPTED:
        load_ml_model()
    return _ML_MODEL_BUNDLE


# Initial load at import time
load_ml_model()


# ─────────────────────────────────────────────────────────────────────────────
# Inference & Prediction Logic
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_tta(model: Any, img: Image.Image, base_transform: Any, device: Any) -> Any:
    """Runs Test-Time Augmentation (4 lightweight transforms) and averages probabilities."""
    import torch

    augmented_tensors = [
        base_transform(img),
        base_transform(img.transpose(Image.FLIP_LEFT_RIGHT)),
        base_transform(img.rotate(10, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))),
        base_transform(img.rotate(-10, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))),
    ]

    batch = torch.stack(augmented_tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)
        all_probs = torch.nn.functional.softmax(outputs, dim=1)
        avg_probs = all_probs.mean(dim=0)

    return avg_probs


def predict_leaf(image_path: str) -> Dict[str, Any]:
    """
    Main entrypoint for leaf classification.
    Runs background removal -> TTA inference -> confidence scoring -> metadata lookup.
    """
    bundle = get_ml_model_bundle()
    if bundle is None:
        from services.mock_classifier import classify_leaf
        result = classify_leaf(image_path)
        result["note"] = "Model PyTorch belum dimuat (menggunakan Mock Classifier)"
        return result

    model, transform, device = bundle

    try:
        import torch

        img = Image.open(image_path).convert('RGB')

        # Step 1: Remove background
        img = remove_background(img)

        # Step 2: TTA Inference
        probabilities = predict_with_tta(model, img, transform, device)

        top_probs, top_indices = torch.topk(probabilities, k=min(3, probabilities.size(0)))
        confidence = top_probs[0].item()
        predicted_idx = top_indices[0].item()

        margin = (top_probs[0] - top_probs[1]).item() if top_probs.size(0) > 1 else 1.0

        if predicted_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_idx]
        else:
            predicted_class = "bukan_herbal"

        clean_class_id = predicted_class.strip().lower().replace(" ", "_")

        # Confidence thresholds
        confidence_level = "high"
        if confidence < 0.40:
            confidence_level = "medium"
        if confidence < 0.20:
            confidence_level = "low"

        # Log Top-3 results
        top3_info = [
            (
                CLASS_NAMES[top_indices[i].item()]
                if top_indices[i].item() < len(CLASS_NAMES)
                else "?",
                round(top_probs[i].item() * 100, 1),
            )
            for i in range(min(3, len(top_probs)))
        ]
        print(f"[ML Service] Top-3: {top3_info} | Margin: {margin:.3f} | Level: {confidence_level}")

        # Try Firestore catalog first, fallback to local JSON
        plant_info = get_plant_from_firestore(clean_class_id)
        if plant_info is None:
            plant_info = HERBAL_METADATA.get(clean_class_id, HERBAL_METADATA.get("bukan_herbal"))

        return {
            "name": plant_info["name"],
            "latin_name": plant_info["latin_name"],
            "family": plant_info["family"],
            "is_herbal": plant_info["is_herbal"],
            "description": plant_info["description"],
            "benefits": plant_info["benefits"],
            "usage": plant_info["usage"],
            "confidence": round(confidence, 4),
            "confidence_percent": int(confidence * 100),
            "confidence_level": confidence_level,
            "class_id": clean_class_id,
            "is_real_model": True,
        }
    except Exception as e:
        print(f"[ML Service] Error predicting leaf: {e}")
        from services.mock_classifier import classify_leaf
        result = classify_leaf(image_path)
        result["note"] = f"Terjadi kesalahan inferensi ({str(e)}), fallback ke Mock Classifier."
        return result
