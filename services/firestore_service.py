import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("firestore_service")

# Global Firestore DB client
db = None

def init_firebase():
    """
    Initialize Firebase Admin SDK if credentials exist.
    Looks for serviceAccountKey.json or FIREBASE_CREDENTIALS env var.
    """
    global db
    if db is not None:
        return db

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_path = os.environ.get(
                "FIREBASE_CREDENTIALS",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serviceAccountKey.json")
            )

            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                logger.info(f"[Firestore] Successfully initialized using {cred_path}")
            else:
                logger.warning(f"[Firestore] Key file not found at {cred_path}. Firestore features will run in fallback/offline mode.")
        else:
            db = firestore.client()
    except Exception as e:
        logger.warning(f"[Firestore] Failed to initialize Firebase Admin SDK: {e}")

    return db


def get_plant_from_firestore(class_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve plant catalog data from Firestore 'plants' collection.
    """
    client = init_firebase()
    if client is None:
        return None

    try:
        doc_ref = client.collection("plants").document(class_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.error(f"[Firestore] Error fetching plant {class_id}: {e}")

    return None


def save_scan_history(user_id: str, scan_data: Dict[str, Any]) -> Optional[str]:
    """
    Save a new scan record to Firestore 'scans' collection.
    Returns document ID on success.
    """
    client = init_firebase()
    if client is None:
        return None

    try:
        from firebase_admin import firestore
        
        payload = {
            "user_id": user_id or "anonymous",
            "class_id": scan_data.get("class_id"),
            "plant_name": scan_data.get("name"),
            "confidence": scan_data.get("confidence"),
            "confidence_percent": scan_data.get("confidence_percent"),
            "confidence_level": scan_data.get("confidence_level"),
            "image_url": scan_data.get("image_url", ""),
            "is_herbal": scan_data.get("is_herbal", False),
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        doc_ref = client.collection("scans").add(payload)
        doc_id = doc_ref[1].id
        logger.info(f"[Firestore] Scan history saved with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"[Firestore] Error saving scan history: {e}")
        return None


def seed_catalog_from_json(json_path: str) -> bool:
    """
    Upload local herbal_metadata.json into Firestore 'plants' collection.
    """
    client = init_firebase()
    if client is None:
        logger.error("[Firestore] Cannot seed catalog: Firestore is not initialized.")
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        batch = client.batch()
        count = 0

        for class_id, plant_info in data.items():
            doc_ref = client.collection("plants").document(class_id)
            batch.set(doc_ref, plant_info)
            count += 1

        batch.commit()
        logger.info(f"[Firestore] Successfully seeded {count} plant catalog documents to Firestore.")
        return True
    except Exception as e:
        logger.error(f"[Firestore] Error seeding catalog: {e}")
        return False
