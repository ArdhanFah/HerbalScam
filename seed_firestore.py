#!/usr/bin/env python3
"""
Seed Script for HerbalScan Firestore Database
------------------------------------------------
Uploads local `data/herbal_metadata.json` into the Firebase Firestore `plants` collection.

Usage:
1. Place your `serviceAccountKey.json` in the project root directory.
2. Run: `python seed_firestore.py`
"""

import os
import sys
from services.firestore_service import seed_catalog_from_json, init_firebase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data", "herbal_metadata.json")

def main():
    print("==================================================")
    print("   HerbalScan Firestore Catalog Seeder Tool       ")
    print("==================================================")
    
    if not os.path.exists(JSON_PATH):
        print(f"[Error] Metadata file not found at: {JSON_PATH}")
        sys.exit(1)

    db = init_firebase()
    if db is None:
        print("[Error] Firebase Admin SDK is not initialized.")
        print("Pastikan file 'serviceAccountKey.json' sudah diletakkan di root project HerbalScan.")
        sys.exit(1)

    print(f"Reading plant catalog from: {JSON_PATH}")
    success = seed_catalog_from_json(JSON_PATH)
    
    if success:
        print("\n[Success] Data katalog berhasil di-upload ke Firestore koleksi 'plants'!")
    else:
        print("\n[Failed] Gagal melakukan seeding ke Firestore. Periksa log di atas.")

if __name__ == "__main__":
    main()
