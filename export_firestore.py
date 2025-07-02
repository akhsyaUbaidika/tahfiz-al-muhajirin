import firebase_admin
from firebase_admin import credentials, firestore
import json

# Ganti dengan path ke file service account JSON kamu
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

collection_name = "hafalan_santri_al_muhajirin"
docs = db.collection(collection_name).stream()

data = {}
for doc in docs:
    data[doc.id] = doc.to_dict()

# Simpan ke file JSON
with open("hafalan_export.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Export selesai! Data disimpan ke hafalan_export.json")
