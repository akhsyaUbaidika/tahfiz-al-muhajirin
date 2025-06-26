import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import pandas as pd

# Inisialisasi Firebase
try:
    if not firebase_admin._apps:
        cred_data = None
        if os.path.exists("firebase_key.json"):
            with open("firebase_key.json", "r") as f:
                cred_data = json.load(f)
        else:
            cred_data = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
            }
        cred = credentials.Certificate(cred_data)
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"Gagal inisialisasi Firebase: {e}")

db = firestore.client()

st.title("📋 Manajemen Data Santri")

# Tambah Santri
st.subheader("➕ Tambah Santri Baru")
nama = st.text_input("Nama Lengkap Santri")
gender = st.selectbox("Jenis Kelamin", ["L", "P"])

if st.button("Simpan Santri"):
    if nama:
        doc_ref = db.collection("santri_master").document(nama)
        doc_ref.set({"nama": nama, "gender": gender})
        st.success(f"Santri '{nama}' berhasil ditambahkan.")
        st.rerun()
    else:
        st.warning("❗ Nama tidak boleh kosong.")

# Tampilkan Data Santri
st.subheader("📄 Daftar Santri")
docs = db.collection("santri_master").stream()
data = [{"nama": doc.to_dict()["nama"], "gender": doc.to_dict()["gender"]} for doc in docs]

if data:
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.subheader("🗑️ Hapus Santri")
    nama_hapus = st.selectbox("Pilih Nama Santri", df["nama"].tolist())
    if st.button("Hapus"):
        db.collection("santri_master").document(nama_hapus).delete()
        st.success(f"Santri '{nama_hapus}' berhasil dihapus.")
        st.rerun()
else:
    st.info("Belum ada data santri.")
