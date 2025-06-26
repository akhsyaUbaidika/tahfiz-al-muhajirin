import streamlit as st
import pandas as pd
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# Inisialisasi Firebase Admin SDK
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
    st.error(f"⚠️ Gagal inisialisasi Firebase: {e}")

# Firestore client
db = firestore.client()

# Fungsi bantu
def ambil_data_dari_firestore(bulan, tahun):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin")
            .where("bulan", "==", bulan).where("tahun", "==", tahun).stream()]

def ambil_data_dari_nama(nama):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin")
            .where("nama", "==", nama).stream()]

def ambil_daftar_santri():
    return sorted([doc.to_dict()["nama"] for doc in db.collection("santri_master").stream()])

# Tampilan utama
st.title("📗 Evaluasi Hafalan Santri - Tampilan Wali")

menu = st.radio("Pilih Tampilan", ["Periode Bulanan", "Riwayat Santri"])

bulan_list = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]
tahun_list = [str(y) for y in range(2023, 2027)]
today = datetime.today()

# === Tampilan Periode Bulanan ===
if menu == "Periode Bulanan":
    bulan = st.selectbox("Bulan", bulan_list, index=today.month - 1)
    tahun = st.selectbox("Tahun", tahun_list, index=tahun_list.index(str(today.year)))

    if st.button("Tampilkan Evaluasi"):
        df = pd.DataFrame(ambil_data_dari_firestore(bulan, tahun))
        if len(df) >= 2:
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else '-')
            df['jumlah_hafalan_berbobot'] = df['jumlah_hafalan']

            features = df[['jumlah_hafalan_berbobot', 'kelancaran_total', 'kehadiran']]
            features_scaled = StandardScaler().fit_transform(features)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
            df['Klaster'] = kmeans.fit_predict(features_scaled)

            order = df.groupby('Klaster')['jumlah_hafalan_berbobot'].mean().sort_values(ascending=False).index
            mapping = {
                order[0]: 'Cepat & Konsisten',
                order[1]: 'Cukup Baik',
                order[2]: 'Perlu Pendampingan'
            }
            df['Kategori'] = df['Klaster'].map(mapping)

            st.success("✅ Data berhasil ditampilkan.")
            st.dataframe(df[[
                'nama', 'juz', 'juz_sedang',
                'ayat_disetor', 'ayat_sedang_disetor',
                'kehadiran', 'Kategori'
            ]])

            st.plotly_chart(px.pie(df, names='Kategori', title='Distribusi Santri Berdasarkan Kategori'))

            fig, ax = plt.subplots()
            sns.scatterplot(
                data=df,
                x='jumlah_hafalan_berbobot',
                y='kelancaran_total',
                hue='Kategori',
                palette='Set2',
                s=100
            )
            ax.set_xlabel("Jumlah Hafalan")
            ax.set_ylabel("Kelancaran Hafalan")
            st.pyplot(fig)
        else:
            st.warning("❗ Minimal 2 data diperlukan untuk evaluasi.")

# === Tampilan Riwayat Santri ===
elif menu == "Riwayat Santri":
    st.subheader("📄 Riwayat Hafalan Santri")
    nama = st.selectbox("Pilih Nama Santri", ambil_daftar_santri())
    if st.button("Tampilkan Riwayat"):
        data = ambil_data_dari_nama(nama)
        if data:
            df = pd.DataFrame(data)
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
            st.dataframe(df.sort_values(by=["tahun", "bulan"], ascending=False))
        else:
            st.warning("📭 Riwayat tidak ditemukan.")
