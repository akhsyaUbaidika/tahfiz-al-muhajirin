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

# Inisialisasi Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("skripsi-akhsya-firebase-adminsdk-fbsvc-5b72d5ad7b.json")
    firebase_admin.initialize_app(cred)
else:
    app = firebase_admin.get_app()

db = firestore.client()

# Ambil data dari Firestore
def ambil_data_dari_firestore(pekan, bulan, tahun):
    query = db.collection("hafalan_santri") \
        .where("pekan", "==", pekan) \
        .where("bulan", "==", bulan) \
        .where("tahun", "==", tahun)
    return [doc.to_dict() for doc in query.stream()]

# Judul halaman
st.title("📗 Tampilan Evaluasi Hafalan Santri untuk Wali")

# Pilihan periode
now = datetime.today()
pekan = st.selectbox("Pekan", [f"Minggu ke-{i}" for i in range(1, 6)], index=(now.day - 1) // 7)
bulan = st.selectbox("Bulan", [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
], index=now.month - 1)
tahun = st.selectbox("Tahun", [str(y) for y in range(2023, 2027)], index=2)

# Tombol untuk tampilkan
if st.button("Tampilkan Evaluasi"):
    df = pd.DataFrame(ambil_data_dari_firestore(pekan, bulan, tahun))
    if len(df) >= 2:
        # Persiapan kolom
        df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else '-')
        bobot_map = {"Pendek": 1.0, "Sedang": 1.5, "Panjang": 2.0}
        df['bobot'] = df['kategori_ayat'].map(bobot_map)
        df['jumlah_hafalan_berbobot'] = df['jumlah_hafalan'] * df['bobot']

        # Clustering
        features = df[['jumlah_hafalan_berbobot', 'kelancaran_total', 'kehadiran']]
        features_scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        df['Klaster'] = kmeans.fit_predict(features_scaled)

        # Label kategori
        order = df.groupby('Klaster')['jumlah_hafalan_berbobot'].mean().sort_values(ascending=False).index
        mapping = {
            order[0]: 'Cepat & Konsisten',
            order[1]: 'Cukup Baik',
            order[2]: 'Perlu Pendampingan'
        }
        df['Kategori'] = df['Klaster'].map(mapping)

        # Tampilkan data
        st.success("✅ Data berhasil ditampilkan.")
        st.dataframe(df[['nama', 'juz', 'kehadiran', 'Kategori']])

        # Pie chart
        fig = px.pie(df, names='Kategori', title='Distribusi Santri Berdasarkan Hasil Evaluasi')
        st.plotly_chart(fig)

        # Scatter chart
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='jumlah_hafalan_berbobot', y='kelancaran_total', hue='Kategori', palette='Set2', s=100)
        ax2.set_xlabel("Jumlah Hafalan")
        ax2.set_ylabel("Kelancaran Hafalan")
        st.pyplot(fig2)
    else:
        st.warning("❗ Minimal 2 data diperlukan untuk evaluasi.")
