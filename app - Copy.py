# Re-run the completed app code after environment reset to preserve context
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import plotly.express as px
from datetime import datetime

# Setup database
conn = sqlite3.connect('hafalan.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS hafalan_santri (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama TEXT,
        jumlah_hafalan INTEGER,
        kategori_ayat TEXT,
        kehadiran INTEGER,
        kelancaran_setoran INTEGER,
        kelancaran_murojaah INTEGER,
        kelancaran_tadarus INTEGER,
        kelancaran_total REAL,
        pekan TEXT,
        bulan TEXT,
        tahun TEXT
    )
''')
conn.commit()

# Streamlit UI
st.title("📘 Analisis Hafalan Santri - Versi Lengkap")

# Periode dropdown
st.subheader("🗓️ Periode Setoran Hafalan")
today = datetime.today()
default_bulan = today.strftime("%B")
default_tahun = today.strftime("%Y")
default_pekan = f"Minggu ke-{((today.day - 1) // 7) + 1}"

pekan = st.selectbox("Pekan", [f"Minggu ke-{i}" for i in range(1, 6)], index=((today.day - 1) // 7))
bulan = st.selectbox("Bulan", [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
], index=today.month - 1)
tahun = st.selectbox("Tahun", [str(y) for y in range(2023, 2027)], index=[str(y) for y in range(2023, 2027)].index(default_tahun))

# Form input
st.subheader("✍️ Input Data Hafalan Santri")
with st.form("input_form"):
    nama = st.text_input("Nama Santri")
    jumlah_hafalan = st.number_input("Jumlah Ayat yang Disetorkan", 0)
    kategori_ayat = st.selectbox("Kategori Ayat", ["Pendek", "Sedang", "Panjang"])
    kehadiran = st.selectbox("Jumlah Kehadiran Pekan Ini", [0, 1, 2, 3])

    st.markdown("**Nilai Kelancaran (0–100):**")
    kelancaran_setoran = st.slider("Kelancaran Setoran", 0, 100)
    kelancaran_murojaah = st.slider("Kelancaran Murojaah", 0, 100)
    kelancaran_tadarus = st.slider("Kelancaran Tadarus", 0, 100)

    simpan = st.form_submit_button("Simpan Data")
    if simpan:
        kelancaran_total = round((kelancaran_setoran + kelancaran_murojaah + kelancaran_tadarus) / 3, 2)
        cursor.execute('''
            INSERT INTO hafalan_santri (
                nama, jumlah_hafalan, kategori_ayat, kehadiran,
                kelancaran_setoran, kelancaran_murojaah, kelancaran_tadarus,
                kelancaran_total, pekan, bulan, tahun
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            nama, jumlah_hafalan, kategori_ayat, kehadiran,
            kelancaran_setoran, kelancaran_murojaah, kelancaran_tadarus,
            kelancaran_total, pekan, bulan, tahun
        ))
        conn.commit()
        st.success("✅ Data berhasil disimpan!")

# Tampilkan data
df = pd.read_sql("SELECT * FROM hafalan_santri", conn)
st.subheader("📄 Data Hafalan Santri")
st.dataframe(df)

# Clustering
st.subheader("🔍 Jalankan KMeans Clustering")
if st.button("Proses Clustering"):
    if len(df) >= 2:
        bobot_map = {"Pendek": 1.0, "Sedang": 1.5, "Panjang": 2.0}
        df['bobot'] = df['kategori_ayat'].map(bobot_map)
        df['jumlah_hafalan_berbobot'] = df['jumlah_hafalan'] * df['bobot']

        features = df[['jumlah_hafalan_berbobot', 'kelancaran_total', 'kehadiran']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        df['Klaster'] = kmeans.fit_predict(features_scaled)

        cluster_order = df.groupby('Klaster')['jumlah_hafalan_berbobot'].mean().sort_values(ascending=False).index
        mapping = {cluster_order[0]: 'Cepat & Konsisten', cluster_order[1]: 'Cukup Baik', cluster_order[2]: 'Perlu Pendampingan'}
        df['Kategori'] = df['Klaster'].map(mapping)

        st.success("✅ Clustering selesai.")
        st.dataframe(df[[
            'nama', 'jumlah_hafalan', 'kategori_ayat', 'kehadiran',
            'kelancaran_total', 'jumlah_hafalan_berbobot', 'Kategori'
        ]])

        # Ringkasan
        st.subheader("📊 Ringkasan Kategori Santri")
        summary = df['Kategori'].value_counts().reset_index()
        summary.columns = ['Kategori', 'Jumlah']
        st.dataframe(summary)

        # Visualisasi
        fig = px.pie(df, names='Kategori', title='Distribusi Santri Berdasarkan Klaster')
        st.plotly_chart(fig)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='jumlah_hafalan_berbobot', y='kelancaran_total', hue='Kategori', palette='Set2', s=100)
        st.pyplot(fig2)

        # Ekspor
        df_export = df[[
            'nama', 'jumlah_hafalan', 'kategori_ayat', 'kehadiran',
            'kelancaran_setoran', 'kelancaran_murojaah', 'kelancaran_tadarus',
            'kelancaran_total', 'pekan', 'bulan', 'tahun', 'Kategori'
        ]]
        df_export.to_excel("hasil_klaster.xlsx", index=False)
        with open("hasil_klaster.xlsx", "rb") as f:
            st.download_button("📥 Download Hasil ke Excel", f, file_name="hasil_klaster.xlsx")
    else:
        st.warning("❗ Tambahkan minimal 2 data untuk clustering.")
