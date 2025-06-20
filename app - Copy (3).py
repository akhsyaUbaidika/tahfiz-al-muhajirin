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

# Inisialisasi Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("skripsi-akhsya-firebase-adminsdk-fbsvc-5b72d5ad7b.json")  # Ganti dengan path ke file JSON Firebase Anda
    firebase_admin.initialize_app(cred)
else:
    app = firebase_admin.get_app()

# Inisialisasi Firestore
db = firestore.client()

# Daftar juz dan bobot kesulitan
juz_bobot = {
    1: ("Sulit", 148),
    2: ("Sedang", 111),
    3: ("Sedang", 126),
    4: ("Sedang", 131),
    5: ("Sedang", 123),
    6: ("Sedang", 110),
    7: ("Sedang", 149),
    8: ("Sedang", 142),
    9: ("Sedang", 159),
    10: ("Sedang", 127),
    11: ("Sedang", 151),
    12: ("Sedang", 170),
    13: ("Sedang", 154),
    14: ("Sulit", 227),
    15: ("Sedang", 185),
    16: ("Sedang", 269),
    17: ("Sedang", 190),
    18: ("Sedang", 202),
    19: ("Sulit", 339),
    20: ("Sedang", 171),
    21: ("Sedang", 178),
    22: ("Sedang", 169),
    23: ("Sulit", 357),
    24: ("Sedang", 175),
    25: ("Sulit", 246),
    26: ("Sedang", 195),
    27: ("Sulit", 399),
    28: ("Sedang", 137),
    29: ("Sulit", 431),
    30: ("Mudah", 564)
}

# Fungsi untuk menyimpan data ke Firestore
def simpan_data_ke_firestore(data):
    doc_ref = db.collection("hafalan_santri").document(data["nama"])  # Nama santri sebagai ID dokumen
    doc_ref.set(data)

# Fungsi untuk mengambil data berdasarkan waktu (pekan, bulan, tahun)
def ambil_data_dari_firestore(pekan, bulan, tahun):
    query = db.collection("hafalan_santri") \
        .where("pekan", "==", pekan) \
        .where("bulan", "==", bulan) \
        .where("tahun", "==", tahun)
    docs = query.stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return data

# Streamlit UI
st.title("📘 Analisis Hafalan Santri - Versi Lengkap")

# Sidebar untuk navigasi antar halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Input Data", "Hasil Analisa"])

# Input data halaman
if page == "Input Data":
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
        
        # Input Juz yang telah disetor
        juz = st.multiselect("Pilih Juz yang sudah disetor", list(juz_bobot.keys()))
        
        # Input Ayat yang disetor
        ayat_disetor = st.number_input("Jumlah Ayat yang Disetor", min_value=0)
        kategori_ayat = st.selectbox("Kategori Ayat", ["Pendek", "Sedang", "Panjang"])

        kehadiran = st.selectbox("Jumlah Kehadiran Pekan Ini", [0, 1, 2, 3])

        st.markdown("**Nilai Kelancaran (0–100):**")
        kelancaran_setoran = st.slider("Kelancaran Setoran", 0, 100)
        kelancaran_murojaah = st.slider("Kelancaran Murojaah", 0, 100)
        kelancaran_tadarus = st.slider("Kelancaran Tadarus", 0, 100)

        simpan = st.form_submit_button("Simpan Data")
        if simpan:
            # Hitung total ayat berdasarkan juz dan bobot
            total_ayat = 0
            for juz_item in juz:
                bobot = 1 if juz_bobot[juz_item][0] == "Mudah" else 1.5 if juz_bobot[juz_item][0] == "Sedang" else 2
                total_ayat += juz_bobot[juz_item][1] * bobot

            kelancaran_total = round((kelancaran_setoran + kelancaran_murojaah + kelancaran_tadarus) / 3, 2)
            
            # Membuat dictionary untuk data yang akan disimpan
            data = {
                "nama": nama,
                "jumlah_hafalan": total_ayat,
                "kategori_ayat": kategori_ayat,
                "kehadiran": kehadiran,
                "kelancaran_setoran": kelancaran_setoran,
                "kelancaran_murojaah": kelancaran_murojaah,
                "kelancaran_tadarus": kelancaran_tadarus,
                "kelancaran_total": kelancaran_total,
                "pekan": pekan,
                "bulan": bulan,
                "tahun": tahun
            }
            # Menyimpan data ke Firestore
            simpan_data_ke_firestore(data)
            st.success("✅ Data berhasil disimpan ke Firebase!")

    # Mengambil data berdasarkan waktu yang dipilih
    df = pd.DataFrame(ambil_data_dari_firestore(pekan, bulan, tahun))
    st.subheader("📄 Data Hafalan Santri")
    st.dataframe(df)

# Hasil Analisa halaman
if page == "Hasil Analisa":
    today = datetime.today()  # Menambahkan definisi 'today' di halaman Hasil Analisa
    default_tahun = today.strftime("%Y")  # Menambahkan default_tahun
    st.subheader("🔍 Hasil Klustering berdasarkan Data")
    
    pekan = st.selectbox("Pekan", [f"Minggu ke-{i}" for i in range(1, 6)], index=((today.day - 1) // 7))
    bulan = st.selectbox("Bulan", [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ], index=today.month - 1)
    tahun = st.selectbox("Tahun", [str(y) for y in range(2023, 2027)], index=[str(y) for y in range(2023, 2027)].index(default_tahun))

    # Menampilkan tombol untuk menjalankan klustering
    if st.button("Lihat Klustering"):
        df = pd.DataFrame(ambil_data_dari_firestore(pekan, bulan, tahun))
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

            st.success("✅ Klustering selesai.")
            st.dataframe(df[[ 
                'nama', 'jumlah_hafalan', 'kategori_ayat', 'kehadiran',
                'kelancaran_total', 'jumlah_hafalan_berbobot', 'Kategori'
            ]])

            # Visualisasi
            fig = px.pie(df, names='Kategori', title='Distribusi Santri Berdasarkan Klaster')
            st.plotly_chart(fig)

            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x='jumlah_hafalan_berbobot', y='kelancaran_total', hue='Kategori', palette='Set2', s=100)
            st.pyplot(fig2)
        else:
            st.warning("❗ Tambahkan minimal 2 data untuk clustering.")
