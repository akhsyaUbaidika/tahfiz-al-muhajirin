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
from google.cloud.firestore_v1.base_query import FieldFilter
import os
import json

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
    st.error(f"⚠️ Gagal inisialisasi Firebase: {e}")

db = firestore.client()

# Daftar bobot per juz
juz_bobot = {
    1: ("Sulit", 148), 2: ("Sedang", 111), 3: ("Sedang", 126), 4: ("Sedang", 131), 5: ("Sedang", 123),
    6: ("Sedang", 110), 7: ("Sedang", 149), 8: ("Sedang", 142), 9: ("Sedang", 159), 10: ("Sedang", 127),
    11: ("Sedang", 151), 12: ("Sedang", 170), 13: ("Sedang", 154), 14: ("Sulit", 227), 15: ("Sedang", 185),
    16: ("Sedang", 269), 17: ("Sedang", 190), 18: ("Sedang", 202), 19: ("Sulit", 339), 20: ("Sedang", 171),
    21: ("Sedang", 178), 22: ("Sedang", 169), 23: ("Sulit", 357), 24: ("Sedang", 175), 25: ("Sulit", 246),
    26: ("Sedang", 195), 27: ("Sulit", 399), 28: ("Sedang", 137), 29: ("Sulit", 431), 30: ("Mudah", 564)
}

# Fungsi bantu
def ambil_daftar_santri():
    return sorted([doc.to_dict()['nama'] for doc in db.collection("santri_master").stream()])

def simpan_data_ke_firestore(data):
    doc_id = f"{data['nama']}_{data['bulan']}_{data['tahun']}"
    db.collection("hafalan_santri_al_muhajirin").document(doc_id).set(data)

def hapus_data_dari_firestore(doc_id):
    db.collection("hafalan_santri_al_muhajirin").document(doc_id).delete()

def ambil_data_dari_firestore(bulan, tahun):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin")
            .where(filter=FieldFilter("bulan", "==", bulan))
            .where(filter=FieldFilter("tahun", "==", tahun)).stream()]

def ambil_data_dari_nama(nama):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin")
            .where(filter=FieldFilter("nama", "==", nama)).stream()]

# UI
st.title("📘 Analisis Hafalan Santri")
page = st.sidebar.selectbox("Pilih Halaman", ["Input Data", "Hasil Analisa", "Riwayat Santri"])

bulan_list = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
tahun_list = [str(y) for y in range(2023, 2027)]
today = datetime.today()
default_bulan = today.strftime("%B")
default_tahun = today.strftime("%Y")

# --- INPUT DATA ---
if page == "Input Data":
    st.subheader("✍️ Input Data Hafalan Santri")

    bulan = st.selectbox("Bulan", bulan_list, index=today.month - 1)
    tahun = st.selectbox("Tahun", tahun_list, index=tahun_list.index(default_tahun))
    nama = st.selectbox("Pilih Nama Santri", ambil_daftar_santri())
    juz = st.multiselect("Pilih Juz yang sudah disetor", list(juz_bobot.keys()))
    juz_sedang = st.selectbox("Pilih Juz yang sedang disetor", list(juz_bobot.keys()))
    ayat_disetor = st.number_input("Jumlah Ayat yang Sudah Disetor", min_value=0)
    ayat_sedang_disetor = st.number_input("Jumlah Ayat yang Sedang Disetor", min_value=0)
    kehadiran = st.number_input("Jumlah Kehadiran Bulan Ini (0–15)", min_value=0, max_value=15)

    st.markdown("**Nilai Kelancaran (0–100):**")
    kelancaran_setoran = st.slider("Kelancaran Setoran", 0, 100)
    kelancaran_murojaah = st.slider("Kelancaran Murojaah", 0, 100)
    kelancaran_tadarus = st.slider("Kelancaran Tadarus", 0, 100)

    if st.button("Simpan Data"):
        total_ayat = sum(juz_bobot[j][1] * (2 if juz_bobot[j][0] == "Sulit" else 1.5 if juz_bobot[j][0] == "Sedang" else 1) for j in juz)
        kelancaran_total = round((kelancaran_setoran + kelancaran_murojaah + kelancaran_tadarus) / 3, 2)
        data = {
            "nama": nama,
            "juz": juz,
            "juz_sedang": juz_sedang,
            "jumlah_hafalan": total_ayat,
            "ayat_disetor": ayat_disetor,
            "ayat_sedang_disetor": ayat_sedang_disetor,
            "kehadiran": kehadiran,
            "kelancaran_setoran": kelancaran_setoran,
            "kelancaran_murojaah": kelancaran_murojaah,
            "kelancaran_tadarus": kelancaran_tadarus,
            "kelancaran_total": kelancaran_total,
            "bulan": bulan,
            "tahun": tahun
        }
        simpan_data_ke_firestore(data)
        st.success("✅ Data berhasil disimpan.")

    # Data preview
    df = pd.DataFrame(ambil_data_dari_firestore(bulan, tahun))
    st.subheader("📄 Data Hafalan Santri")
    if not df.empty:
        for i, row in df.iterrows():
            doc_id = f"{row['nama']}_{row['bulan']}_{row['tahun']}"
            with st.expander(f"{row['nama']} | Juz: {row.get('juz', [])}"):
                st.write(row)
                col1, col2 = st.columns([3, 1])
                with col1:
                    konfirmasi = st.checkbox(f"✔️ Yakin ingin hapus data {row['nama']}?", key=f"confirm_{i}")
                with col2:
                    if konfirmasi and st.button("❌ Hapus", key=f"hapus_{i}"):
                        hapus_data_dari_firestore(doc_id)
                        st.success(f"Data {row['nama']} berhasil dihapus.")
                        st.rerun()
    else:
        st.info("Belum ada data untuk periode ini.")

# --- ANALISA ---
elif page == "Hasil Analisa":
    st.subheader("🔍 Hasil Klustering berdasarkan Data")
    bulan = st.selectbox("Bulan", bulan_list, index=today.month - 1)
    tahun = st.selectbox("Tahun", tahun_list, index=tahun_list.index(default_tahun))

    if st.button("Lihat Klustering"):
        df = pd.DataFrame(ambil_data_dari_firestore(bulan, tahun))
        if len(df) >= 2:
            df['jumlah_hafalan_berbobot'] = df['jumlah_hafalan']
            features = df[['jumlah_hafalan_berbobot', 'kelancaran_total', 'kehadiran']]
            features_scaled = StandardScaler().fit_transform(features)
            df['Klaster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(features_scaled)

            order = df.groupby('Klaster')['jumlah_hafalan_berbobot'].mean().sort_values(ascending=False).index
            mapping = {
                order[0]: 'Cepat & Konsisten',
                order[1]: 'Cukup Baik',
                order[2]: 'Perlu Pendampingan'
            }
            df['Kategori'] = df['Klaster'].map(mapping)
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

            st.dataframe(df[['nama', 'jumlah_hafalan', 'kehadiran', 'kelancaran_total', 'Kategori']])
            st.plotly_chart(px.pie(df, names='Kategori', title='Distribusi Klaster'))
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='jumlah_hafalan_berbobot', y='kelancaran_total', hue='Kategori', palette='Set2', s=100)
            st.pyplot(fig)
        else:
            st.warning("❗ Tambahkan minimal 2 data untuk analisa.")

# --- RIWAYAT ---
elif page == "Riwayat Santri":
    st.subheader("📄 Riwayat Hafalan Santri")
    nama = st.selectbox("Pilih Nama", ambil_daftar_santri())
    if st.button("Lihat Riwayat"):
        hasil = ambil_data_dari_nama(nama)
        if hasil:
            df = pd.DataFrame(hasil)
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
            st.dataframe(df.sort_values(by=["tahun", "bulan"], ascending=False))
        else:
            st.warning("Data tidak ditemukan.")
