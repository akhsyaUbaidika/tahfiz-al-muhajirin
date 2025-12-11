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
from sklearn.metrics import silhouette_score


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

# ---------- Fungsi bantu pembobotan & ambil data harian ----------

def faktor_juz(juz: int) -> float:
    """Mengembalikan faktor kesulitan berdasarkan juz."""
    if pd.isna(juz):
        return 1.0
    try:
        j = int(juz)
    except Exception:
        return 1.0
    info = juz_bobot.get(j)
    if not info:
        return 1.0
    tingkat, _ = info
    if tingkat == "Sulit":
        return 2.0
    elif tingkat == "Sedang":
        return 1.5
    return 1.0  # Mudah atau default


def hitung_bobot_hafalan(row) -> float:
    """
    bobot_ayat_dihafal =
        Σ (jumlah_ayat_juz * faktor(juz)) untuk semua juz di 'juz_sudah_dihafal'
      + (ayat_sudah_dihafal * faktor(juz_sedang_disetor))
    """
    total = 0.0

    # Juz yang sudah selesai
    juz_selesai = row.get("juz_sudah_dihafal") or []
    if isinstance(juz_selesai, str):
        try:
            juz_selesai = [int(x.strip()) for x in juz_selesai.split(",") if x.strip()]
        except Exception:
            juz_selesai = []

    for j in juz_selesai:
        try:
            j = int(j)
        except Exception:
            continue
        tingkat, jumlah_ayat = juz_bobot.get(j, ("Mudah", 0))
        f = faktor_juz(j)
        total += jumlah_ayat * f

    # Ayat di juz yang sedang berjalan
    ayat_sudah = float(row.get("ayat_sudah_dihafal") or 0)
    juz_sedang = row.get("juz_sedang_disetor")
    if juz_sedang:
        f_sedang = faktor_juz(juz_sedang)
        total += ayat_sudah * f_sedang

    return total


def hitung_bobot_setoran(row) -> float:
    """bobot_ayat_disetor = ayat_sedang_disetor * faktor(juz_sedang_disetor)."""
    ayat_hari_ini = float(row.get("ayat_sedang_disetor") or 0)
    juz_sedang = row.get("juz_sedang_disetor")
    if not juz_sedang:
        return 0.0
    f = faktor_juz(juz_sedang)
    return ayat_hari_ini * f


def ambil_data_harian(bulan: str, tahun: str) -> pd.DataFrame:
    """
    Ambil data harian untuk bulan & tahun tertentu dari koleksi hafalan_harian_santri.
    Asumsi di Firestore sudah ada field 'bulan' & 'tahun' (string),
    misal bulan = 'Mei', tahun = '2025'.
    """
    docs = (
        db.collection(KOLEKSI_HARIAN)
        .where("bulan", "==", bulan)
        .where("tahun", "==", tahun)
        .stream()
    )
    data = [d.to_dict() for d in docs]
    return pd.DataFrame(data) if data else pd.DataFrame()


# Fungsi bantu
def ambil_daftar_santri():
    return sorted([doc.to_dict()['nama'] for doc in db.collection("santri_master").stream()])

# === Koleksi baru untuk data harian ===
KOLEKSI_HARIAN = "hafalan_harian_santri"  # ganti kalau mau nama lain

def simpan_data_harian_ke_firestore(data):
    """Simpan satu record harian ke koleksi baru."""
    db.collection(KOLEKSI_HARIAN).add(data)

def ambil_data_harian_berdasarkan_tanggal(tanggal_str: str):
    """
    Ambil semua data harian untuk satu tanggal (format 'YYYY-MM-DD').
    Dikembalikan list of dict + doc_id.
    """
    docs = db.collection(KOLEKSI_HARIAN).where("tanggal_str", "==", tanggal_str).stream()
    hasil = []
    for d in docs:
        row = d.to_dict()
        row["doc_id"] = d.id
        hasil.append(row)
    return hasil

def hapus_data_harian_dari_firestore(doc_id: str):
    db.collection(KOLEKSI_HARIAN).document(doc_id).delete()


def simpan_data_ke_firestore(data):
    doc_id = f"{data['nama']}_{data['bulan']}_{data['tahun']}"
    db.collection("hafalan_santri_al_muhajirin_mei_sep_2025").document(doc_id).set(data)

def hapus_data_dari_firestore(doc_id):
    db.collection("hafalan_santri_al_muhajirin_mei_sep_2025").document(doc_id).delete()

def ambil_data_dari_firestore(bulan, tahun):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin_mei_sep_2025")
            .where(filter=FieldFilter("bulan", "==", bulan))
            .where(filter=FieldFilter("tahun", "==", tahun)).stream()]

def ambil_data_dari_nama(nama):
    return [doc.to_dict() for doc in db.collection("hafalan_santri_al_muhajirin_mei_sep_2025")
            .where(filter=FieldFilter("nama", "==", nama)).stream()]

# UI
# st.title("📘 Analisis Hafalan Santri")

# page = st.sidebar.selectbox("Pilih Halaman", ["Home","Input Data", "Hasil Analisa", "Riwayat Santri", "Manajemen Santri"])

page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Input Data", "Hasil Analisa", "Manajemen Santri"]
)


bulan_list = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
tahun_list = [str(y) for y in range(2023, 2027)]
today = datetime.today()
default_bulan = today.strftime("%B")
default_tahun = today.strftime("%Y")

# --- HOME ---
if page == "Home":
    st.title("📘 Dashboard Hafalan Santri")
    st.markdown("---")

    st.subheader("👋 Selamat datang di Aplikasi Analisis Hafalan")
    st.markdown("""
    Aplikasi ini dirancang untuk memantau dan menganalisis hafalan santri berdasarkan data setoran, kehadiran, dan evaluasi mingguan.

    **Navigasi halaman:**  
    Gunakan sidebar di sebelah kiri untuk memilih:

    - 📝 *Input Data*: Masukkan data hafalan santri baru (khusus pembina).
    - 🔍 *Hasil Analisa*: Lihat hasil clustering dan performa hafalan santri.
    - 📄 *Riwayat Santri*: Tampilkan perkembangan hafalan tiap santri.
    - 📋 *Manajemen Santri*: Tambah atau hapus data master santri.

    **Catatan:**  
    Data akan diproses otomatis dengan metode *K-Means clustering* untuk mengelompokkan santri berdasarkan hafalan dan kelancaran.
    """)

    st.info("Silakan pilih halaman di sidebar untuk memulai.")

elif page == "Input Data":
    st.subheader("✍️ Input Data Harian Hafalan Santri")

    # --------- PILIH TANGGAL (DIPAKAI FORM + LIST DI BAWAH) ---------
    today = datetime.today().date()
    tanggal = st.date_input("Tanggal setoran", value=today)
    tanggal_str = tanggal.strftime("%Y-%m-%d")

    hari_idx = tanggal.weekday()
    hari_map = {
        0: "Senin",
        1: "Selasa",
        2: "Rabu",
        3: "Kamis",
        4: "Jumat",
        5: "Sabtu",
        6: "Ahad",
    }
    hari = hari_map.get(hari_idx, "-")
    bulan = bulan_list[tanggal.month - 1]
    tahun = str(tanggal.year)

    st.write(f"Hari: **{hari}**")
    if hari not in ["Senin", "Rabu", "Jumat"]:
        st.info(
            "⚠️ Tanggal ini bukan Senin/Rabu/Jumat. "
            "Tetap boleh diinput, tapi pastikan ini memang hari setoran."
        )

    # --------- FORM INPUT (AUTO RESET SETELAH SUBMIT) ---------
    with st.form("form_input_harian", clear_on_submit=True):
        nama = st.selectbox("Pilih Nama Santri", ambil_daftar_santri())

        st.markdown("### Kehadiran")
        kehadiran_label = st.radio(
            "Status kehadiran",
            options=["Hadir", "Tidak hadir"],
            horizontal=True,
            index=0,
        )
        kehadiran = 1 if kehadiran_label == "Hadir" else 0

        st.markdown("### Hafalan")
        col1, col2 = st.columns(2)

        with col1:
            juz_sudah_dihafal = st.multiselect(
                "Juz sudah dihafal (tuntas)",
                options=list(juz_bobot.keys()),
                help="Pilih semua juz yang sudah selesai dihafal.",
            )

        with col2:
            juz_sedang_disetor = st.selectbox(
                "Juz yang sedang disetor",
                options=list(juz_bobot.keys()),
                help="Juz yang saat ini sedang berjalan/setoran.",
            )

        col3, col4 = st.columns(2)
        with col3:
            ayat_sudah_dihafal = st.number_input(
                "Jumlah ayat yang sudah dihafal di juz yang sedang",
                min_value=0,
                step=1,
                help="Total ayat yang sudah dihafal di juz yang sedang (kumulatif).",
            )
        with col4:
            ayat_sedang_disetor = st.number_input(
                "Jumlah ayat disetor hari ini",
                min_value=0,
                step=1,
                help="Berapa ayat yang disetor pada hari ini.",
            )

        st.markdown("### Kelancaran (0–10)")
        col5, col6, col7 = st.columns(3)
        with col5:
            kelancaran_setoran = st.slider(
                "Kelancaran setoran",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
        with col6:
            kelancaran_murojaah = st.slider(
                "Kelancaran murojaah",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
        with col7:
            kelancaran_tadarus = st.slider(
                "Kelancaran tadarus",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )

        if kehadiran == 0:
            st.caption(
                "Santri **tidak hadir** → nilai **ayat disetor hari ini** dan semua **kelancaran** "
                "akan disimpan 0. Hafalan kumulatif tetap disimpan sesuai input."
            )

        submitted = st.form_submit_button("💾 Simpan Data Harian")

    # --------- PROSES SIMPAN ---------
    if submitted:
        errors = []
        if not nama:
            errors.append("Nama santri belum dipilih.")
        if ayat_sudah_dihafal < 0:
            errors.append("Jumlah ayat sudah dihafal tidak boleh negatif.")
        if ayat_sedang_disetor < 0:
            errors.append("Jumlah ayat disetor hari ini tidak boleh negatif.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # Kalau tidak hadir, set nilai harian ke 0 (selain hafalan kumulatif)
            if kehadiran == 0:
                ayat_sedang_disetor_to_save = 0
                kelancaran_setoran_to_save = 0.0
                kelancaran_murojaah_to_save = 0.0
                kelancaran_tadarus_to_save = 0.0
            else:
                ayat_sedang_disetor_to_save = int(ayat_sedang_disetor)
                kelancaran_setoran_to_save = float(kelancaran_setoran)
                kelancaran_murojaah_to_save = float(kelancaran_murojaah)
                kelancaran_tadarus_to_save = float(kelancaran_tadarus)

            data = {
                "nama": nama,
                "tanggal": datetime(tanggal.year, tanggal.month, tanggal.day),
                "tanggal_str": tanggal_str,
                "hari": hari,
                "bulan": bulan,
                "tahun": tahun,
                "kehadiran": kehadiran,  # 1/0

                "juz_sudah_dihafal": juz_sudah_dihafal,
                "juz_sedang_disetor": int(juz_sedang_disetor),
                "ayat_sudah_dihafal": int(ayat_sudah_dihafal),
                "ayat_sedang_disetor": ayat_sedang_disetor_to_save,

                "kelancaran_setoran": kelancaran_setoran_to_save,
                "kelancaran_murojaah": kelancaran_murojaah_to_save,
                "kelancaran_tadarus": kelancaran_tadarus_to_save,

                "created_at": datetime.utcnow(),
            }

            try:
                simpan_data_harian_ke_firestore(data)
                st.success("✅ Data harian berhasil disimpan.")
            except Exception as e:
                st.error(f"❌ Gagal menyimpan data: {e}")

    # --------- LIST DATA UNTUK TANGGAL YANG DIPILIH ---------
    st.markdown("---")
    st.subheader(f"📄 Data setoran pada tanggal {tanggal_str}")

    hasil_harian = ambil_data_harian_berdasarkan_tanggal(tanggal_str)
    if hasil_harian:
        df_harian = pd.DataFrame(hasil_harian)
        df_harian = df_harian.sort_values("nama")

        for i, row in df_harian.iterrows():
            with st.expander(f"{row['nama']} | Kehadiran: {'Hadir' if row['kehadiran'] == 1 else 'Tidak hadir'}"):
                st.write({
                    "hari": row.get("hari"),
                    "bulan": row.get("bulan"),
                    "tahun": row.get("tahun"),
                    "juz_sudah_dihafal": row.get("juz_sudah_dihafal"),
                    "juz_sedang_disetor": row.get("juz_sedang_disetor"),
                    "ayat_sudah_dihafal": row.get("ayat_sudah_dihafal"),
                    "ayat_sedang_disetor": row.get("ayat_sedang_disetor"),
                    "kelancaran_setoran": row.get("kelancaran_setoran"),
                    "kelancaran_murojaah": row.get("kelancaran_murojaah"),
                    "kelancaran_tadarus": row.get("kelancaran_tadarus"),
                })

                col1, col2 = st.columns([3, 1])
                with col1:
                    konfirmasi = st.checkbox(
                        f"✔️ Yakin ingin hapus data {row['nama']} tanggal {row['tanggal_str']}?",
                        key=f"confirm_harian_{i}"
                    )
                with col2:
                    if konfirmasi and st.button("❌ Hapus", key=f"hapus_harian_{i}"):
                        hapus_data_harian_dari_firestore(row["doc_id"])
                        st.success(f"Data {row['nama']} tanggal {row['tanggal_str']} berhasil dihapus.")
                        st.rerun()
    else:
        st.info("Belum ada data setoran untuk tanggal ini.")


# --- ANALISA ---
elif page == "Hasil Analisa":
    st.title("Analisis Pola Hafalan Al-Qur'an - K-Means (Per Santri)")

    col1, col2, col3 = st.columns([1.2, 1.2, 2])

    with col1:
        bulan = st.selectbox("Bulan", bulan_list, index=today.month - 1)
    with col2:
        tahun = st.selectbox("Tahun", tahun_list, index=tahun_list.index(default_tahun))
    with col3:
        k_default = 3
        k = st.number_input("Jumlah Cluster (K)", min_value=2, max_value=8, value=k_default, step=1)

    st.markdown("---")

    if st.button("🔍 PROSES ANALISIS"):
        # 1. Kumpulkan data
        st.subheader("1. Kumpulkan Data")
        df_raw = ambil_data_harian(bulan, tahun)

        if df_raw.empty:
            st.warning("Data harian untuk periode ini belum tersedia.")
            st.stop()

        st.write(f"Jumlah data mentah: **{len(df_raw)}** baris")
        with st.expander("Lihat data mentah"):
            st.dataframe(df_raw)

        # 2. Tentukan atribut
        st.subheader("2. Tentukan Atribut yang Digunakan")
        st.markdown(
            "- **bobot_ayat_dihafal**  (total hafalan berbobot)\n"
            "- **bobot_ayat_disetor**  (setoran berbobot)\n"
            "- **kelancaran_setoran**  (0–10)\n"
            "- **kelancaran_murojaah** (0–10)\n"
            "- **kelancaran_tadarus**  (0–10)\n"
            "- **kehadiran**           (0 = tidak hadir, 1 = hadir; akan dirata-ratakan per santri)"
        )

        # 3. Screening data
        st.subheader("3. Screening Data")
        df = df_raw.copy()

        required_cols = [
            "nama", "juz_sudah_dihafal", "juz_sedang_disetor", "ayat_sudah_dihafal",
            "ayat_sedang_disetor", "kelancaran_setoran", "kelancaran_murojaah",
            "kelancaran_tadarus", "kehadiran"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Kolom berikut belum ada di koleksi Firestore: {missing}")
            st.stop()

        # Cast numerik
        for col in ["kelancaran_setoran", "kelancaran_murojaah", "kelancaran_tadarus"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Nilai kelancaran di luar 0–10 dianggap anomali -> NaN
        for col in ["kelancaran_setoran", "kelancaran_murojaah", "kelancaran_tadarus"]:
            df.loc[(df[col] < 0) | (df[col] > 10), col] = None

        df["juz_sedang_disetor"] = pd.to_numeric(df["juz_sedang_disetor"], errors="coerce")
        df["ayat_sudah_dihafal"] = pd.to_numeric(df["ayat_sudah_dihafal"], errors="coerce")
        df["ayat_sedang_disetor"] = pd.to_numeric(df["ayat_sedang_disetor"], errors="coerce")

        # Kehadiran: 1 = hadir, 0 = tidak hadir
        df["kehadiran"] = df["kehadiran"].fillna(0)
        df["kehadiran"] = df["kehadiran"].apply(
            lambda x: 1 if str(x).lower() in ["1", "true", "hadir", "ya"] else 0
        )

        before_screen = len(df)
        # Buang baris yang benar-benar rusak (juz_sedang / ayat utama kosong)
        df = df.dropna(
            subset=["juz_sedang_disetor", "ayat_sudah_dihafal", "ayat_sedang_disetor"],
            how="any"
        )
        after_screen = len(df)
        st.write(f"Data setelah screening: **{after_screen}** baris (dari {before_screen}).")

        if after_screen < 2:
            st.error("Data setelah screening terlalu sedikit untuk analisis.")
            st.stop()

        # 4. Transformasi data: pembobotan (harian) + agregasi per santri + normalisasi
        st.subheader("4. Transformasi Data (Pembobotan, Agregasi Per Santri & Normalisasi)")

        # Pembobotan per baris harian
        df["bobot_ayat_dihafal"] = df.apply(hitung_bobot_hafalan, axis=1)
        df["bobot_ayat_disetor"] = df.apply(hitung_bobot_setoran, axis=1)

        fitur = [
            "bobot_ayat_dihafal",
            "bobot_ayat_disetor",
            "kelancaran_setoran",
            "kelancaran_murojaah",
            "kelancaran_tadarus",
            "kehadiran",
        ]

        st.write("Contoh data harian setelah pembobotan (sebelum agregasi):")
        st.dataframe(df[["nama"] + fitur].head())

        # 🔴 DI SINI: agregasi per santri
        df_agg = df.groupby("nama").agg({
            "bobot_ayat_dihafal": "mean",
            "bobot_ayat_disetor": "mean",
            "kelancaran_setoran": "mean",
            "kelancaran_murojaah": "mean",
            "kelancaran_tadarus": "mean",
            "kehadiran": "mean",
        }).reset_index()

        st.write("Ringkasan fitur per santri (rata-rata selama bulan tersebut):")
        st.dataframe(df_agg[["nama"] + fitur])

        if len(df_agg) < int(k):
            st.error(f"Jumlah santri ({len(df_agg)}) lebih kecil dari K={int(k)}. Turunkan nilai K atau kumpulkan lebih banyak data.")
            st.stop()

        df_fitur = df_agg[fitur].astype(float).copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_fitur)

        # 5. Implementasi metode K-Means
        st.subheader("5. Implementasi Metode (K-Means)")

        kmeans = KMeans(n_clusters=int(k), init="k-means++", n_init=20, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        df_agg["cluster"] = labels

        st.success(f"Algoritma K-Means berhasil dijalankan dengan K = {int(k)}.")
        st.write("Centroid (di ruang fitur terstandarisasi):")
        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=fitur)
        st.dataframe(centroids_df)

        # 6. Uji metode: Elbow & Silhouette
        st.subheader("6. Uji Metode (Elbow & Silhouette)")

        max_k = min(8, len(df_agg) - 1)  # batas aman
        sse = []
        sil_scores = []
        k_range = range(2, max_k + 1)
        for kk in k_range:
            km = KMeans(n_clusters=kk, init="k-means++", n_init=10, random_state=42)
            lbl = km.fit_predict(X_scaled)
            sse.append(km.inertia_)
            try:
                sil = silhouette_score(X_scaled, lbl)
            except Exception:
                sil = None
            sil_scores.append(sil)

        elbow_df = pd.DataFrame({"K": list(k_range), "SSE": sse, "Silhouette": sil_scores})

        c1, c2 = st.columns(2)
        with c1:
            st.write("Grafik Elbow (SSE vs K):")
            fig_elbow = px.line(elbow_df, x="K", y="SSE", markers=True)
            st.plotly_chart(fig_elbow, use_container_width=True)
        with c2:
            st.write("Nilai Silhouette per K:")
            st.dataframe(elbow_df)

        # 7. Uji validitas & reliabilitas (sederhana)
        st.subheader("7. Uji Validitas dan Reliabilitas (Sederhana)")

        cluster_summary = df_agg.groupby("cluster")[fitur].mean().reset_index()
        st.write("Rata-rata setiap fitur per cluster (level santri):")
        st.dataframe(cluster_summary)

        try:
            sil_current = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette score untuk K={int(k)}: **{sil_current:.3f}**")
        except Exception:
            st.write("Silhouette score untuk K saat ini tidak dapat dihitung.")

        st.markdown(
            "- **Validitas**: dicek secara kualitatif dari pola rata-rata tiap cluster (apakah sesuai dengan kategori hafalan).\n"
            "- **Reliabilitas sederhana**: model memakai inisialisasi `k-means++` dengan beberapa iterasi (n_init). "
            "Bisa diuji lagi dengan membandingkan bulan berbeda."
        )

        # 8. Hasil & kesimpulan
        st.subheader("8. Hasil dan Kesimpulan")

        # Penamaan kategori berdasarkan rata-rata bobot_ayat_dihafal
        order = cluster_summary.sort_values("bobot_ayat_dihafal", ascending=False)["cluster"].tolist()
        kategori_map = {}
        if len(order) >= 3:
            kategori_map[order[0]] = "Hafalan Tinggi & Stabil"
            kategori_map[order[1]] = "Hafalan Sedang"
            kategori_map[order[2]] = "Perlu Pendampingan"
            for idx, c in enumerate(order[3:], start=3):
                kategori_map[c] = f"Cluster {idx+1}"
        else:
            for idx, c in enumerate(order):
                kategori_map[c] = f"Cluster {idx+1}"

        df_agg["Kategori"] = df_agg["cluster"].map(kategori_map)

        st.write("Tabel hasil clustering per santri:")
        st.dataframe(df_agg[["nama", "Kategori"] + fitur])

        # Visualisasi 2D: bobot_ayat_dihafal vs bobot_ayat_disetor
        fig_scatter = px.scatter(
            df_agg,
            x="bobot_ayat_dihafal",
            y="bobot_ayat_disetor",
            color="Kategori",
            hover_data=["nama", "kelancaran_setoran", "kelancaran_murojaah", "kelancaran_tadarus", "kehadiran"],
            title="Sebaran Santri Berdasarkan Bobot Hafalan & Setoran (Per Santri)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.success("Proses analisis selesai. Tabel & grafik di atas sudah dalam level **santri**, siap untuk pembahasan skripsi. ✅")





elif page == "Manajemen Santri":
    st.subheader("📋 Manajemen Data Santri")

    # === Tambah Santri ===
    st.markdown("#### ➕ Tambah Santri Baru")
    colA, colB = st.columns([2,1])
    with colA:
        nama_baru = st.text_input("Nama Lengkap Santri")
    with colB:
        gender_baru = st.selectbox("Jenis Kelamin", ["L", "P"])

    if st.button("Simpan Santri"):
        if nama_baru.strip():
            db.collection("santri_master").document(nama_baru.strip()).set({
                "nama": nama_baru.strip(),
                "gender": gender_baru
            })
            st.success(f"Santri '{nama_baru}' berhasil ditambahkan.")
            st.rerun()
        else:
            st.warning("❗ Nama tidak boleh kosong.")

    st.divider()

    # === Daftar Santri ===
    st.markdown("#### 📄 Daftar Santri")
    docs = db.collection("santri_master").stream()
    data = [{"nama": d.to_dict().get("nama",""), "gender": d.to_dict().get("gender","-")} for d in docs]
    if data:
        import pandas as pd
        df_master = pd.DataFrame(data).sort_values("nama")
        st.dataframe(df_master, use_container_width=True)

        st.markdown("#### 🗑️ Hapus Santri")
        nama_hapus = st.selectbox("Pilih Nama Santri", df_master["nama"].tolist())

        # Cek apakah punya data hafalan; cegah hapus tanpa konfirmasi keras
        from google.cloud.firestore_v1.base_query import FieldFilter
        terkait = list(
            db.collection("hafalan_santri_al_muhajirin_mei_sep_2025")
              .where(filter=FieldFilter("nama", "==", nama_hapus)).stream()
        )
        jumlah_terkait = len(terkait)
        if jumlah_terkait > 0:
            st.warning(f"⚠️ Ada {jumlah_terkait} data hafalan terkait '{nama_hapus}'. "
                       "Menghapus master akan meninggalkan data hafalan yatim (orphan).")

        col1, col2 = st.columns([3,1])
        with col1:
            konfirmasi = st.checkbox(f"✔️ Saya paham risikonya, tetap hapus '{nama_hapus}' dari master.")
        with col2:
            if konfirmasi and st.button("❌ Hapus"):
                db.collection("santri_master").document(nama_hapus).delete()
                st.success(f"Santri '{nama_hapus}' berhasil dihapus dari master.")
                st.rerun()
    else:
        st.info("Belum ada data santri.")