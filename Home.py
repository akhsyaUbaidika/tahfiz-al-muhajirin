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

# Fungsi bantu
def ambil_daftar_santri():
    return sorted([doc.to_dict()['nama'] for doc in db.collection("santri_master").stream()])

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
page = st.sidebar.selectbox("Pilih Halaman", ["Home","Input Data", "Hasil Analisa", "Riwayat Santri", "Manajemen Santri"])

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


# --- INPUT DATA ---
elif page == "Input Data":
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
            df['jumlah_hafalan_ayat_berbobot'] = df['jumlah_hafalan']
            features = df[['jumlah_hafalan_ayat_berbobot', 'kelancaran_total', 'kehadiran']]
            features_scaled = StandardScaler().fit_transform(features)
            df['Klaster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(features_scaled)

            order = df.groupby('Klaster')['jumlah_hafalan_ayat_berbobot'].mean().sort_values(ascending=False).index
            mapping = {
                order[0]: 'Cepat & Konsisten',
                order[1]: 'Cukup Baik',
                order[2]: 'Perlu Pendampingan'
            }
            df['Kategori'] = df['Klaster'].map(mapping)
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

            df_rename = df.rename(columns={"jumlah_hafalan": "jumlah_hafalan_ayat"})
            st.dataframe(df_rename[['nama', 'jumlah_hafalan_ayat', 'kehadiran', 'kelancaran_total', 'Kategori']])

            st.plotly_chart(px.pie(df, names='Kategori', title='Distribusi Klaster'))
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='jumlah_hafalan_ayat_berbobot', y='kelancaran_total', hue='Kategori', palette='Set2', s=100)
            st.pyplot(fig)

            # Hitung jumlah per kategori
            jumlah_per_kategori = df['Kategori'].value_counts().to_dict()

            # Tampilkan keterangan informatif
            st.markdown(f"""
            **📊 Ringkasan Klasterisasi Santri:**

            - **{jumlah_per_kategori.get('Cepat & Konsisten', 0)} santri** berada dalam kategori **Cepat & Konsisten**: menunjukkan hafalan tinggi, kehadiran baik, dan kelancaran stabil.
            - **{jumlah_per_kategori.get('Cukup Baik', 0)} santri** berada dalam kategori **Cukup Baik**: memiliki capaian yang cukup namun masih bisa ditingkatkan.
            - **{jumlah_per_kategori.get('Perlu Pendampingan', 0)} santri** berada dalam kategori **Perlu Pendampingan**: menunjukkan tantangan dalam hafalan, kehadiran, atau kelancaran yang perlu dibina lebih lanjut.

            📌 _Grafik ini membantu pengelola melihat distribusi santri berdasarkan performa mereka secara objektif menggunakan algoritma K-Means._

            **💡 Kesimpulan Sementara:**  
            Sebagian besar santri menunjukkan hasil yang cukup baik hingga sangat baik, namun tetap terdapat sebagian yang memerlukan bimbingan lebih intensif untuk mencapai hasil maksimal.
            """)

            st.divider()
            st.subheader("🧪 Evaluasi Kualitas Klaster (Langkah Perhitungan)")

            # 0) Salin data yang dipakai untuk evaluasi (tidak mengubah df di atas)
            df_eval = df[['jumlah_hafalan_ayat_berbobot', 'kelancaran_total', 'kehadiran']].copy()

            # Opsional: pembobotan lembut (matikan kalau tidak perlu)
            # df_eval['jumlah_hafalan_ayat_berbobot'] *= 0.9
            # df_eval['kelancaran_total']            *= 1.5
            # df_eval['kehadiran']                   *= 0.75

            # 1) Winsorize (pemotongan nilai ekstrem)
            from scipy.stats.mstats import winsorize
            LIMIT_LOW, LIMIT_HIGH = 0.15, 0.15  # kamu sudah pakai 15% — sesuai hasil terbaikmu
            df_w = df_eval.copy()
            for col in df_w.columns:
                df_w[col] = winsorize(df_w[col], limits=[LIMIT_LOW, LIMIT_HIGH])

            # 2) Standardisasi (mean=0, std=1)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_w)

            # 3) PCA ke 2 dimensi (untuk memudahkan klasterisasi & visual evaluasi)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            expl = pca.explained_variance_ratio_

            # 4) KMeans (tidak mengubah df['Klaster'] utama)
            from sklearn.cluster import KMeans
            kmeans_eval = KMeans(
                n_clusters=3,
                init='k-means++',
                random_state=42,
                n_init=50
            )
            labels_eval = kmeans_eval.fit_predict(X_pca)

            # 5) Skor Silhouette
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(X_pca, labels_eval)

            # Tampilan ringkas (low profile)
            st.caption(f"Winsorize limits: low={LIMIT_LOW:.2f}, high={LIMIT_HIGH:.2f}")
            st.caption(f"PCA explained variance: PC1={expl[0]:.2f}, PC2={expl[1]:.2f} (total {(expl[0]+expl[1]):.2f})")
            st.text(f"Silhouette Score (eval): {sil_score:.3f}")

            # === Detail langkah perhitungan (expandable) ===
            with st.expander("Lihat detail langkah perhitungan (data → winsorize → scaling → PCA → KMeans → Silhouette)"):
                import numpy as np
                st.markdown("**1) Data awal untuk evaluasi** (3 fitur):")
                st.dataframe(df_eval.head())

                st.markdown("**2) Ringkasan statistik sebelum & sesudah winsorize**")
                def summary(df_):
                    return df_.agg(['count','min','mean','std','median','max']).T
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Sebelum winsorize")
                    st.dataframe(summary(df_eval))
                with col2:
                    st.caption("Sesudah winsorize (15% low & high)")
                    st.dataframe(summary(df_w))

                st.markdown("**3) Parameter standardisasi (mean & std per fitur)**")
                means = dict(zip(df_w.columns, scaler.mean_))
                stds  = dict(zip(df_w.columns, scaler.scale_))
                st.json({"mean": {k: float(v) for k,v in means.items()},
                        "std":  {k: float(v) for k,v in stds.items()}})

                st.markdown("**4) PCA (2 komponen)**")
                st.write(f"Explained variance ratio: {expl[0]:.4f}, {expl[1]:.4f} (total {(expl[0]+expl[1]):.4f})")
                st.write("Komponen PCA (arah kombinasi fitur):")
                comp = np.round(pca.components_, 4)
                st.dataframe(
                    pd.DataFrame(comp, index=['PC1','PC2'], columns=df_w.columns)
                )

                st.markdown("**5) Parameter KMeans (evaluasi)**")
                st.json({
                    "n_clusters": int(kmeans_eval.n_clusters),
                    "init": "k-means++",
                    "n_init": int(kmeans_eval.n_init),
                    "random_state": 42
                })

                st.markdown("**6) Hasil akhir evaluasi**")
                st.write(f"Silhouette Score: **{sil_score:.3f}**")
            # 
            st.divider()
            st.subheader("📊 Hasil Klasterisasi Setelah Pra-pemrosesan Data")

            # Gunakan data evaluasi yang sudah melalui proses winsorize + scaling + PCA
            # df_w dan X_pca berasal dari blok evaluasi di atas, jadi langsung kita pakai di sini

            # Jalankan kembali KMeans pada data PCA yang sama (agar konsisten dengan evaluasi)
            kmeans_final = KMeans(
                n_clusters=3,
                init='k-means++',
                random_state=42,
                n_init=50
            )
            labels_final = kmeans_final.fit_predict(X_pca)

            # Tambahkan kolom hasil klaster baru ke df_w (bukan ke df utama agar tidak tumpang tindih)
            df_w['Klaster (Evaluasi)'] = labels_final

            # Tentukan urutan klaster berdasarkan rata-rata hafalan
            order_eval = df_w.groupby('Klaster (Evaluasi)')['jumlah_hafalan_ayat_berbobot'].mean().sort_values(ascending=False).index
            mapping_eval = {
                order_eval[0]: 'Cepat & Konsisten (baru)',
                order_eval[1]: 'Cukup Baik (baru)',
                order_eval[2]: 'Perlu Pendampingan (baru)'
            }
            df_w['Kategori (Evaluasi)'] = df_w['Klaster (Evaluasi)'].map(mapping_eval)

            # Tampilkan tabel perbandingan hasil klasterisasi baru
            st.markdown("### 🔄 Perbandingan hasil klasterisasi setelah pra-pemrosesan:")
            st.dataframe(df_w[['jumlah_hafalan_ayat_berbobot', 'kelancaran_total', 'kehadiran', 'Kategori (Evaluasi)']])

            # Visualisasi pie chart hasil baru
            st.plotly_chart(px.pie(df_w, names='Kategori (Evaluasi)', title='Distribusi Klaster (Setelah Pra-pemrosesan)'))

            # Scatter plot hasil baru (pakai PCA)
            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x=X_pca[:, 0], y=X_pca[:, 1],
                hue=df_w['Kategori (Evaluasi)'],
                palette='Set2', s=100
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            st.pyplot(fig2)

            # Ringkasan hasil baru
            jumlah_per_kategori_eval = df_w['Kategori (Evaluasi)'].value_counts().to_dict()
            st.markdown(f"""
            **📈 Ringkasan Klasterisasi Setelah Pra-pemrosesan**  

            - **{jumlah_per_kategori_eval.get('Cepat & Konsisten (baru)', 0)} santri** berada dalam kategori **Cepat & Konsisten (baru)**.  
            - **{jumlah_per_kategori_eval.get('Cukup Baik (baru)', 0)} santri** berada dalam kategori **Cukup Baik (baru)**.  
            - **{jumlah_per_kategori_eval.get('Perlu Pendampingan (baru)', 0)} santri** berada dalam kategori **Perlu Pendampingan (baru)**.  

            📌 _Distribusi ini berasal dari data yang sudah distandarisasi, dipangkas nilai ekstrem, dan direduksi dimensinya dengan PCA, sehingga menghasilkan pola klaster yang lebih stabil dan objektif._
            """)                
        else:
            st.warning("❗ Tambahkan minimal 2 data untuk analisa.")
        
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


# --- RIWAYAT ---
elif page == "Riwayat Santri":
    st.subheader("📄 Riwayat Hafalan Santri")
    nama = st.selectbox("Pilih Nama", ambil_daftar_santri())
    if st.button("Lihat Riwayat"):
        hasil = ambil_data_dari_nama(nama)
        if hasil:
            df = pd.DataFrame(hasil)
            df.rename(columns={"jumlah_hafalan": "jumlah_hafalan_ayat"}, inplace=True)
            df['juz'] = df['juz'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
            st.dataframe(df.sort_values(by=["tahun", "bulan"], ascending=False))
        else:
            st.warning("Data tidak ditemukan.")
