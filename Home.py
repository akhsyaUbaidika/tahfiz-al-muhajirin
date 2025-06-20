import streamlit as st

st.set_page_config(
    page_title="Dashboard Hafalan Santri",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Dashboard Hafalan Santri")
st.markdown("---")

st.subheader("👋 Selamat datang di Aplikasi Analisis Hafalan")
st.markdown("""
Aplikasi ini dirancang untuk memantau dan menganalisis hafalan santri berdasarkan data setoran, kehadiran, dan evaluasi mingguan.

**Navigasi halaman:**  
Gunakan sidebar di sebelah kiri untuk memilih:

- 📊 *Evaluasi Santri*: Untuk melihat hasil clustering dan performa hafalan santri.
- 👪 *Wali Santri*: Tampilan ringkas untuk wali santri.
- 📝 *Input Data*: Masukkan data hafalan santri baru (khusus staf/lembaga).

**Catatan:** Data akan diproses secara otomatis dengan metode *K-Means clustering* untuk mengelompokkan santri berdasarkan kelancaran dan hafalan.
""")

st.info("Silakan pilih halaman di sidebar untuk memulai.")
