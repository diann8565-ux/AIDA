# Dokumentasi Sistem Prediksi Churn Pelanggan Telco

## 1. Pendahuluan
Sistem ini adalah aplikasi web berbasis Machine Learning yang dirancang untuk memprediksi apakah pelanggan layanan telekomunikasi akan berhenti berlangganan (churn) atau tidak. Sistem ini menggunakan algoritma **XGBoost** yang ditingkatkan dengan teknik **SMOTE** untuk menangani ketidakseimbangan data.

Tujuan utama sistem ini adalah membantu tim pemasaran dan retensi pelanggan untuk mengidentifikasi pelanggan berisiko tinggi dan memberikan rekomendasi strategi yang tepat sasaran.

## 2. Fitur Utama
Aplikasi ini memiliki 4 modul utama:

### a. Dashboard (Halaman Utama)
- Formulir input interaktif untuk memasukkan data pelanggan secara manual.
- Prediksi instan probabilitas churn.
- Penentuan tingkat risiko (Rendah, Sedang, Tinggi).
- **Rekomendasi Strategi Otomatis** dalam Bahasa Indonesia yang disesuaikan dengan profil pelanggan.

### b. Analisis Model (Analysis)
- Menampilkan metrik performa model terkini:
  - **Akurasi**: Seberapa sering model benar.
  - **Presisi**: Ketepatan prediksi positif churn.
  - **Recall**: Kemampuan menemukan semua pelanggan yang churn.
  - **F1-Score**: Keseimbangan antara presisi dan recall.
  - **ROC-AUC**: Kemampuan membedakan kelas positif dan negatif.
- Visualisasi **Confusion Matrix** untuk melihat detail kesalahan prediksi.
- Grafik **Feature Importance** untuk mengetahui faktor apa yang paling mempengaruhi keputusan pelanggan (misal: kontrak bulanan, biaya, dll).
- Tombol **Latih Ulang (Retrain)** untuk memperbarui model jika ada data baru di database.

### c. Unggah Dataset (Upload Dataset)
- Memungkinkan pengguna mengunggah file CSV berisi data pelanggan baru.
- Validasi otomatis format kolom.
- Proses impor data ke database PostgreSQL/SQLite.

### d. Riwayat Prediksi (History)
- Tabel log semua prediksi yang pernah dilakukan.
- Filter berdasarkan kategori risiko.
- Pencatatan waktu dan hasil prediksi untuk audit.

## 3. Arsitektur Teknis

### Teknologi yang Digunakan
- **Bahasa Pemrograman**: Python 3.14
- **Framework Web**: Streamlit
- **Database**: PostgreSQL (via Neon) atau SQLite (Lokal)
- **Machine Learning**:
  - `scikit-learn`: Preprocessing (StandardScaler, OneHotEncoder)
  - `imbalanced-learn`: SMOTE (Synthetic Minority Over-sampling Technique)
  - `xgboost`: Algoritma klasifikasi gradient boosting
- **ORM**: SQLAlchemy (untuk interaksi database)
- **Visualisasi**: Plotly & Matplotlib

### Alur Kerja Machine Learning
1. **Data Loading**: Data diambil dari database.
2. **Preprocessing**:
   - Data numerik (Tenure, Charges) distandarisasi.
   - Data kategorikal (Gender, Partner, dll) diubah menjadi angka (One-Hot Encoding).
3. **Resampling (SMOTE)**: Karena data churn biasanya sedikit (minoritas), SMOTE membuat data sintetis agar model belajar seimbang. **Penting:** SMOTE hanya diterapkan pada data latih (training set), bukan data uji.
4. **Modeling**: XGBoost Classifier belajar pola dari data latih.
5. **Evaluasi**: Model diuji pada data tes yang belum pernah dilihat sebelumnya.

## 4. Struktur Database
Sistem menggunakan tabel-tabel berikut:
1. `customers`: Menyimpan data profil pelanggan (demografi, layanan, tagihan).
2. `predictions`: Menyimpan hasil prediksi per pelanggan.
3. `model_metrics`: Menyimpan riwayat performa setiap kali model dilatih ulang.
4. `feature_importance`: Menyimpan skor faktor-faktor yang paling berpengaruh.
5. `retention_recommendations`: Menyimpan template strategi retensi.

## 5. Cara Menjalankan Aplikasi

### Prasyarat
Pastikan Python sudah terinstal. Kemudian instal dependensi:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn streamlit plotly sqlalchemy psycopg2-binary
```

### Menjalankan di Lokal
Buka terminal di folder proyek dan jalankan:
```bash
streamlit run Home.py
```
Aplikasi akan terbuka otomatis di browser (biasanya di `http://localhost:8501`).

### Konfigurasi Database Neon (Opsional)
Untuk menggunakan database cloud Neon, set environment variable sebelum menjalankan aplikasi:

**Di Windows (PowerShell):**
```powershell
$env:DATABASE_URL="postgresql://user:password@endpoint.neon.tech/dbname"
streamlit run Home.py
```

**Di Linux/Mac:**
```bash
export DATABASE_URL="postgresql://user:password@endpoint.neon.tech/dbname"
streamlit run Home.py
```

## 6. Interpretasi Hasil

- **Probabilitas > 70% (Risiko Tinggi)**: Pelanggan sangat mungkin pindah. Butuh tindakan agresif seperti diskon besar atau kontak personal.
- **Probabilitas 30-70% (Risiko Sedang)**: Pelanggan ragu-ragu. Tawarkan insentif layanan (misal: upgrade speed) tanpa memotong harga terlalu banyak.
- **Probabilitas < 30% (Risiko Rendah)**: Pelanggan aman. Cukup jaga kepuasan dengan layanan standar atau loyalty reward kecil.

---
*Dibuat oleh Tim AI Engineer (Trae AI)*
