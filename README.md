# Telco Churn Prediction AI

Aplikasi berbasis web untuk memprediksi potensi pelanggan berhenti berlangganan (Churn) menggunakan Machine Learning (XGBoost) dan memberikan rekomendasi strategi retensi otomatis menggunakan Generative AI (Gemini).

## 🚀 Fitur Utama

1.  **Analisis Pelanggan Instan**: Input data pelanggan manual dan dapatkan prediksi churn beserta probabilitasnya secara real-time.
2.  **Rekomendasi AI Cerdas**: Mendapatkan saran strategi retensi yang dipersonalisasi langsung dari AI (berperan sebagai Konsultan Senior).
3.  **Manajemen Dataset**: Upload dan kelola dataset CSV/Excel untuk pelatihan model.
4.  **Auto-Preprocessing**: Sistem otomatis mengubah data teks (String) menjadi angka (Numeric/One-Hot Encoding) siap pakai.
5.  **Imbalanced Learning (SMOTE)**: Menangani ketidakseimbangan data churn secara otomatis untuk meningkatkan akurasi deteksi pelanggan yang berisiko.
6.  **Database Cloud**: Terintegrasi penuh dengan PostgreSQL (Neon DB) untuk penyimpanan data yang aman dan terpusat.

## 🛠️ Persyaratan Sistem

*   **Python**: Versi 3.9 atau lebih baru.
*   **Database**: PostgreSQL (Disarankan menggunakan Neon Tech).
*   **Koneksi Internet**: Diperlukan untuk akses API AI dan Database Cloud.

## 📦 Cara Instalasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Persiapkan Project
Download atau Clone repository ini ke folder komputer Anda.

### 2. Buat Virtual Environment (Disarankan)
Buka terminal/command prompt di folder project dan jalankan:

```bash
# Untuk Windows
python -m venv venv
.\venv\Scripts\activate

# Untuk Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install semua library yang dibutuhkan menggunakan pip:

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment (.env)
1.  Buat file baru bernama `.env` (atau duplikat dari `.env.example`).
2.  Isi konfigurasi berikut (sesuaikan dengan kredensial Anda):

```ini
# Database Neon PostgreSQL
DATABASE_URL=postgresql://neondb_owner:password_anda@host_anda/neondb?sslmode=require

# API Key AI (Dapatkan key Anda)
AI_API_KEY=API_KEY_ANDA_DISINI
AI_MODEL=gemini-2.5-flash
AI_API_URL=https://one.apprentice.cyou/api/v1/chat/completions
```

> **Catatan**: Pastikan `DATABASE_URL` menggunakan format `postgresql://`. Jika Anda menyalin dari dashboard Neon yang menggunakan `postgres://`, sistem akan otomatis memperbaikinya, namun lebih baik langsung disesuaikan.

## ▶️ Cara Menjalankan Aplikasi

Setelah instalasi selesai, jalankan perintah berikut di terminal:

```bash
python main.py
```

Atau menggunakan Uvicorn langsung:
```bash
uvicorn main:app --reload
```

Buka browser dan akses alamat: **http://localhost:8000**

## 📂 Struktur Folder

*   `main.py`: File utama aplikasi (Backend FastAPI).
*   `ml_pipeline.py`: Logika Machine Learning (Preprocessing, SMOTE, Training, Prediction).
*   `models.py`: Definisi tabel database (SQLAlchemy).
*   `database.py`: Konfigurasi koneksi database.
*   `templates/`: File HTML untuk tampilan antarmuka (Frontend).
*   `datasets/`: Folder penyimpanan file CSV/Excel yang diupload.
*   `pages/`: (Opsional) Halaman tambahan untuk analisis Streamlit.

## 🤖 Cara Kerja AI (Singkat)
1.  **Input**: User memasukkan data (misal: Gender="Male").
2.  **Encoding**: Sistem otomatis mengubahnya menjadi `[0, 1]`.
3.  **Prediksi**: Model XGBoost menghitung peluang Churn.
4.  **Saran**: Hasil prediksi dikirim ke API Gemini untuk dibuatkan narasi strategi retensi.

---
*Dibuat dengan ❤️ menggunakan FastAPI, XGBoost, dan Neon PostgreSQL.*
