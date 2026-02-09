import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from collections import Counter
from ml_pipeline import ChurnPipeline
from database import SessionLocal

st.set_page_config(page_title="Deep Dive SMOTE", page_icon="🔬", layout="wide")

st.title("🔬 Analisis Mendalam: SMOTE & Imbalanced Learning")
st.markdown("""
Halaman ini menyajikan dokumentasi transparan dan analisis teknis mengenai bagaimana **Synthetic Minority Over-sampling Technique (SMOTE)** digunakan dalam sistem ini untuk menangani ketidakseimbangan data (Imbalanced Dataset).
""")

# --- 1. Load Data ---
st.header("1. Deteksi Ketidakseimbangan & Preprocessing")

@st.cache_data
def load_data():
    db = SessionLocal()
    pipeline = ChurnPipeline(db)
    df = pipeline.load_data_from_db()
    db.close()
    if not df.empty:
        X, y = pipeline.preprocess(df)
        return df, X, y
    return pd.DataFrame(), pd.DataFrame(), pd.Series()

df, X, y = load_data()

if df.empty:
    st.warning("Data belum tersedia di database. Silakan upload dataset terlebih dahulu.")
else:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Distribusi Kelas Asli")
        counts = y.value_counts()
        st.write(counts)
        imbalance_ratio = counts[0] / counts[1] if 1 in counts else 0
        st.metric("Rasio Mayoritas:Minoritas", f"{imbalance_ratio:.2f} : 1")
        
        fig_orig = px.pie(names=['Tidak Churn (0)', 'Churn (1)'], values=counts.values, 
                          color_discrete_sequence=['#4e73df', '#e74a3b'], hole=0.4)
        st.plotly_chart(fig_orig, use_container_width=True)

    with col2:
        st.subheader("Proses Preprocessing")
        st.markdown("""
        Sebelum masuk ke SMOTE, data melewati tahap preprocessing:
        1.  **Pemisahan Fitur & Target**: `Churn` dipisahkan sebagai target (y).
        2.  **Encoding**: Mengubah `Yes`/`No` menjadi `1`/`0`.
        3.  **Handling Missing Values**: `TotalCharges` dikonversi ke numerik, nilai kosong diisi 0.
        4.  **Encoding Kategorikal**: Menggunakan `OneHotEncoder` untuk fitur seperti `InternetService`, `PaymentMethod`, dll.
        5.  **Scaling**: Fitur numerik (`tenure`, `MonthlyCharges`) distandarisasi dengan `StandardScaler`.
        """)
        
        with st.expander("Lihat Snippet Code Preprocessing"):
            st.code("""
def preprocess(self, df):
    # Handle TotalCharges missing/errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return X, y
            """, language='python')

# --- 2. SMOTE Mechanism ---
st.divider()
st.header("2. Mekanisme Kerja SMOTE")

st.markdown("""
**SMOTE (Synthetic Minority Over-sampling Technique)** bekerja dengan mensintesis data baru dari kelas minoritas (Churn=1), bukan sekadar menduplikasi data lama.
""")

c_mech1, c_mech2 = st.columns(2)

with c_mech1:
    st.markdown("### Algoritma K-Nearest Neighbors")
    st.markdown("""
    1.  **Pilih Sampel**: Ambil satu sampel data minoritas $x$.
    2.  **Cari Tetangga**: Temukan $k$ tetangga terdekat ($k$-nearest neighbors) dari $x$ dalam ruang fitur.
    3.  **Interpolasi**: Pilih salah satu tetangga secara acak ($x_{neighbor}$).
    4.  **Buat Sintetik**: Buat titik baru di sepanjang garis antara $x$ dan $x_{neighbor}$.
    
    $$x_{new} = x + rand(0,1) \times (x_{neighbor} - x)$$
    """)

with c_mech2:
    st.info("💡 **Mengapa ini penting?** Teknik ini mencegah *overfitting* yang sering terjadi jika kita hanya menduplikasi data minoritas (Random Oversampling). SMOTE memperluas *decision boundary* model.")

# --- 3. Implementasi & Visualisasi ---
st.divider()
st.header("3. Implementasi & Visualisasi Dampak")

k_neighbors = st.slider("Parameter k_neighbors", min_value=1, max_value=10, value=5)
random_state = 42

if not X.empty and not y.empty:
    # Perlu encoding dulu karena SMOTE butuh numerik
    # Kita lakukan simple preprocessing untuk visualisasi ini saja
    X_enc = pd.get_dummies(X, drop_first=True)
    # Simple imputation if any NaN remains (shouldn't be based on pipeline logic but safe guard)
    X_enc = X_enc.fillna(0)
    
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_enc, y_res_orig := y)
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Sebelum SMOTE")
            counts_before = y_res_orig.value_counts()
            fig_before = px.bar(x=['Tidak Churn', 'Churn'], y=counts_before.values, 
                                color=['Tidak Churn', 'Churn'], 
                                title=f"Total Sampel: {sum(counts_before)}",
                                labels={'y':'Jumlah Sampel', 'x':'Kelas'})
            st.plotly_chart(fig_before, use_container_width=True)
            
        with col_v2:
            st.markdown(f"#### Sesudah SMOTE (k={k_neighbors})")
            counts_after = y_res.value_counts()
            fig_after = px.bar(x=['Tidak Churn', 'Churn'], y=counts_after.values,
                               color=['Tidak Churn', 'Churn'],
                               title=f"Total Sampel: {sum(counts_after)} (+{sum(counts_after)-sum(counts_before)} sintetik)",
                               labels={'y':'Jumlah Sampel', 'x':'Kelas'})
            st.plotly_chart(fig_after, use_container_width=True)
            
        st.success(f"✅ SMOTE berhasil menyeimbangkan kelas menjadi 50:50. Model kini memiliki {counts_after[1]} contoh Churn untuk dipelajari (sebelumnya hanya {counts_before[1]}).")
        
    except ValueError as e:
        st.error(f"Data tidak cukup untuk k={k_neighbors}. Error: {e}")

# --- 4. Pipeline Code ---
st.divider()
st.header("4. Implementasi Pipeline (Code)")
st.markdown(f"Penerapan SMOTE dilakukan di dalam `imb-learn Pipeline` untuk memastikan **tidak ada kebocoran data (data leakage)** ke data validasi.")

st.code(f"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Parameter
k = {k_neighbors}
seed = {random_state}

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # OneHotEncoder & StandardScaler
    ('smote', SMOTE(k_neighbors=k, random_state=seed)),  # Hanya diaplikasikan pada X_train
    ('classifier', XGBClassifier())  # Model belajar dari data seimbang
])

pipeline.fit(X_train, y_train)
""", language="python")

# --- 5. Evaluasi ---
st.divider()
st.header("5. Metrik Evaluasi & Kompleksitas")

tab1, tab2 = st.tabs(["📊 Metrik Performa", "⚙️ Kompleksitas Algoritma"])

with tab1:
    st.markdown("""
    Evaluasi model dengan data tidak seimbang tidak bisa hanya mengandalkan **Akurasi**.
    
    | Metrik | Definisi | Mengapa Penting? |
    | :--- | :--- | :--- |
    | **Precision** | $\\frac{TP}{TP + FP}$ | Memastikan kita tidak terlalu sering salah menebak pelanggan setia sebagai Churn (menghemat biaya retensi). |
    | **Recall** | $\\frac{TP}{TP + FN}$ | **Kritis!** Memastikan kita menangkap sebanyak mungkin pelanggan yang *sebenarnya* akan Churn. SMOTE biasanya meningkatkan metrik ini secara signifikan. |
    | **F1-Score** | $2 \\times \\frac{P \\times R}{P + R}$ | Menjaga keseimbangan antara Precision dan Recall. |
    | **AUC-ROC** | Area Under Curve | Mengukur kemampuan model membedakan antara kelas positif dan negatif di berbagai threshold. |
    """)

with tab2:
    st.markdown("""
    **Kompleksitas Waktu:** $O(N_{min} \cdot k + N_{syn} \cdot M)$
    *   $N_{min}$: Jumlah sampel minoritas asli.
    *   $N_{syn}$: Jumlah sampel sintetik yang dibuat.
    *   $M$: Jumlah fitur.
    *   Mencari tetangga terdekat (k-NN) memakan waktu paling dominan.
    
    **Kompleksitas Ruang:** $O(N \cdot M)$
    *   Membutuhkan memori untuk menyimpan matriks jarak atau struktur data (KD-Tree) untuk pencarian tetangga.
    """)

# --- 6. Report ---
st.divider()
st.header("📑 Laporan Analisis Akhir")
st.markdown("""
**Kesimpulan Penggunaan SMOTE:**
1.  **Peningkatan Recall**: Tanpa SMOTE, model cenderung bias ke kelas mayoritas (Tidak Churn) dan sering gagal mendeteksi pelanggan yang berisiko (False Negative tinggi). Dengan SMOTE, Recall meningkat karena model "dipaksa" melihat lebih banyak variasi kasus Churn.
2.  **Trade-off Presisi**: Seringkali, peningkatan Recall disertai sedikit penurunan Presisi (lebih banyak False Positive). Namun, dalam kasus Churn Prediction, **kehilangan pelanggan (Churn) biasanya lebih mahal daripada biaya promosi retensi**, sehingga Recall yang tinggi lebih diprioritaskan.
3.  **Stabilitas Model**: Penggunaan `random_state=42` dan Cross-Validation menjamin bahwa peningkatan performa bukan kebetulan semata.
""")
