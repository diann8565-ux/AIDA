import streamlit as st
import pandas as pd
from database import SessionLocal
from ml_pipeline import ChurnPipeline
from models import Customer
from retention_logic import get_retention_strategy
import plotly.graph_objects as go

st.set_page_config(
    page_title="Telco Churn AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Academic/Clean" look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Sistem Prediksi Churn Pelanggan Telco")
st.markdown("### Dashboard Machine Learning Tingkat Lanjut")

st.info("Gunakan navigasi di sidebar untuk Mengunggah Data, Menganalisis Model, atau melihat Riwayat.")

# Quick Dashboard for Individual Prediction
st.header("⚡ Analisis Pelanggan Instan")

col1, col2, col3 = st.columns(3)

# Mapping Dictionaries for Display vs Model Values
gender_map = {"Laki-laki": "Male", "Perempuan": "Female"}
yes_no_map = {"Ya": "Yes", "Tidak": "No"}
internet_map = {"DSL": "DSL", "Fiber optic": "Fiber optic", "Tidak Ada": "No"}
service_map = {"Tidak ada layanan internet": "No internet service", "Tidak": "No", "Ya": "Yes"}
multiple_lines_map = {"Tidak ada layanan telepon": "No phone service", "Tidak": "No", "Ya": "Yes"}
contract_map = {"Bulanan (Month-to-month)": "Month-to-month", "Satu Tahun": "One year", "Dua Tahun": "Two year"}
payment_map = {
    "Cek Elektronik": "Electronic check",
    "Cek Pos": "Mailed check",
    "Transfer Bank (Otomatis)": "Bank transfer (automatic)",
    "Kartu Kredit (Otomatis)": "Credit card (automatic)"
}

with col1:
    gender_disp = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
    senior = st.selectbox("Warga Senior (Lansia)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    partner_disp = st.selectbox("Memiliki Pasangan", list(yes_no_map.keys()))
    dependents_disp = st.selectbox("Memiliki Tanggungan", list(yes_no_map.keys()))
    tenure = st.slider("Masa Berlangganan (Bulan)", 0, 72, 12)

with col2:
    phone_disp = st.selectbox("Layanan Telepon", list(yes_no_map.keys()))
    lines_disp = st.selectbox("Saluran Ganda (Multiple Lines)", list(multiple_lines_map.keys()))
    internet_disp = st.selectbox("Layanan Internet", list(internet_map.keys()))
    
    # Context-aware filtering for service options usually not needed if handled by backend, but let's keep simple mapping
    security_disp = st.selectbox("Keamanan Online", list(service_map.keys()))
    backup_disp = st.selectbox("Backup Online", list(service_map.keys()))

with col3:
    contract_disp = st.selectbox("Kontrak", list(contract_map.keys()))
    billing_disp = st.selectbox("Tagihan Tanpa Kertas", list(yes_no_map.keys()))
    payment_disp = st.selectbox("Metode Pembayaran", list(payment_map.keys()))
    monthly = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Biaya ($)", 0.0, 10000.0, monthly * tenure)

# Hidden/Additional fields defaults
tech_disp = st.selectbox("Dukungan Teknis", list(service_map.keys()))
device_disp = st.selectbox("Proteksi Perangkat", list(service_map.keys()))
tv_disp = st.selectbox("Streaming TV", list(service_map.keys()))
movies_disp = st.selectbox("Streaming Film", list(service_map.keys()))

# Construct input dictionary with Model Values
input_data = {
    'gender': gender_map[gender_disp],
    'SeniorCitizen': senior,
    'Partner': yes_no_map[partner_disp],
    'Dependents': yes_no_map[dependents_disp],
    'tenure': tenure,
    'PhoneService': yes_no_map[phone_disp],
    'MultipleLines': multiple_lines_map[lines_disp],
    'InternetService': internet_map[internet_disp],
    'OnlineSecurity': service_map[security_disp],
    'OnlineBackup': service_map[backup_disp],
    'DeviceProtection': service_map[device_disp],
    'TechSupport': service_map[tech_disp],
    'StreamingTV': service_map[tv_disp],
    'StreamingMovies': service_map[movies_disp],
    'Contract': contract_map[contract_disp],
    'PaperlessBilling': yes_no_map[billing_disp],
    'PaymentMethod': payment_map[payment_disp],
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

if st.button("🚀 Prediksi Probabilitas Churn"):
    try:
        pipeline = ChurnPipeline()
        result = pipeline.predict_one(input_data)
        
        st.divider()
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            st.markdown("### Hasil Prediksi")
            risk_color = "red" if result['risk_category'] == "High" else "orange" if result['risk_category'] == "Medium" else "green"
            
            # Translate Risk Output
            risk_trans = {"High": "Tinggi", "Medium": "Sedang", "Low": "Rendah"}
            pred_trans = {"Yes": "Ya (Akan Churn)", "No": "Tidak (Bertahan)"}
            
            risk_display = risk_trans.get(result['risk_category'], result['risk_category'])
            pred_display = pred_trans.get(result['churn_prediction'], result['churn_prediction'])

            st.markdown(f"<h1 style='color:{risk_color};'>{result['churn_probability']:.1%}</h1>", unsafe_allow_html=True)
            st.markdown(f"**Tingkat Risiko:** <span style='color:{risk_color}; font-weight:bold'>{risk_display}</span>", unsafe_allow_html=True)
            st.markdown(f"**Prediksi:** {pred_display}")
            
        with r_col2:
            st.markdown("### 🛡️ Strategi Retensi yang Direkomendasikan")
            strategy = get_retention_strategy(result['risk_category'], input_data)
            st.success(strategy)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}. Pastikan model sudah dilatih di halaman 'Analisis'.")
