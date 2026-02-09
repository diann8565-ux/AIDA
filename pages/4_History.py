import streamlit as st
import pandas as pd
from database import SessionLocal
from models import Prediction

st.set_page_config(page_title="Riwayat - Telco Churn", page_icon="📜")
st.title("📜 Riwayat Prediksi")

db = SessionLocal()

# Filters
filter_risk = st.multiselect("Filter berdasarkan Kategori Risiko", ["High", "Medium", "Low"], format_func=lambda x: {"High": "Tinggi", "Medium": "Sedang", "Low": "Rendah"}.get(x, x))

query = db.query(Prediction)
if filter_risk:
    query = query.filter(Prediction.risk_category.in_(filter_risk))

predictions = query.order_by(Prediction.created_at.desc()).limit(100).all()

if predictions:
    data = [{
        'Waktu': p.created_at,
        'ID Pelanggan': p.customer_id, 
        'Prediksi': "Ya" if p.churn_prediction == "Yes" else "Tidak",
        'Probabilitas': f"{p.churn_probability:.1%}",
        'Risiko': {"High": "Tinggi", "Medium": "Sedang", "Low": "Rendah"}.get(p.risk_category, p.risk_category)
    } for p in predictions]
    
    df = pd.DataFrame(data)
    
    def color_risk(val):
        color = 'green'
        if val == 'Tinggi': color = 'red'
        elif val == 'Sedang': color = 'orange'
        return f'color: {color}'

    st.dataframe(df.style.applymap(color_risk, subset=['Risiko']), use_container_width=True)
else:
    st.info("Belum ada riwayat prediksi.")
