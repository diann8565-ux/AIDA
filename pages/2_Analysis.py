import streamlit as st
import pandas as pd
from database import SessionLocal
from models import ModelMetrics, ConfusionMatrix, FeatureImportance
import plotly.express as px
import plotly.graph_objects as go
from ml_pipeline import ChurnPipeline

st.set_page_config(page_title="Analisis - Telco Churn", page_icon="📈", layout="wide")
st.title("📈 Analisis & Performa Model")

db = SessionLocal()

# Train Trigger
with st.expander("⚙️ Panel Kontrol", expanded=True):
    if st.button("🔄 Latih Ulang Model pada Database Saat Ini"):
        with st.spinner("Melatih XGBoost Classifier dengan SMOTE..."):
            pipeline = ChurnPipeline(db)
            metrics = pipeline.train()
            if metrics:
                st.success("Model berhasil dilatih ulang!")
                st.json(metrics)
            else:
                st.warning("Tidak ada data ditemukan di database untuk dilatih.")

# Fetch latest metrics
latest_run = db.query(ModelMetrics).order_by(ModelMetrics.created_at.desc()).first()

if latest_run:
    st.markdown(f"### Proses Terakhir: {latest_run.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Akurasi (Accuracy)", f"{latest_run.accuracy:.2%}")
    m2.metric("Presisi (Precision)", f"{latest_run.precision:.2%}")
    m3.metric("Recall", f"{latest_run.recall:.2%}")
    m4.metric("Skor F1 (F1 Score)", f"{latest_run.f1_score:.2%}")
    m5.metric("ROC AUC", f"{latest_run.roc_auc:.2%}")
    
    col_viz1, col_viz2 = st.columns(2)
    
    # Confusion Matrix
    cm_data = db.query(ConfusionMatrix).filter_by(run_id=latest_run.run_id).first()
    if cm_data:
        z = [[cm_data.true_negative, cm_data.false_positive],
             [cm_data.false_negative, cm_data.true_positive]]
        
        fig_cm = px.imshow(z, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Diprediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Churn', 'Churn'], y=['Tidak Churn', 'Churn'])
        fig_cm.update_layout(title="Matriks Kebingungan (Confusion Matrix)")
        with col_viz1:
            st.plotly_chart(fig_cm, use_container_width=True)
            
    # Feature Importance
    feats = db.query(FeatureImportance).filter_by(run_id=latest_run.run_id).order_by(FeatureImportance.importance_score.desc()).limit(15).all()
    if feats:
        feat_df = pd.DataFrame([{
            'Fitur': f.feature_name,
            'Kepentingan': f.importance_score
        } for f in feats])
        
        fig_imp = px.bar(feat_df, x='Kepentingan', y='Fitur', orientation='h',
                         title="15 Fitur Paling Berpengaruh", color='Kepentingan')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        with col_viz2:
            st.plotly_chart(fig_imp, use_container_width=True)
            
else:
    st.info("Metrik model tidak ditemukan. Silakan latih model terlebih dahulu.")
