import streamlit as st
import pandas as pd
from database import SessionLocal
from models import Customer
from sqlalchemy.exc import IntegrityError

st.set_page_config(page_title="Unggah Data - Telco Churn", page_icon="📂")
st.title("📂 Unggah Dataset Pelanggan")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Pratinjau:", df.head())
        
        required_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                         'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Kolom yang diperlukan hilang. Diharapkan: {required_cols}")
        else:
            if st.button(f"Impor {len(df)} Baris ke Database"):
                db = SessionLocal()
                progress_bar = st.progress(0)
                
                added_count = 0
                for i, row in df.iterrows():
                    # Handle total charges
                    tc = row['TotalCharges']
                    if isinstance(tc, str):
                        tc = float(tc) if tc.strip() and tc != " " else 0.0
                        
                    customer = Customer(
                        customer_id=str(row['customerID']),
                        gender=row['gender'],
                        senior_citizen=int(row['SeniorCitizen']),
                        partner=row['Partner'],
                        dependents=row['Dependents'],
                        tenure=int(row['tenure']),
                        phone_service=row['PhoneService'],
                        multiple_lines=row['MultipleLines'],
                        internet_service=row['InternetService'],
                        online_security=row['OnlineSecurity'],
                        online_backup=row['OnlineBackup'],
                        device_protection=row['DeviceProtection'],
                        tech_support=row['TechSupport'],
                        streaming_tv=row['StreamingTV'],
                        streaming_movies=row['StreamingMovies'],
                        contract=row['Contract'],
                        paperless_billing=row['PaperlessBilling'],
                        payment_method=row['PaymentMethod'],
                        monthly_charges=float(row['MonthlyCharges']),
                        total_charges=tc,
                        churn=row['Churn']
                    )
                    
                    try:
                        # Simple upsert or ignore check
                        existing = db.query(Customer).filter_by(customer_id=customer.customer_id).first()
                        if not existing:
                            db.add(customer)
                            added_count += 1
                    except Exception:
                        db.rollback()
                        continue
                        
                    if i % 100 == 0:
                        progress_bar.progress(min(i / len(df), 1.0))
                
                db.commit()
                db.close()
                st.success(f"Berhasil mengimpor {added_count} pelanggan baru!")
                
    except Exception as e:
        st.error(f"Error memproses file: {e}")
