from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import pandas as pd
import io
import os
import shutil
from ml_pipeline import ChurnPipeline
from retention_logic import get_retention_strategy
from math import ceil
import httpx
import requests
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Telco Churn AI")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints for AJAX ---

@app.get("/api/customers")
async def get_customers(page: int = 1, limit: int = 20, db: Session = Depends(get_db)):
    total = db.query(models.Customer).count()
    offset = (page - 1) * limit
    customers = db.query(models.Customer).offset(offset).limit(limit).all()
    
    data = [{
        'customer_id': c.customer_id,
        'gender': c.gender,
        'contract': c.contract,
        'tenure': c.tenure,
        'monthly_charges': c.monthly_charges,
        'total_charges': c.total_charges,
        'churn': c.churn
    } for c in customers]
    
    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": ceil(total / limit)
    }

@app.get("/api/training/logs")
async def get_training_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(models.TrainingLog).order_by(models.TrainingLog.timestamp.desc()).limit(limit).all()
    return [{"timestamp": l.timestamp.isoformat(), "level": l.level, "message": l.message} for l in logs]

@app.get("/api/models")
async def get_models(db: Session = Depends(get_db)):
    models_list = db.query(models.ModelRegistry).order_by(models.ModelRegistry.created_at.desc()).all()
    return [{
        "version": m.version,
        "algorithm": m.algorithm,
        "accuracy": m.accuracy,
        "f1_score": m.f1_score,
        "created_at": m.created_at.isoformat(),
        "is_active": m.is_active
    } for m in models_list]

@app.get("/api/models/download/{version}")
async def download_model(version: str, db: Session = Depends(get_db)):
    model_entry = db.query(models.ModelRegistry).filter_by(version=version).first()
    if not model_entry or not os.path.exists(model_entry.filepath):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_entry.filepath, filename=f"churn_model_{version}.pkl")

# --- Background Task ---
def run_training_task():
    db = SessionLocal()
    try:
        pipeline = ChurnPipeline(db)
        pipeline.train_with_tuning(n_iter=5)
    except Exception as e:
        pipeline.log(f"Training Crash: {str(e)}", "ERROR")
    finally:
        db.close()

@app.post("/api/train")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_task)
    return {"message": "Training started in background. Check logs for progress."}

# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request, db: Session = Depends(get_db)):
    latest_run = db.query(models.ModelMetrics).order_by(models.ModelMetrics.created_at.desc()).first()
    cm_data = None
    if latest_run:
        cm_data = db.query(models.ConfusionMatrix).filter_by(run_id=latest_run.run_id).first()
    features = []
    if latest_run:
        features = db.query(models.FeatureImportance).filter_by(run_id=latest_run.run_id).order_by(models.FeatureImportance.importance_score.desc()).limit(10).all()
    
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "metrics": latest_run,
        "cm": cm_data,
        "features": features
    })

@app.get("/smote-analysis", response_class=HTMLResponse)
async def smote_analysis_page(request: Request):
    return templates.TemplateResponse("smote_analysis.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).order_by(models.Prediction.created_at.desc()).limit(100).all()
    return templates.TemplateResponse("history.html", {"request": request, "predictions": predictions})

@app.post("/predict")
async def predict(
    request: Request,
    gender: str = Form(...),
    senior_citizen: int = Form(...),
    partner: str = Form(...),
    dependents: str = Form(...),
    tenure: int = Form(...),
    phone_service: str = Form(...),
    multiple_lines: str = Form(...),
    internet_service: str = Form(...),
    online_security: str = Form(...),
    online_backup: str = Form(...),
    device_protection: str = Form(...),
    tech_support: str = Form(...),
    streaming_tv: str = Form(...),
    streaming_movies: str = Form(...),
    contract: str = Form(...),
    paperless_billing: str = Form(...),
    payment_method: str = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...)
):
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    try:
        pipeline = ChurnPipeline()
        result = pipeline.predict_one(input_data)
        
        # Save prediction to DB
        db = SessionLocal()
        pred_entry = models.Prediction(
            customer_id="MANUAL",
            churn_prediction=result['churn_prediction'],
            churn_probability=result['churn_probability'],
            risk_category=result['risk_category']
        )
        db.add(pred_entry)
        db.commit()
        db.close()
        
        strategy = get_retention_strategy(result['risk_category'], input_data)
        strategy_html = strategy.replace('\n', '<br>').replace('**', '<b>').replace('🚨', '').replace('⚠️', '').replace('✅', '')
        
        # --- AI Recommendation ---
        ai_recommendation = "AI recommendation unavailable."
        try:
            # Prepare prompt
            prompt = f"""
            Sebagai Ahli Strategi Retensi Pelanggan, berikan rekomendasi langsung dan spesifik berdasarkan data ini.
            
            Profil Pelanggan:
            {json.dumps(input_data, indent=2)}
            
            Prediksi Churn: {result['churn_prediction']} (Probabilitas: {result['churn_probability']:.1%})
            
            Berikan jawaban langsung tanpa pembuka, dalam format HTML:
            <h3>1. Analisis Situasi</h3>
            [Jelaskan mengapa dia berisiko/aman dalam 2 kalimat]
            
            <h3>2. Penawaran Retensi Spesifik</h3>
            [Sebutkan 2-3 langkah konkret atau penawaran diskon/layanan yang pas]
            
            <h3>3. Panduan Komunikasi</h3>
            [Contoh kalimat langsung untuk CS saat menghubungi pelanggan ini]
            """
            
            # Hardcoded configuration as requested
            api_key = "ok_LVcqtjxR6TplFpDXOgQBk7jhydGf4NaU"
            api_url = "https://one.apprentice.cyou/api/v1/chat/completions"
            model = "gemini-2.5-flash"
            
            # Switch to synchronous requests to match test script success
            print(f"DEBUG: Calling AI API at {api_url}")
            
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=45.0
            )
            
            if response.status_code == 200:
                ai_data = response.json()
                ai_content = ai_data['choices'][0]['message']['content']
                ai_recommendation = ai_content.replace('\n', '<br>').replace('**', '<b>')
            else:
                print(f"DEBUG: AI Error Status {response.status_code}: {response.text}")
                ai_recommendation = f"Gagal menghubungi AI: {response.status_code} - {response.text}"
                    
        except Exception as ai_err:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: Exception: {str(ai_err)}")
            ai_recommendation = f"AI Error: {str(ai_err)}"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "strategy": strategy_html,
            "ai_recommendation": ai_recommendation,
            "input": input_data
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.get("/api/datasets")
async def list_datasets():
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        return []
    
    files = []
    for f in os.listdir(datasets_dir):
        fp = os.path.join(datasets_dir, f)
        if os.path.isfile(fp):
            files.append({
                "filename": f,
                "size": os.path.getsize(fp),
                "created_at": datetime.fromtimestamp(os.path.getctime(fp)).strftime('%Y-%m-%d %H:%M:%S')
            })
    return sorted(files, key=lambda x: x['created_at'], reverse=True)

@app.get("/api/datasets/{filename}")
async def get_dataset_preview(filename: str):
    file_path = os.path.join("datasets", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Preview first 10 rows
        preview = df.head(10).fillna("").to_dict(orient='records')
        columns = df.columns.tolist()
        return {"filename": filename, "columns": columns, "data": preview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/datasets/{filename}")
async def delete_dataset(filename: str):
    file_path = os.path.join("datasets", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/upload_csv")
async def upload_csv(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        filename = file.filename
        
        # Save file to datasets folder
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            
        file_path = os.path.join(datasets_dir, filename)
        with open(file_path, "wb") as f:
            f.write(contents)
            
        file_stream = io.BytesIO(contents)
        
        df = None
        if filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_stream, encoding='utf-8')
            except UnicodeDecodeError:
                file_stream.seek(0)
                df = pd.read_csv(file_stream, encoding='latin-1')
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_stream)
        else:
            return templates.TemplateResponse("upload.html", {
                "request": request, 
                "error": "Format file tidak didukung. Harap unggah CSV atau Excel (.xlsx)."
            })
        
        required_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                         'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
        
        # Normalize column names to match requirements (case insensitive check could be better, but strict for now)
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             return templates.TemplateResponse("upload.html", {
                 "request": request, 
                 "error": f"Kolom hilang: {', '.join(missing_cols)}"
             })

        # Helper for robust float conversion
        def safe_float(val):
            try:
                if pd.isna(val) or str(val).strip() == "":
                    return 0.0
                return float(val)
            except (ValueError, TypeError):
                # Log or handle specific cases if needed, e.g. datetime
                return 0.0

        added_count = 0
        for i, row in df.iterrows():
            tc = safe_float(row['TotalCharges'])
            mc = safe_float(row['MonthlyCharges'])
                
            customer = models.Customer(
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
                monthly_charges=mc,
                total_charges=tc,
                churn=row['Churn']
            )
            
            try:
                existing = db.query(models.Customer).filter_by(customer_id=customer.customer_id).first()
                if not existing:
                    db.add(customer)
                    added_count += 1
                else:
                    # Optional: Update existing record logic here
                    pass
            except:
                db.rollback()
                continue
        
        db.commit()
        return templates.TemplateResponse("upload.html", {
            "request": request, 
            "success": f"Berhasil mengimpor {added_count} baris data baru."
        })
        
    except Exception as e:
        return templates.TemplateResponse("upload.html", {"request": request, "error": f"Gagal memproses file: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
