import sqlite3
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base, Customer, Prediction, ModelMetrics, ConfusionMatrix, FeatureImportance, RetentionRecommendation, ModelRegistry, TrainingLog
import sys

# Load env variables to get the target DATABASE_URL
load_dotenv()

SQLITE_DB_PATH = "churn.db"
NEON_DB_URL = os.getenv("DATABASE_URL")

if not os.path.exists(SQLITE_DB_PATH):
    print(f"Error: SQLite database '{SQLITE_DB_PATH}' not found.")
    sys.exit(1)

if not NEON_DB_URL or "postgresql" not in NEON_DB_URL:
    print("Error: DATABASE_URL is not set or not a PostgreSQL URL.")
    print("Please check your .env file.")
    sys.exit(1)

print(f"Source: SQLite ({SQLITE_DB_PATH})")
print(f"Target: Neon PostgreSQL")

# 1. Connect to Source (SQLite)
print("Connecting to SQLite...")
sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
sqlite_conn.row_factory = sqlite3.Row
sqlite_cursor = sqlite_conn.cursor()

# 2. Connect to Target (PostgreSQL) using SQLAlchemy
print("Connecting to Neon PostgreSQL...")
engine = create_engine(NEON_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# 3. Create Schema in Target
print("Creating tables in Neon PostgreSQL...")
try:
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")
except Exception as e:
    print(f"Error creating tables: {e}")
    sys.exit(1)

# Clear existing data to prevent duplicates
print("Clearing existing data in target database...")
try:
    # Order matters due to FK constraints
    db.execute(text("TRUNCATE TABLE confusion_matrix, feature_importance, predictions, model_metrics, retention_recommendations, model_registry, training_logs, customers RESTART IDENTITY CASCADE;"))
    db.commit()
    print("Target database cleared.")
except Exception as e:
    print(f"Warning: Could not clear tables (might be first run): {e}")
    db.rollback()

# 4. Migrate Data
tables = [
    (Customer, "customers"),
    (Prediction, "predictions"),
    (ModelMetrics, "model_metrics"),
    (ConfusionMatrix, "confusion_matrix"),
    (FeatureImportance, "feature_importance"),
    (RetentionRecommendation, "retention_recommendations"),
    (ModelRegistry, "model_registry"),
    (TrainingLog, "training_logs")
]

# Helper to ensure MANUAL customer exists
def ensure_manual_customer(db_session):
    manual_customer = db_session.query(Customer).filter_by(customer_id="MANUAL").first()
    if not manual_customer:
        print("  Creating placeholder 'MANUAL' customer for manual predictions...")
        manual_customer = Customer(
            customer_id="MANUAL",
            gender="Unknown",
            senior_citizen=0,
            partner="No",
            dependents="No",
            tenure=0,
            phone_service="No",
            multiple_lines="No",
            internet_service="No",
            online_security="No",
            online_backup="No",
            device_protection="No",
            tech_support="No",
            streaming_tv="No",
            streaming_movies="No",
            contract="Month-to-month",
            paperless_billing="No",
            payment_method="Unknown",
            monthly_charges=0.0,
            total_charges=0.0,
            churn="No"
        )
        db_session.add(manual_customer)
        db_session.commit()

try:
    for model_class, table_name in tables:
        print(f"Migrating table: {table_name}...")
        
        # Pre-migration checks
        if table_name == "predictions":
            ensure_manual_customer(db)

        # Check if table exists in SQLite
        try:
            sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            rows = sqlite_cursor.fetchall()
        except sqlite3.OperationalError:
             print(f"Skipping {table_name} (table not found in SQLite)")
             continue
             
        if not rows:
            print(f"  No data in {table_name}, skipping insert.")
            continue

        print(f"  Found {len(rows)} rows. Inserting into PostgreSQL...")
        
        count = 0
        batch_size = 1000
        batch = []
        
        for row in rows:
            # Convert SQLite Row to dict
            data = dict(row)
            obj = model_class(**data)
            db.add(obj)
            count += 1
            
            # Commit in batches to avoid huge transactions
            if count % batch_size == 0:
                db.commit()
                
        db.commit()
        print(f"  Successfully migrated {count} rows for {table_name}.")

    print("\nMigration completed successfully!")

except Exception as e:
    print(f"\nMigration failed: {e}")
    db.rollback()
finally:
    sqlite_conn.close()
    db.close()
