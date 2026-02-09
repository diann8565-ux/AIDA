-- SQL Migration Script for Neon (PostgreSQL)

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR PRIMARY KEY,
    gender VARCHAR,
    senior_citizen INTEGER,
    partner VARCHAR,
    dependents VARCHAR,
    tenure INTEGER,
    phone_service VARCHAR,
    multiple_lines VARCHAR,
    internet_service VARCHAR,
    online_security VARCHAR,
    online_backup VARCHAR,
    device_protection VARCHAR,
    tech_support VARCHAR,
    streaming_tv VARCHAR,
    streaming_movies VARCHAR,
    contract VARCHAR,
    paperless_billing VARCHAR,
    payment_method VARCHAR,
    monthly_charges FLOAT,
    total_charges FLOAT,
    churn VARCHAR
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    churn_prediction VARCHAR,
    churn_probability FLOAT,
    risk_category VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    run_id SERIAL PRIMARY KEY,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS confusion_matrix (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES model_metrics(run_id),
    true_negative INTEGER,
    false_positive INTEGER,
    false_negative INTEGER,
    true_positive INTEGER
);

CREATE TABLE IF NOT EXISTS feature_importance (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES model_metrics(run_id),
    feature_name VARCHAR,
    importance_score FLOAT
);

CREATE TABLE IF NOT EXISTS retention_recommendations (
    id SERIAL PRIMARY KEY,
    risk_category VARCHAR UNIQUE,
    recommendation_text TEXT
);

-- Insert default retention strategies
INSERT INTO retention_recommendations (risk_category, recommendation_text) VALUES 
('High', 'Immediate retention call. Offer discount or contract upgrade.'),
('Medium', 'Targeted promotion. Offer bonus services.'),
('Low', 'Loyalty program. Upselling opportunity.')
ON CONFLICT (risk_category) DO NOTHING;
