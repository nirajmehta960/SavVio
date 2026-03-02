import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "temp_data", "financial_featured.csv")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "artifacts")
    
    # --- MLflow Configuration ---
    # These match the docker-compose environment variables you established
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    EXPERIMENT_NAME = "Financial_Wellbeing_Prediction"

    # --- Target Definition ---
    # The columns we use to dynamically calculate our target "Good Financial Condition" Y variable
    TARGET_CALC_COLS = ["credit_score", "saving_to_income_ratio", "debt_to_income_ratio"]
    
    # --- Feature Engineering ---
    CATEGORICAL_FEATURES = ["gender", "education_level", "employment_status", "job_title", "has_loan", "loan_type", "region"]
    COLUMNS_TO_DROP = TARGET_CALC_COLS + ["user_id", "record_date"]

    # --- Sensitive Features for Bias Tracking (Fairlearn) ---
    SENSITIVE_FEATURES = ["gender", "region", "education_level"]
