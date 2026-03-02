import os
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "temp_data", "financial_featured.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "model")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
EXPERIMENT_NAME = "Financial_Wellbeing_Prediction_Linear"

TARGET_CALC_COLS = ["credit_score", "savings_to_income_ratio", "debt_to_income_ratio"]
CATEGORICAL_FEATURES = ["gender", "education_level", "employment_status", "job_title", "has_loan", "loan_type", "region"]
SENSITIVE_FEATURES = ["gender", "region", "education_level"]
COLUMNS_TO_DROP = TARGET_CALC_COLS + ["user_id", "record_date"]

# ==============================================================================
# 2. DATA LOADING & TARGET DEFINITION
# ==============================================================================
def load_and_prepare_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Calculate composite target (1 = Good Financial Condition)
    y = (
        (df["credit_score"] >= 700) &
        (df["saving_to_income_ratio"] > 3.5) &
        (df["debt_to_income_ratio"] < 3.0)
    ).astype(int)
    
    return df, y

# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df):
    print("Engineering features...")
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    existing_cat_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    # Ordinal Encoding
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[existing_cat_cols] = encoder.fit_transform(X[existing_cat_cols])
    
    # Save the encoder for future inference/deployment
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    encoder_path = os.path.join(MODEL_SAVE_DIR, "categorical_encoder.pkl")
    joblib.dump(encoder, encoder_path)
    
    return X, encoder_path

# ==============================================================================
# 4. LLM GUARDRAILS (SIMULATION)
# ==============================================================================
def run_llm_guardrails(prediction, context_data):
    """Simulates passing the model prediction to an LLM for safety checking."""
    print("Running LLM Guardrails Context Check...")
    # A real LLM call (e.g. via Langchain or OpenAI) would go here.
    return True, "Prediction aligns with financial guidance policy."

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load Data
    df, y = load_and_prepare_data()
    
    # 2. Engineer Features
    X, encoder_path = engineer_features(df)
    sensitive_df = df[SENSITIVE_FEATURES]

    # 3. Split Data
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_df, test_size=0.2, random_state=42
    )
    
    # 4. Define Models to Try
    models = [
        {"name": "xgboost", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50}},
        {"name": "linearboost", "params": {"booster": "gblinear", "learning_rate": 0.1, "n_estimators": 50}},
        {"name": "lightgbm", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50}}
    ]

    best_f1 = 0
    best_run_id = None

    # 5. Train & Track Loop
    for config in models:
        model_name = config["name"]
        params = config["params"]
        
        print(f"\n======================================")
        print(f"🎬 Starting MLflow Run: {model_name}")
        print(f"======================================")
        
        with mlflow.start_run(run_name=f"{model_name}_run"):
            # A. Log Config
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            
            # B. Train Model
            if model_name in ["xgboost", "linearboost"]:
                model = XGBClassifier(**params, random_state=42)
            else:
                model = LGBMClassifier(**params, random_state=42)
                
            model.fit(X_train, y_train)
            
            # C. Evaluate Accuracy
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            if hasattr(model, "predict_proba"):
                metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                
            mlflow.log_metrics(metrics)
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

            # D. Evaluate Bias (Fairlearn)
            print("  Running Bias Detection (Fairlearn)...")
            fairness_metrics = {}
            for col in sens_test.columns:
                dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test[col])
                eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test[col])
                fairness_metrics[f"bias_dpd_{col}"] = dpd
                fairness_metrics[f"bias_eod_{col}"] = eod
            mlflow.log_metrics(fairness_metrics)
            
            # E. Simulate LLM
            mlflow.log_param("llm_prompt_template_version", "v1.0")
            is_safe, expl = run_llm_guardrails(y_pred[0], context_data={"region": sens_test.iloc[0]["region"]})
            
            # F. Save Artifacts to MLflow
            print("  Logging artifacts (Model & Encoders) to Storage...")
            mlflow.log_artifact(encoder_path, "preprocessing")
            
            signature = infer_signature(X_train, model.predict(X_train))
            if model_name in ["xgboost", "linearboost"]:
                mlflow.xgboost.log_model(model, "model", signature=signature)
            else:
                mlflow.lightgbm.log_model(model, "model", signature=signature)

            # Keep track of winner
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = mlflow.active_run().info.run_id

    # 6. Model Registry Handoff
    print("\n🎉 Pipeline Complete!")
    if best_run_id:
        print(f"🏆 Best Model Run ID: {best_run_id} (F1: {best_f1:.4f})")
        print(f"This model is ready to be Registered in MLflow for Deployment.")

if __name__ == "__main__":
    main()
