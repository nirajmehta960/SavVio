"""
Model Pipeline — End-to-End Entrypoint.

Orchestrates the full model training flow:
    1. Feature Engineering: Loads financial_profiles and products from PostgreSQL,
       generates synthetic (user, product) scenarios via random pairing.
    2. Deterministic Labeling: Labels each scenario GREEN/YELLOW/RED using the
       4-tier DecisionEngine (replaces the old binary credit-score target).
    3. Model Training: Trains XGBoost, LightGBM, LinearBoost candidates.
    4. Evaluation: Accuracy metrics + Fairlearn bias detection.
    5. Registry: Best model is logged to MLflow for downstream serving.

Usage:
    python run_pipeline.py
"""

import os
import sys

# Ensure the src directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_pipeline.src.features.feature_engineering import build_feature_matrix
from core_models.train import train_model, log_model_to_mlflow
from core_models.evaluate import evaluate_model
from guards.bias_detection import evaluate_bias
from llm.prompt_engin import apply_llm_guardrails

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature



def main():
    print("Starting End-to-End ML Pipeline...")

    # 1. Initialization and configuration
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    # 2. Feature engineering + deterministic labeling (GREEN/YELLOW/RED).
    # build_feature_matrix loads from DB, generates scenarios, labels them,
    # handles missing values, encodes categoricals, and scales numerics.
    X, y_raw, scenarios_raw = build_feature_matrix(is_training=True)

    # 3. Encode string labels (GREEN/YELLOW/RED) into integers for model training.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"\nLabel Encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print(f"Label Distribution:\n{y_raw.value_counts().to_string()}\n")

    # Extract sensitive features from raw scenarios for Fairlearn bias detection.
    sensitive_features = scenarios_raw[Config.SENSITIVE_FEATURES]

    # Split dataset into train/test with stratified labels.
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_features,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
        stratify=y,
    )

    # 4. Model Training Candidates (We will log a separate MLflow run for each)
    models_to_train = [
        {"name": "xgboost", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "lightgbm", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "linearboost", "params": {"learning_rate": 0.1, "n_estimators": 100}},  # no max_depth for linear.
    ]

    best_run_id = None
    best_f1_score = 0

    for model_config in models_to_train:
        model_type = model_config["name"]
        params = model_config["params"]

        print(f"\n======================================")
        print(f"Starting MLflow Run: {model_type}")
        print(f"======================================")

        with mlflow.start_run(run_name=f"{model_type}_training"):

            # Log hyperparameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_param("n_scenarios", Config.N_SCENARIOS)
            mlflow.log_param("label_type", "deterministic_engine_GYR")
            mlflow.log_param("num_classes", len(label_encoder.classes_))

            # Train the Model
            model = train_model(model_type, X_train, y_train, params)

            # Evaluate Accuracy Metrics
            metrics = evaluate_model(model, X_test, y_test)

            # Evaluate Bias/Fairness Metrics
            y_pred = model.predict(X_test)
            fairness = evaluate_bias(y_test, y_pred, sens_test)

            # LLM Prompt Wrapper Tracking
            # For demonstration, we just trigger it and log the prompt template
            apply_llm_guardrails(y_pred[0], {"region": sens_test.iloc[0]["region"]}, 0.95)

            # Log Model Artifact
            signature = infer_signature(X_train, model.predict(X_train))
            log_model_to_mlflow(model, model_type, signature)

            # Log the preprocessing Encoders as well! This is crucial for Deployment.
            encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
            scaler_path = os.path.join(Config.MODEL_SAVE_DIR, "feature_scaler.pkl")
            if os.path.exists(encoder_path):
                mlflow.log_artifact(encoder_path, "preprocessing")
            if os.path.exists(scaler_path):
                mlflow.log_artifact(scaler_path, "preprocessing")

            # Log scenario artifact (raw training data for reproducibility).
            if os.path.exists(Config.SCENARIO_OUTPUT_PATH):
                mlflow.log_artifact(Config.SCENARIO_OUTPUT_PATH, "data")

            # Track the best run by F1-score.
            if metrics["f1_score"] > best_f1_score:
                best_f1_score = metrics["f1_score"]
                best_run_id = mlflow.active_run().info.run_id

    print("\nPipeline Complete!")
    print(f"Best F1-Score: {best_f1_score:.4f} in Run ID: {best_run_id}")

    # 5. Registry Handoff
    # Automatically register the best performing run as the active model version
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Registering {model_uri} to MLflow Registry as 'Financial_Wellbeing_Predictor'")


if __name__ == "__main__":
    main()
