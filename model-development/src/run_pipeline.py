import os
import sys

# Ensure the src directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import load_data, define_target
from features.engineering import preprocess_features
from core_models.train import train_model, log_model_to_mlflow
from core_models.evaluate import evaluate_model
from guards.bias_detection import evaluate_bias
from llm.prompt_engin import apply_llm_guardrails

import mlflow
from sklearn.model_selection import train_test_split

def main():
    print("Starting End-to-End ML Pipeline...")
    
    # 1. Initialization and configuration
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    # 2. Data Loading
    df = load_data()
    y = define_target(df)
    
    # 3. Feature Engineering
    X = preprocess_features(df, is_training=True)

    # Note: Extract sensitive features *before* dropping/encoding them for Fairlearn to use
    sensitive_features = df[Config.SENSITIVE_FEATURES]

    # Split dataset
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42
    )

    # 4. Model Training Candidates (We will log a separate MLflow run for each)
    models_to_train = [
        {"name": "xgboost", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "lightgbm", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}},
        {"name": "linearboost", "params": {"learning_rate": 0.1, "n_estimators": 100}}, # no max_depth for linear
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
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            log_model_to_mlflow(model, model_type, signature)
            
            # Log the preprocessing Encoders as well! This is crucial for Deployment.
            encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
            mlflow.log_artifact(encoder_path, "preprocessing")

            # Tracking the best run
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
        # Note: In a live environment, you would uncomment the following block:
        # import mlflow.tracking
        # client = mlflow.tracking.MlflowClient()
        # client.create_registered_model("Financial_Wellbeing_Predictor")
        # mv = client.create_model_version("Financial_Wellbeing_Predictor", model_uri, best_run_id)
        # client.transition_model_version_stage(
        #     name="Financial_Wellbeing_Predictor",
        #     version=mv.version,
        #     stage="Staging"
        # )

if __name__ == "__main__":
    main()
