import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from typing import Dict, Any

def train_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]):
    """
    Trains a model of `model_type` using the provided hyperparameters.
    Supported types: 'xgboost', 'lightgbm', 'linearboost'
    """
    print(f" Training {model_type}...")

    if model_type == 'xgboost':
        # Default XGBoost Tree
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

    elif model_type == 'linearboost':
        # XGBoost with linear booster
        model = XGBClassifier(booster='gblinear', **params, random_state=42)
        model.fit(X_train, y_train)

    elif model_type == 'lightgbm':
        # LightGBM Classifier
        model = LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f" Unsupported model_type: {model_type}")

    return model

def log_model_to_mlflow(model, model_type: str, signature):
    """
    Registers the trained model into the active MLflow run based on the package.
    """
    if model_type in ['xgboost', 'linearboost']:
        mlflow.xgboost.log_model(model, "model", signature=signature)
    elif model_type == 'lightgbm':
        mlflow.lightgbm.log_model(model, "model", signature=signature)
