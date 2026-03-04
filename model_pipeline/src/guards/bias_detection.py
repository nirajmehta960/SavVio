import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score
import mlflow

def evaluate_bias(y_test, y_pred, sensitive_features: pd.DataFrame):
    """
    Uses Fairlearn to detect bias in the model based on sensitive attributes
    (e.g., region, employment_status). Logs unfairness metrics directly to MLflow.
    Supports both binary and multi-class classification.
    """
    print("\nRunning Bias & Fairness Detection...")
    
    # MLflow metrics dictionary for tracking
    fairness_metrics = {}
    
    for feature_name in sensitive_features.columns:
        sf_col = sensitive_features[feature_name]
        
        # Demographic Parity: Are predictions independent of the sensitive feature?
        dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_col)
        
        # Per-group accuracy difference (works for any number of classes).
        mf = MetricFrame(
            metrics=accuracy_score,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sf_col,
        )
        acc_diff = mf.difference()
        
        print(f"  Feature: {feature_name}")
        print(f"    - Demographic Parity Diff: {dpd:.4f} (Ideal: 0)")
        print(f"    - Accuracy Diff (max-min): {acc_diff:.4f} (Ideal: 0)")
        
        # Record into MLflow with the feature name attached.
        fairness_metrics[f"bias_dpd_{feature_name}"] = dpd
        fairness_metrics[f"bias_acc_diff_{feature_name}"] = acc_diff
        
    # Log it upstream
    mlflow.log_metrics(fairness_metrics)
    
    return fairness_metrics
