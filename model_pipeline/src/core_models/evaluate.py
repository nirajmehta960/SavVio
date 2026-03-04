from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import numpy as np
import mlflow

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model against the holdout set and logs metrics to MLflow.
    Supports both binary and multi-class classification.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1
    }
    
    # ROC AUC — handle both binary and multi-class.
    if y_prob is not None:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        metrics["roc_auc"] = roc_auc
        
    print(f"Evaluation Results:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Log all metrics to the active MLflow run.
    mlflow.log_metrics(metrics)
    
    return metrics
