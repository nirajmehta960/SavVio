from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import mlflow

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model against the holdout set and logs metrics to MLflow.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1
    }
    
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        metrics["roc_auc"] = roc_auc
        
    print(f"📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Log all metrics to the active MLflow run
    mlflow.log_metrics(metrics)
    
    return metrics
