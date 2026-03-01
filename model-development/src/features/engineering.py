import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from config import Config
import joblib
import os

def preprocess_features(df: pd.DataFrame, is_training: bool = True):
    """
    Applies feature engineering (Ordinal Encoding, Dropping leaked columns).
    If `is_training` is True, it fits a new OrdinalEncoder and saves it to disk 
    so it can be uploaded to MLflow as an artifact later.
    """
    from config import Config # localized import to avoid circular dependencies

    # Drop columns that are used to compute the Target to prevent data leakage, plus IDs/Dates
    cols_to_drop = [c for c in Config.COLUMNS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Filter numerical/categorical features
    existing_cat_cols = [col for col in Config.CATEGORICAL_FEATURES if col in X.columns]
    
    if is_training:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[existing_cat_cols] = encoder.fit_transform(X[existing_cat_cols])
        
        # Save the encoder so MLflow can register it as an artifact later
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
        joblib.dump(encoder, encoder_path)
        print(f"OrdinalEncoder fitted and saved to {encoder_path}")
    else:
        # NOTE: At inference time, you would load the encoder downloaded from MLflow
        # This branch is just a placeholder for the future FastAPI integration.
        encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
        encoder = joblib.load(encoder_path)
        X[existing_cat_cols] = encoder.transform(X[existing_cat_cols])
        
    return X
