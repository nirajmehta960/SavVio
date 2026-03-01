import pandas as pd
from config import Config

def load_data(filepath: str = Config.DATA_PATH) -> pd.DataFrame:
    """
    Loads the featured dataset exported by the Airflow Data Pipeline.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {filepath}")
        print("Make sure the Airflow pipeline has run and exported financial_featured.csv to temp_data/")
        raise

def define_target(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the composite target variable (Y).
    Criteria: Credit Score >= 700 AND Savings Ratio > 3.5 AND Debt Ratio < 3.0.
    """
    target = (
        (df["credit_score"] >= 700) &
        (df["savings_to_income_ratio"] > 3.5) &
        (df["debt_to_income_ratio"] < 3.0)
    ).astype(int)
    
    # Optional: Log the class imbalance
    distribution = target.value_counts(normalize=True)
    print(f"Target Distribution:\n{distribution}")
    return target
