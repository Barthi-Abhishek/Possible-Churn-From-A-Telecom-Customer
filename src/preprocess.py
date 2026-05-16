"""
preprocess.py
-------------
Data preprocessing pipeline for the Telco Customer Churn dataset.
Handles missing values, encoding, scaling, and feature/label splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_churn.csv')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

TARGET_COL = 'Churn'
DROP_COLS  = ['customerID']


# ── Core functions ─────────────────────────────────────────────────────────────

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Drop irrelevant columns
    - Fix TotalCharges (stored as string with spaces)
    - Impute missing numeric values with median
    - Encode binary target: Yes → 1, No → 0
    """
    df = df.copy()

    # Drop customer ID — not predictive
    df.drop(columns=DROP_COLS, errors='ignore', inplace=True)

    # TotalCharges has whitespace entries that should be NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Impute numeric missing values with column median
    for col in NUMERIC_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[clean_data] Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # Encode target variable
    df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})

    # SeniorCitizen is already 0/1 — leave it
    print(f"[clean_data] Churn distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features (drop first to avoid multicollinearity)."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    print(f"[encode_features] Shape after encoding: {df.shape}")
    return df


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   fit: bool = True,
                   scaler_path: str = SCALER_PATH
                   ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.
    If fit=True the scaler is fitted on X_train and saved; otherwise it is loaded.
    Returns scaled X_train, X_test, and the scaler object.
    """
    # Only scale the numeric columns that exist in the encoded frames
    numeric_in_frame = [c for c in NUMERIC_COLS if c in X_train.columns]

    if fit:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train[numeric_in_frame] = scaler.fit_transform(X_train[numeric_in_frame])
        X_test[numeric_in_frame]  = scaler.transform(X_test[numeric_in_frame])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[scale_features] Scaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train[numeric_in_frame] = scaler.transform(X_train[numeric_in_frame])
        X_test[numeric_in_frame]  = scaler.transform(X_test[numeric_in_frame])
        print(f"[scale_features] Loaded scaler from {scaler_path}")

    return X_train, X_test, scaler


def split_features_labels(df: pd.DataFrame
                           ) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X from target vector y."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def run_pipeline(filepath: str = DATA_PATH,
                 test_size: float = 0.20,
                 random_state: int = 42
                 ) -> tuple:
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)

    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split] Train: {X_train.shape}, Test: {X_test.shape}")

    X_train, X_test, _ = scale_features(X_train, X_test, fit=True)

    feature_names = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names


# ── CLI entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = run_pipeline()
    print(f"\nPreprocessing complete.")
    print(f"  Features : {len(features)}")
    print(f"  Train set: {X_train.shape}")
    print(f"  Test set : {X_test.shape}")
