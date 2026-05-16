"""
predict.py
----------
Prediction script for the trained churn model.
Accepts raw customer feature data, preprocesses it, and returns a prediction.
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models', 'churn_model.pkl')
SCALER_PATH   = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
FEATURE_PATH  = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')


# ── Loader helpers ─────────────────────────────────────────────────────────────

def load_model():
    """Load the trained classifier from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python src/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_scaler():
    """Load the fitted StandardScaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run `python src/train_model.py` first."
        )
    return joblib.load(SCALER_PATH)


def load_feature_names() -> list:
    """Load the ordered list of feature names used during training."""
    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            f"Feature names not found at {FEATURE_PATH}. "
            "Run `python src/train_model.py` first."
        )
    return joblib.load(FEATURE_PATH)


# ── Preprocessing helper ───────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def preprocess_input(customer: dict,
                     scaler,
                     feature_names: list) -> pd.DataFrame:
    """
    Convert a raw customer dict into a scaled, one-hot-encoded DataFrame
    aligned to the training feature space.

    Parameters
    ----------
    customer     : dict  — raw feature values keyed by column name
    scaler       : fitted StandardScaler
    feature_names: list  — ordered column names from training

    Returns
    -------
    pd.DataFrame of shape (1, n_features)
    """
    df = pd.DataFrame([customer])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to training columns (adds missing dummy cols as 0)
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric columns
    numeric_in_frame = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_in_frame] = scaler.transform(df[numeric_in_frame])

    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_churn(customer: dict) -> dict:
    """
    Predict whether a customer will churn.

    Parameters
    ----------
    customer : dict
        Raw feature dict, e.g.:
        {
            'SeniorCitizen'   : 0,
            'tenure'          : 12,
            'MonthlyCharges'  : 65.0,
            'TotalCharges'    : 780.0,
            'gender'          : 'Male',
            'Partner'         : 'No',
            'Dependents'      : 'No',
            'PhoneService'    : 'Yes',
            'MultipleLines'   : 'No',
            'InternetService' : 'Fiber optic',
            'OnlineSecurity'  : 'No',
            'OnlineBackup'    : 'No',
            'DeviceProtection': 'No',
            'TechSupport'     : 'No',
            'StreamingTV'     : 'Yes',
            'StreamingMovies' : 'Yes',
            'Contract'        : 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod'   : 'Electronic check',
        }

    Returns
    -------
    dict with keys:
        prediction  : int   (0 = Stay, 1 = Churn)
        probability : float (probability of churn)
        message     : str   (human-readable result)
    """
    model         = load_model()
    scaler        = load_scaler()
    feature_names = load_feature_names()

    X = preprocess_input(customer, scaler, feature_names)

    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    message = (
        "⚠️  Customer likely to CHURN"
        if prediction == 1
        else "✅  Customer likely to STAY"
    )

    return {
        'prediction' : prediction,
        'probability': round(probability, 4),
        'message'    : message,
    }


# ── CLI demo ───────────────────────────────────────────────────────────────────

EXAMPLE_CUSTOMER = {
    'SeniorCitizen'   : 0,
    'tenure'          : 5,
    'MonthlyCharges'  : 80.0,
    'TotalCharges'    : 400.0,
    'gender'          : 'Female',
    'Partner'         : 'No',
    'Dependents'      : 'No',
    'PhoneService'    : 'Yes',
    'MultipleLines'   : 'No',
    'InternetService' : 'Fiber optic',
    'OnlineSecurity'  : 'No',
    'OnlineBackup'    : 'No',
    'DeviceProtection': 'No',
    'TechSupport'     : 'No',
    'StreamingTV'     : 'Yes',
    'StreamingMovies' : 'Yes',
    'Contract'        : 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod'   : 'Electronic check',
}


if __name__ == '__main__':
    print("Running prediction on example customer …\n")
    result = predict_churn(EXAMPLE_CUSTOMER)
    print(f"  {result['message']}")
    print(f"  Churn probability : {result['probability']:.2%}")
    print(f"  Raw prediction    : {result['prediction']}  "
          f"(0 = Stay, 1 = Churn)")
