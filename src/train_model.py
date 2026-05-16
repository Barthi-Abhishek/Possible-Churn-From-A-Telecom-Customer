"""
train_model.py
--------------
Trains multiple classifiers on the preprocessed Telco Churn dataset,
evaluates and compares them, then saves the best model to disk.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

warnings.filterwarnings('ignore')

# Make src importable when running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR      = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH     = os.path.join(MODEL_DIR, 'churn_model.pkl')
FEATURE_PATH   = os.path.join(MODEL_DIR, 'feature_names.pkl')
FIGURES_DIR    = os.path.join(PROJECT_ROOT, 'notebooks', 'figures')

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42


# ── Model definitions ──────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dict of {name: estimator} for all candidate models."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE
        ),
    }


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute and print full evaluation metrics for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model'    : name,
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1 Score' : f1_score(y_test, y_pred),
        'ROC AUC'  : roc_auc_score(y_test, y_proba),
    }
    return metrics


def plot_confusion_matrix(model, name: str, X_test, y_test) -> None:
    """Save a confusion-matrix heatmap for the given model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stay', 'Churn'],
        yticklabels=['Stay', 'Churn'], ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_')
    plt.savefig(os.path.join(FIGURES_DIR, f'cm_{safe_name}.png'), dpi=120)
    plt.close()


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    """Overlay ROC curves for all models and save."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_title('ROC Curves — Model Comparison')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=120)
    plt.close()
    print(f"[plot] ROC curves saved.")


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """Save a horizontal bar chart of the top-N feature importances."""
    if not hasattr(model, 'feature_importances_'):
        print("[feature_importance] Model does not expose feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette('viridis', top_n)
    ax.barh(top_features[::-1], top_values[::-1], color=palette[::-1])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=120)
    plt.close()
    print(f"[feature_importance] Saved. Top 5: {top_features[:5]}")


# ── Main training loop ─────────────────────────────────────────────────────────

def train_and_evaluate() -> None:
    """End-to-end training, evaluation, and model persistence."""

    # 1. Load preprocessed data
    print("=" * 60)
    print("STEP 1 — Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test, feature_names = run_pipeline()

    # 2. Train all candidate models
    print("\n" + "=" * 60)
    print("STEP 2 — Training Models")
    print("=" * 60)
    models       = get_models()
    results      = []
    trained      = {}

    for name, model in models.items():
        print(f"\n  → Training {name} …")
        model.fit(X_train, y_train)
        trained[name] = model
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        print(f"     Accuracy={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1 Score']:.4f}  "
              f"ROC-AUC={metrics['ROC AUC']:.4f}")
        plot_confusion_matrix(model, name, X_test, y_test)

    # 3. Print comparison table
    print("\n" + "=" * 60)
    print("STEP 3 — Model Comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results).set_index('Model')
    print(results_df.to_string())

    # 4. Select best model (by ROC-AUC)
    best_name  = results_df['ROC AUC'].idxmax()
    best_model = trained[best_name]
    print(f"\n  ✓ Best model: {best_name}  (ROC-AUC = {results_df.loc[best_name, 'ROC AUC']:.4f})")

    # Detailed classification report for best model
    y_pred = best_model.predict(X_test)
    print(f"\nClassification Report — {best_name}:\n")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

    # 5. Visualisations
    plot_roc_curves(trained, X_test, y_test)
    plot_feature_importance(best_model, feature_names)

    # 6. Save model + feature names
    print("\n" + "=" * 60)
    print("STEP 4 — Saving Artifacts")
    print("=" * 60)
    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print(f"  Model saved   → {MODEL_PATH}")
    print(f"  Features saved→ {FEATURE_PATH}")

    print("\nTraining complete!")


if __name__ == '__main__':
    train_and_evaluate()
