"""Train the final models and persist inference artifacts to `models/`.

Models
------
1. Logistic-regression-style linear model (SGDClassifier, hinge loss) —
   merged configuration of grid-search candidates 227/263/299
   (identical models: `l1_ratio` is ignored by SGDClassifier when
   `penalty="l1"`). Trained for a single epoch over mini-batches on the
   full training set, exactly as in `04_log_reg_final.ipynb`.
   Kaggle balanced accuracy: 0.88298.

2. CatBoost baseline — configuration from `05_grad_boost.ipynb`
   (tuning in `06_grad_boost_tuning.ipynb` gave no meaningful gain,
   so the simpler baseline is kept). Trained on an 80% stratified split
   with early stopping on the remaining 20%, as in the original notebook.
   Kaggle balanced accuracy: 0.87466.

Artifacts (see `predict.py` for usage)
--------------------------------------
models/log_reg.joblib        dict: model, median, encoder, scaler,
                             feature_order, labels
models/catboost_baseline.cbm CatBoost model in native format
models/catboost_meta.joblib  dict: classes, categorical_features,
                             feature_order

Usage:  python train.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import DataProcessing

TRAIN_CSV = "datasets/train.csv"
MODELS_DIR = "models"

CATEGORICAL_FEATURES = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]

# Merged configuration of grid-search models 227/263/299
# (they differ only in l1_ratio, which SGDClassifier ignores with penalty="l1").
LOG_REG_PARAMS = {
    "alpha": 0.0001,
    "eta0": 0.01,
    "learning_rate": "optimal",
    "loss": "hinge",
    "penalty": "l1",
    "power_t": 0.5,
    "random_state": 42,
    "shuffle": True,
}
BATCH_SIZE = 128

LABELS = {0: "at-risk", 1: "unhealthy", 2: "fit"}  # inverse of DataProcessing translate_dict


def train_log_reg(df: pd.DataFrame) -> None:
    """Single-epoch mini-batch training on the full training set."""
    print("=" * 60)
    print("Training log_reg (SGDClassifier, hinge loss, merged config)")
    print("=" * 60)

    prep = DataProcessing(df, data_type="train")
    # make inference robust to unseen categories (does not affect fitted data)
    prep.encoder.set_params(handle_unknown="ignore")
    train_df = prep.df_prepr()

    X = train_df.drop(columns=["health_condition"])
    y = train_df["health_condition"]

    class_weight = dict(
        zip(np.unique(y), y.shape[0] / (len(np.unique(y)) * np.bincount(y)))
    )
    model = SGDClassifier(**LOG_REG_PARAMS, class_weight=class_weight)

    n_batches = int(np.ceil(len(X) / BATCH_SIZE))
    classes = np.unique(y)
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(X))
        model.partial_fit(X.iloc[start:end], y.iloc[start:end], classes=classes)
        if (i + 1) % 500 == 0 or i == n_batches - 1:
            print(f"  batch {i + 1}/{n_batches}")

    artifact = {
        "model": model,
        "median": prep.median,
        "encoder": prep.encoder,
        "scaler": prep.scaler,
        "feature_order": X.columns.tolist(),
        "labels": LABELS,
    }
    path = os.path.join(MODELS_DIR, "log_reg.joblib")
    joblib.dump(artifact, path)
    print(f"Saved: {path} ({os.path.getsize(path) / 1024:.0f} KB)\n")


def train_catboost(df: pd.DataFrame) -> None:
    """Baseline CatBoost from 05_grad_boost.ipynb (80/20 split, early stopping)."""
    print("=" * 60)
    print("Training CatBoost baseline")
    print("=" * 60)

    # same row filter as in the notebooks: drop samples with 3+ missing values
    df = df[~(df.isnull().sum(axis=1) >= 3)]
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown").astype(str)

    le = LabelEncoder()
    df["health_condition"] = le.fit_transform(df["health_condition"])

    y = df["health_condition"]
    df_train, df_val = train_test_split(df, stratify=y, random_state=42, test_size=0.2)

    X_train = df_train.drop(columns=["health_condition", "id"])
    y_train = df_train["health_condition"]
    X_val = df_val.drop(columns=["health_condition", "id"])
    y_val = df_val["health_condition"]

    model = CatBoostClassifier(
        iterations=1000,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        early_stopping_rounds=50,
        random_seed=42,
        train_dir="catboost_info",  # git-ignored scratch dir for CatBoost logs
        verbose=100,
    )
    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val, y_val),
        verbose=100,
    )

    bal_acc = balanced_accuracy_score(y_val, model.predict(X_val))
    print(f"Validation balanced accuracy: {bal_acc:.4f}")

    cbm_path = os.path.join(MODELS_DIR, "catboost_baseline.cbm")
    model.save_model(cbm_path)
    meta = {
        "classes": le.classes_.tolist(),  # index -> class name
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": X_train.columns.tolist(),
    }
    meta_path = os.path.join(MODELS_DIR, "catboost_meta.joblib")
    joblib.dump(meta, meta_path)
    print(f"Saved: {cbm_path} ({os.path.getsize(cbm_path) / 1024:.0f} KB)")
    print(f"Saved: {meta_path}\n")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    train_log_reg(df)
    train_catboost(df)
    print("Done. Use predict.py for inference.")


if __name__ == "__main__":
    main()
