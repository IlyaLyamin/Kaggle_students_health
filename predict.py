"""Inference with the artifacts produced by `train.py` (folder `models/`).

Batch mode (CSV in, CSV out):
    python predict.py --model log_reg --csv datasets/test.csv --out predictions.csv

Single-sample mode (interactive prompt):
    python predict.py --model catboost --interactive

`--model` accepts `log_reg` (default) or `catboost`.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from data_preprocessing import DataProcessing

MODELS_DIR = "models"

NUMERIC_FEATURES = {
    "sleep_duration": "hours of sleep per day, e.g. 7.5",
    "heart_rate": "resting heart rate, bpm, e.g. 70",
    "bmi": "body mass index, e.g. 22.5",
    "calorie_expenditure": "daily calorie expenditure, e.g. 2500",
    "step_count": "daily steps, e.g. 8000",
    "exercise_duration": "minutes of exercise per day, e.g. 45",
    "water_intake": "daily water intake, e.g. 2.0",
}
CATEGORICAL_FEATURES = {
    "diet_type": ["veg", "non-veg", "balanced"],
    "stress_level": ["low", "medium", "high"],
    "sleep_quality": ["poor", "average", "good"],
    "physical_activity_level": ["sedentary", "moderate", "active"],
    "smoking_alcohol": ["yes", "occasional", "no"],
    "gender": ["female", "male", "other"],
}


def load_log_reg():
    return joblib.load(os.path.join(MODELS_DIR, "log_reg.joblib"))


def load_catboost():
    model = CatBoostClassifier()
    model.load_model(os.path.join(MODELS_DIR, "catboost_baseline.cbm"))
    meta = joblib.load(os.path.join(MODELS_DIR, "catboost_meta.joblib"))
    return model, meta


def predict_log_reg(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Preprocess raw rows with the saved artifacts and predict class labels."""
    ids = df["id"].copy() if "id" in df.columns else None
    work = df.drop(columns=["health_condition"], errors="ignore")

    prep = DataProcessing(work, data_type="test")
    prep.median = artifact["median"]
    prep.encoder = artifact["encoder"]
    prep.scaler = artifact["scaler"]
    X = prep.df_prepr().reindex(columns=artifact["feature_order"])

    preds = artifact["model"].predict(X)
    labels = np.array([artifact["labels"][int(p)] for p in preds])
    return labels, ids


def predict_catboost(df: pd.DataFrame, model, meta: dict) -> np.ndarray:
    ids = df["id"].copy() if "id" in df.columns else None
    work = df.drop(columns=["health_condition", "id"], errors="ignore").copy()

    work[meta["categorical_features"]] = (
        work[meta["categorical_features"]].fillna("Unknown").astype(str)
    )
    work = work.reindex(columns=meta["feature_order"])

    preds = model.predict(work).astype(int).ravel()
    classes = meta["classes"]
    labels = np.array([classes[p] for p in preds])
    return labels, ids


def read_single_sample() -> pd.DataFrame:
    """Prompt the user for one sample. Empty input = missing value."""
    print("\nEnter feature values (press Enter to skip / mark as missing):")
    row = {}
    for name, hint in NUMERIC_FEATURES.items():
        raw = input(f"  {name} ({hint}): ").strip()
        row[name] = float(raw) if raw else np.nan
    for name, allowed in CATEGORICAL_FEATURES.items():
        raw = input(f"  {name} {allowed}: ").strip().lower()
        row[name] = raw if raw else np.nan
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["log_reg", "catboost"], default="log_reg")
    parser.add_argument("--csv", help="path to a CSV with raw feature columns")
    parser.add_argument("--out", default="predictions.csv", help="output CSV (batch mode)")
    parser.add_argument("--interactive", action="store_true",
                        help="prompt for a single sample instead of reading a CSV")
    args = parser.parse_args()

    if args.model == "log_reg":
        artifact = load_log_reg()
        predict_fn = lambda df: predict_log_reg(df, artifact)
    else:
        model, meta = load_catboost()
        predict_fn = lambda df: predict_catboost(df, model, meta)

    if args.interactive or not args.csv:
        df = read_single_sample()
        labels, _ = predict_fn(df)
        print(f"\nPredicted health_condition: {labels[0]}")
        return

    df = pd.read_csv(args.csv)
    labels, ids = predict_fn(df)

    result = pd.DataFrame({"health_condition": labels})
    if ids is not None:
        result.insert(0, "id", ids.values)
    result.to_csv(args.out, index=False)

    print(f"\nPredictions for {len(result)} rows written to: {args.out}")
    print(result["health_condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
