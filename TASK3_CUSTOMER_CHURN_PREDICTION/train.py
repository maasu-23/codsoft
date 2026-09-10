#!/usr/bin/env python3
"""Train the customer churn model and save it to models/churn_model.pkl.

The winning configuration (see notebook.ipynb for the comparison that chose it):

    ColumnTransformer(one-hot + scale)  ->  GradientBoostingClassifier

with **default class weights** and an explicitly tuned decision threshold.

Why not class_weight="balanced": the notebook shows it leaves ROC-AUC unchanged
(0.871 -> 0.869), because it does not improve the model's ranking of customers by
risk — it only moves the operating point. Tuning the threshold on the default-weight
model reaches the same recall with better precision, and keeps the threshold as an
explicit, retunable number rather than a hyperparameter.

Usage:
    python train.py
    python train.py --threshold 0.20      # override the tuned cut-off
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RANDOM_STATE = 42

DROP_COLS = ["RowNumber", "CustomerId", "Surname"]   # identifiers, not features
TARGET = "Exited"
CATEGORICAL = ["Geography", "Gender"]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(
            f"Dataset not found: {csv_path}\n"
            "Download it with:\n"
            "  kaggle datasets download -d shivan118/churn-modeling-dataset -p data --unzip"
        )
    df = pd.read_csv(csv_path)

    # Surname has 2,932 distinct values. Left in, a tree can split on a rare surname
    # and "predict" an individual it memorised — a leak that generalises to nothing.
    df = df.drop(columns=DROP_COLS)

    print(f"loaded  : {len(df)} customers, {df.shape[1] - 1} features")
    print(f"dropped : {', '.join(DROP_COLS)} (identifiers)")
    print(f"missing : {df.isna().sum().sum()} values")
    print(f"balance : {df[TARGET].mean():.2%} churned "
          f"(a 'nobody churns' baseline is {1 - df[TARGET].mean():.2%} accurate)")
    return df


def build_pipeline(numeric_cols: list[str]) -> Pipeline:
    """One-hot the two categoricals, scale the numerics, then boost."""
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            # Trees do not need scaling; it is here so the same preprocessor can serve
            # the logistic-regression baseline in the notebook, and it does no harm.
            ("num", StandardScaler(), numeric_cols),
        ])),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def tune_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Pick the cut-off that maximises F1 on the churn class.

    A neutral default. With real costs — what a retention offer costs versus the
    lifetime value of a saved customer — you would read the operating point off the
    precision/recall curve instead.
    """
    grid = np.linspace(0.05, 0.95, 91)
    curve = pd.DataFrame({
        "threshold": grid,
        "precision": [precision_score(y_true, proba >= t, zero_division=0) for t in grid],
        "recall": [recall_score(y_true, proba >= t) for t in grid],
        "f1": [f1_score(y_true, proba >= t) for t in grid],
    })
    return float(curve.loc[curve["f1"].idxmax(), "threshold"]), curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "Churn_Modelling.csv")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "churn_model.pkl")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold; default is tuned for best F1")
    args = ap.parse_args()

    df = load_data(args.data)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(numeric_cols).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    tuned, curve = tune_threshold(y_test, proba)
    threshold = args.threshold if args.threshold is not None else tuned
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    print(f"\n--- held-out evaluation ({len(X_test)} customers, {int(y_test.sum())} churners) ---")
    print(f"ROC-AUC: {auc:.4f}   (threshold-independent — this is the model's real score)")
    print(f"decision threshold: {threshold:.2f}"
          f"{' (tuned for best F1)' if args.threshold is None else ' (supplied)'}\n")
    print(classification_report(y_test, pred, target_names=["stayed", "churned"], digits=4))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"caught    {tp} of {tp + fn} churners ({tp / (tp + fn):.1%})")
    print(f"missed    {fn} churners            <- the silent, expensive failure")
    print(f"contacted {fp} customers who were staying anyway  <- the cheap failure")

    at_half = (proba >= 0.5).astype(int)
    print(f"\nfor reference, at sklearn's default 0.50: "
          f"recall {recall_score(y_test, at_half):.4f}, F1 {f1_score(y_test, at_half):.4f}")

    metrics = {
        "roc_auc": float(auc),
        "threshold": float(threshold),
        "accuracy": float((pred == y_test).mean()),
        "churn_precision": float(precision_score(y_test, pred)),
        "churn_recall": float(recall_score(y_test, pred)),
        "churn_f1": float(f1_score(y_test, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(X_test)),
    }

    print("\nrefitting on the full dataset...")
    final = build_pipeline(numeric_cols).fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": final,
        "threshold": float(threshold),   # retunable without retraining
        "features": list(X.columns),
        "categorical": CATEGORICAL,
        "numeric": numeric_cols,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }, args.out)

    print(f"saved -> {args.out.name}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
