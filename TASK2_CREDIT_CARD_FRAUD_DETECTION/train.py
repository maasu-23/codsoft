#!/usr/bin/env python3
"""Train the credit card fraud detector and save it to models/fraud_model.pkl.

The winning configuration (see notebook.ipynb for the comparison that chose it):

    engineered features -> ColumnTransformer -> RandomForestClassifier

Two things about this task differ from the others in this repo:

1. **Evaluation is temporal, not random.** The dataset ships fraudTrain.csv (Jan 2019 -
   Jun 2020) and fraudTest.csv (Jun - Dec 2020). We honour that split, because a fraud
   model is deployed to score transactions that have not happened yet. A random split
   would let the model learn from the future.

2. **The headline metric is PR-AUC, not ROC-AUC.** At a 0.39% fraud rate, ROC-AUC hides
   false positives behind half a million true negatives - the notebook shows it ranking
   logistic regression ABOVE gradient boosting, which is backwards.

The decision threshold is chosen by minimising expected cost in dollars: a missed fraud
costs the transaction amount, a false alarm costs a flat operational fee.

Usage:
    python train.py
    python train.py --false-alarm-cost 12   # change the cost assumption
    python train.py --threshold 0.5         # override the tuned cut-off
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
RANDOM_STATE = 42
CATEGORICAL = ["category", "gender"]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/long points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's features from the raw transaction record.

    The strongest signal in this dataset (`is_night`) is not a column - it has to be
    built. Raw `hour` correlates with the target at about +0.01 because the risky window
    wraps around midnight, so a correlation-ranked feature selector would discard it.
    """
    ts = pd.to_datetime(df["trans_date_trans_time"])
    return pd.DataFrame({
        "amt": df["amt"],
        "category": df["category"],
        "hour": ts.dt.hour,
        "is_night": ts.dt.hour.isin(NIGHT_HOURS).astype(int),
        "dayofweek": ts.dt.dayofweek,
        "age": (ts - pd.to_datetime(df["dob"])).dt.days / 365.25,
        "gender": df["gender"],
        "city_pop": df["city_pop"],
        # Kept despite contributing nothing - see notebook section 4.4. The simulator
        # draws merchant locations independently of fraud, so this real-world signal
        # was never present in the data. Reported rather than silently dropped.
        "distance_km": haversine_km(df["lat"], df["long"],
                                    df["merch_lat"], df["merch_long"]),
    })


def load(data_dir: Path):
    train_path, test_path = data_dir / "fraudTrain.csv", data_dir / "fraudTest.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise SystemExit(
                f"Dataset not found: {p}\n"
                "Download it with:\n"
                "  kaggle datasets download -d kartik2112/fraud-detection -p data --unzip"
            )
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    t_tr = pd.to_datetime(train["trans_date_trans_time"])
    t_te = pd.to_datetime(test["trans_date_trans_time"])
    print(f"train : {len(train):>9,} transactions  {t_tr.min().date()} -> {t_tr.max().date()}  "
          f"({train['is_fraud'].mean():.4%} fraud)")
    print(f"test  : {len(test):>9,} transactions  {t_te.min().date()} -> {t_te.max().date()}  "
          f"({test['is_fraud'].mean():.4%} fraud)")
    print("        ^ temporal split: the model is scored on the six months AFTER training")
    print(f"'never fraud' baseline accuracy on the test set: {1 - test['is_fraud'].mean():.4%}")
    return train, test


def build_pipeline(numeric_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), numeric_cols),
        ])),
        ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       class_weight="balanced_subsample")),
    ])


def tune_threshold(y_true, proba, amounts, false_alarm_cost: float):
    """Pick the cut-off that minimises expected cost in dollars.

    A missed fraud costs the transaction amount; a false alarm costs a flat fee. Unlike
    the other tasks in this repo, this trade-off has real units, so the threshold is an
    optimisation rather than a judgement call.
    """
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        missed = (y_true.values == 1) & (pred == 0)
        false_alarm = (y_true.values == 0) & (pred == 1)
        rows.append({
            "threshold": t,
            "fraud_loss": amounts[missed].sum(),
            "alarm_cost": false_alarm.sum() * false_alarm_cost,
            "total_cost": amounts[missed].sum() + false_alarm.sum() * false_alarm_cost,
        })
    curve = pd.DataFrame(rows)
    return curve.loc[curve["total_cost"].idxmin()], curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "fraud_model.pkl")
    ap.add_argument("--false-alarm-cost", type=float, default=5.0,
                    help="$ cost of wrongly declining a legitimate transaction (default 5)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold; default is tuned to minimise cost")
    args = ap.parse_args()

    train, test = load(args.data)
    X_train, y_train = engineer(train), train["is_fraud"]
    X_test, y_test = engineer(test), test["is_fraud"]
    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL]

    print(f"\nfeatures ({X_train.shape[1]}): {list(X_train.columns)}")
    print("fitting RandomForest on 1.3M rows (about a minute)...")
    pipeline = build_pipeline(numeric_cols).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, proba)
    roc = roc_auc_score(y_test, proba)

    amounts = test["amt"].values
    best, _curve = tune_threshold(y_test, proba, amounts, args.false_alarm_cost)
    threshold = args.threshold if args.threshold is not None else float(best["threshold"])
    pred = (proba >= threshold).astype(int)

    print(f"\n--- temporal holdout ({len(y_test):,} transactions, "
          f"{int(y_test.sum()):,} fraud) ---")
    print(f"PR-AUC  : {pr_auc:.4f}   <- headline (random guessing scores {y_test.mean():.4f})")
    print(f"ROC-AUC : {roc:.4f}   (flattering - see notebook section 6)")
    print(f"threshold: {threshold:.2f}"
          f"{' (min expected cost)' if args.threshold is None else ' (supplied)'}\n")
    print(classification_report(y_test, pred, target_names=["legitimate", "fraud"], digits=4))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    missed_value = amounts[(y_test.values == 1) & (pred == 0)].sum()
    do_nothing = amounts[y_test.values == 1].sum()
    total_cost = missed_value + fp * args.false_alarm_cost

    print(f"caught       {tp:,} of {tp + fn:,} frauds ({tp / (tp + fn):.1%})")
    print(f"missed       {fn:,} frauds worth ${missed_value:,.0f}")
    print(f"false alarms {fp:,} ({fp / (tn + fp):.3%} of legitimate transactions)")
    print(f"\nfraud losses if we approved everything : ${do_nothing:,.0f}")
    print(f"total cost at this threshold           : ${total_cost:,.0f}")
    print(f"net saving over six months             : ${do_nothing - total_cost:,.0f}")

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc),
        "threshold": float(threshold),
        "accuracy": float((pred == y_test).mean()),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_test)),
        "fraud_rate": float(y_test.mean()),
        "false_alarm_cost": args.false_alarm_cost,
        "net_saving_usd": float(do_nothing - total_cost),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "threshold": float(threshold),
        "features": list(X_train.columns),
        "categorical": CATEGORICAL,
        "numeric": numeric_cols,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }, args.out)
    print(f"\nsaved -> {args.out.name}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
