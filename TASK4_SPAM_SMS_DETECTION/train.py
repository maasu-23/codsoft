#!/usr/bin/env python3
"""Train the spam SMS classifier and save it to models/spam_classifier.pkl.

The winning configuration (see notebook.ipynb for the comparison that chose it):

    character-level TF-IDF  ->  LinearSVC(class_weight="balanced")

LinearSVC is wrapped in CalibratedClassifierCV so the saved model can report a real
probability instead of an unbounded distance from the decision boundary. Calibration
fits a sigmoid over the decision values and does not move the boundary itself.

Usage:
    python train.py                 # train, evaluate, save
    python train.py --test-size 0.3 # different holdout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
RANDOM_STATE = 42
LABELS = ["ham", "spam"]          # index == the integer the model predicts


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load spam.csv, repairing the rows the original CSV conversion broke.

    The file is latin-1 encoded and has three overflow columns. Those columns are not
    junk: 50 messages containing a comma inside a broken quote were split across them,
    so we re-join with the comma that split them rather than truncating the message.
    """
    if not csv_path.exists():
        raise SystemExit(
            f"Dataset not found: {csv_path}\n"
            "Download it with:\n"
            "  kaggle datasets download -d uciml/sms-spam-collection-dataset -p data --unzip"
        )

    raw = pd.read_csv(csv_path, encoding="latin-1")
    junk = [c for c in raw.columns if c.startswith("Unnamed")]

    def repair(row) -> str:
        parts = [str(row["v2"])] + [str(row[c]) for c in junk if pd.notna(row[c])]
        return ",".join(parts)

    df = pd.DataFrame({"label": raw["v1"], "message": raw.apply(repair, axis=1)})

    n_repaired = int((df["message"].str.len() > raw["v2"].str.len()).sum())

    # Exact duplicates leak between train and test and inflate the score. Drop and report.
    n_before = len(df)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    print(f"loaded   : {n_before} rows from {csv_path.name}")
    print(f"repaired : {n_repaired} messages truncated by the CSV conversion")
    print(f"deduped  : dropped {n_before - len(df)} exact duplicates -> {len(df)} unique messages")
    print(f"balance  : {(df['label'] == 'spam').mean():.2%} spam")
    return df


def build_pipeline() -> Pipeline:
    """Character n-gram TF-IDF into a calibrated linear SVM."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",      # char n-grams see through FR33 / c-l-a-i-m obfuscation
            ngram_range=(3, 5),
            sublinear_tf=True,       # 1 + log(tf): the 5th "FREE" adds little
            min_df=2,
            lowercase=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=5, method="sigmoid",
        )),
    ])


def evaluate(pipeline: Pipeline, X, y, test_size: float) -> dict:
    """Fit on a stratified split and report honest held-out numbers."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print(f"\n--- held-out evaluation ({len(X_test)} messages, {int(y_test.sum())} spam) ---")
    print(classification_report(y_test, pred, target_names=LABELS, digits=4))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  {fp} real messages wrongly flagged as spam")
    print(f"  {fn} spam messages let through")

    return {
        "accuracy": float((pred == y_test).mean()),
        "spam_precision": float(precision_score(y_test, pred)),
        "spam_recall": float(recall_score(y_test, pred)),
        "spam_f1": float(f1_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_size": test_size,
        "n_test": int(len(X_test)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "spam.csv")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "spam_classifier.pkl")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data)
    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    metrics = evaluate(build_pipeline(), X, y, args.test_size)

    # Refit on 100% of the data: the split existed to produce the estimate above, and
    # the shipped model should not throw away a fifth of its training signal.
    print("\nrefitting on the full dataset...")
    pipeline = build_pipeline().fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "labels": LABELS,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X)),
    }, args.out)

    print(f"saved -> {args.out.relative_to(HERE) if args.out.is_relative_to(HERE) else args.out}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
