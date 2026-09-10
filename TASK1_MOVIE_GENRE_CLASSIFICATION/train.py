#!/usr/bin/env python3
"""Train the movie genre classifier and save it to models/genre_classifier.pkl.

The winning configuration (see notebook.ipynb for the comparison that chose it):

    word TF-IDF (1-2 grams) over "title . plot"  ->  LinearSVC(class_weight="balanced")

Evaluation uses the dataset's real held-out test set (test_data_solution.txt), not a
split carved out of the training data.

The saved model wraps LinearSVC in CalibratedClassifierCV so predict.py can rank the
top 3 genres by probability. On 27 classes calibration measurably changes predictions
(it re-anchors to the true class priors, partly undoing class_weight="balanced"), so
both sets of numbers are reported rather than only the flattering one.

Usage:
    python train.py
    python train.py --no-title      # train on the plot description alone
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
RANDOM_STATE = 42


def read_delimited(path: Path, columns: list[str]) -> pd.DataFrame:
    """Parse a ' ::: '-delimited file. The C parser only takes single-char separators."""
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Download it with:\n"
            "  kaggle datasets download -d hijest/genre-classification-dataset-imdb "
            "-p data --unzip"
        )
    df = pd.read_csv(path, sep=" ::: ", engine="python", names=columns, encoding="utf-8")
    for col in df.columns[1:]:
        df[col] = df[col].str.strip()
    return df


def build_text(df: pd.DataFrame, use_title: bool) -> pd.Series:
    """Model input: the plot, optionally with the title (year stripped) prepended."""
    if not use_title:
        return df["description"]
    title = df["title"].str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    return title + " . " + df["description"]


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),      # bigrams catch "world war", "serial killer"
        sublinear_tf=True,
        min_df=2,                # terms in a single film cannot generalise
        stop_words="english",
        strip_accents="unicode",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "models" / "genre_classifier.pkl")
    ap.add_argument("--no-title", action="store_true",
                    help="train on the description alone (the notebook shows title helps)")
    ap.add_argument("--no-refit", action="store_true",
                    help="skip the final refit on train+test and save the evaluated model")
    args = ap.parse_args()

    use_title = not args.no_title
    train = read_delimited(args.data / "train_data.txt", ["id", "title", "genre", "description"])
    solution = read_delimited(args.data / "test_data_solution.txt",
                              ["id", "title", "genre", "description"])

    X_train, y_train = build_text(train, use_title), train["genre"]
    X_test, y_test = build_text(solution, use_title), solution["genre"]

    print(f"train : {len(train):,} films, {y_train.nunique()} genres")
    print(f"test  : {len(solution):,} films (real held-out set, labels from "
          f"test_data_solution.txt)")
    print(f"input : {'title + description' if use_title else 'description only'}")
    counts = y_train.value_counts()
    print(f"balance: {counts.index[0]} {counts.iloc[0]:,} ... {counts.index[-1]} "
          f"{counts.iloc[-1]:,}  (ratio {counts.iloc[0] / counts.iloc[-1]:.0f}:1)")

    print("\nfitting LinearSVC...")
    svc = Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)
    svc_pred = svc.predict(X_test)

    acc = accuracy_score(y_test, svc_pred)
    macro = f1_score(y_test, svc_pred, average="macro")
    print(f"\n--- held-out evaluation ({len(y_test):,} films) ---")
    print(f"macro F1 : {macro:.4f}   <- headline metric (27 genres, 103:1 imbalance)")
    print(f"accuracy : {acc:.4f}   (an 'always drama' baseline scores "
          f"{(y_test == 'drama').mean():.4f})")

    report = pd.DataFrame(classification_report(y_test, svc_pred, output_dict=True,
                                                zero_division=0)).T
    per_genre = report.drop(["accuracy", "macro avg", "weighted avg"]).sort_values("f1-score")
    print("\nworst 5 genres:")
    print(per_genre.head(5)[["precision", "recall", "f1-score", "support"]].round(3).to_string())
    print("\nbest 5 genres:")
    print(per_genre.tail(5)[["precision", "recall", "f1-score", "support"]].round(3).to_string())

    print("\nfitting calibrated model (needed for top-3 probabilities)...")
    calibrated = Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=3, method="sigmoid")),
    ]).fit(X_train, y_train)
    cal_pred = calibrated.predict(X_test)
    cal_acc = accuracy_score(y_test, cal_pred)
    cal_macro = f1_score(y_test, cal_pred, average="macro")

    print(f"\ncalibration trade — it re-anchors to the true class priors, which partly")
    print(f"undoes class_weight='balanced' and drifts back toward the large genres:")
    print(f"  uncalibrated : accuracy {acc:.4f}   macro F1 {macro:.4f}")
    print(f"  calibrated   : accuracy {cal_acc:.4f}   macro F1 {cal_macro:.4f}")

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "calibrated_accuracy": float(cal_acc),
        "calibrated_macro_f1": float(cal_macro),
        "n_test": int(len(y_test)),
        "n_genres": int(y_train.nunique()),
        "used_title": use_title,
    }

    if not args.no_refit:
        print("\nrefitting on train + test for the shipped model "
              "(evaluation above is already complete)...")
        calibrated.fit(pd.concat([X_train, X_test], ignore_index=True),
                       pd.concat([y_train, y_test], ignore_index=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": calibrated,
        "genres": sorted(y_train.unique()),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }, args.out)
    print(f"\nsaved -> {args.out.name}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
