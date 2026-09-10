#!/usr/bin/env python3
"""Classify an SMS message as spam or ham using the trained model.

Usage:
    python predict.py "Congratulations! You've won a free iPhone, click here to claim"
    python predict.py "hey are you coming to class tomorrow" "URGENT! call 09061701461 now"
    echo "your message here" | python predict.py
    python predict.py --json "free entry to win"        # machine-readable output

Run train.py first to create models/spam_classifier.pkl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "spam_classifier.pkl"

BOLD, DIM, RED, GREEN, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[0m"


def load_model(path: Path):
    if not path.exists():
        raise SystemExit(f"Model not found: {path}\nTrain it first:  python train.py")
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle.get("labels", ["ham", "spam"]), bundle


def confidence_bar(p: float, width: int = 24) -> str:
    filled = round(p * width)
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="*", help="one or more messages to classify")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="spam probability above which to flag (default 0.5; "
                         "lower it to catch more spam at the cost of false alarms)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    messages = args.message or [line.strip() for line in sys.stdin if line.strip()]
    if not messages:
        raise SystemExit("No message given. Pass one as an argument or pipe it on stdin.")

    pipeline, labels, bundle = load_model(args.model)
    probs = pipeline.predict_proba(messages)

    results = []
    for msg, p in zip(messages, probs):
        spam_p = float(p[1])
        results.append({
            "message": msg,
            "label": "spam" if spam_p >= args.threshold else "ham",
            "spam_probability": round(spam_p, 4),
            "confidence": round(max(spam_p, 1 - spam_p), 4),
        })

    if args.json:
        print(json.dumps(results, indent=2))
        return

    color_out = sys.stdout.isatty()
    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if color_out else s

    for r in results:
        is_spam = r["label"] == "spam"
        tag = c(BOLD + (RED if is_spam else GREEN), f"{r['label'].upper():>4}")
        print(f'\n  "{r["message"][:100]}{"..." if len(r["message"]) > 100 else ""}"')
        print(f"  {tag}   spam probability {r['spam_probability']:6.1%}  "
              f"{c(DIM, confidence_bar(r['spam_probability']))}")

    if (m := bundle.get("metrics")):
        print(c(DIM, f"\n  model: spam F1 {m['spam_f1']:.3f} / recall {m['spam_recall']:.3f} "
                     f"on {m['n_test']} held-out messages"))


if __name__ == "__main__":
    main()
