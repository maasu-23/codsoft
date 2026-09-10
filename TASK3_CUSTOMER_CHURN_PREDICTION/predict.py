#!/usr/bin/env python3
"""Predict the churn probability for a single bank customer.

Usage:
    python predict.py --age 52 --geography Germany --gender Female \
        --credit-score 600 --tenure 2 --balance 120000 --num-products 1 \
        --is-active-member 0 --has-cr-card 1 --estimated-salary 90000

Every flag has a default set to the dataset median/mode, so you can vary one at a time:

    python predict.py --age 25
    python predict.py --age 55 --num-products 4 --is-active-member 0

Run train.py first to create models/churn_model.pkl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "churn_model.pkl"

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m")


def risk_band(p: float, threshold: float) -> tuple[str, str]:
    if p >= threshold:
        return ("HIGH RISK", RED) if p >= threshold + 0.25 else ("AT RISK", YELLOW)
    return "LIKELY TO STAY", GREEN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are the dataset's median (numeric) / mode (categorical).
    ap.add_argument("--credit-score", type=int, default=652)
    ap.add_argument("--geography", choices=["France", "Germany", "Spain"], default="France")
    ap.add_argument("--gender", choices=["Male", "Female"], default="Male")
    ap.add_argument("--age", type=int, default=37)
    ap.add_argument("--tenure", type=int, default=5)
    ap.add_argument("--balance", type=float, default=97198.54)
    ap.add_argument("--num-products", type=int, choices=[1, 2, 3, 4], default=1)
    ap.add_argument("--has-cr-card", type=int, choices=[0, 1], default=1)
    ap.add_argument("--is-active-member", type=int, choices=[0, 1], default=1)
    ap.add_argument("--estimated-salary", type=float, default=100193.92)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the trained decision threshold")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nTrain it first:  python train.py")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.5)

    customer = pd.DataFrame([{
        "CreditScore": args.credit_score,
        "Geography": args.geography,
        "Gender": args.gender,
        "Age": args.age,
        "Tenure": args.tenure,
        "Balance": args.balance,
        "NumOfProducts": args.num_products,
        "HasCrCard": args.has_cr_card,
        "IsActiveMember": args.is_active_member,
        "EstimatedSalary": args.estimated_salary,
    }])[bundle["features"]]

    prob = float(pipeline.predict_proba(customer)[0, 1])
    label, colour = risk_band(prob, threshold)

    if args.json:
        print(json.dumps({
            "churn_probability": round(prob, 4),
            "threshold": round(threshold, 4),
            "prediction": "churn" if prob >= threshold else "stay",
            "risk_band": label,
            "customer": customer.iloc[0].to_dict(),
        }, indent=2, default=str))
        return

    tty = sys.stdout.isatty()
    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if tty else s

    print("\n  Customer")
    for k, v in customer.iloc[0].items():
        print(f"    {k:<18} {v}")

    filled = round(prob * 30)
    print(f"\n  {c(BOLD + colour, label)}")
    print(f"  churn probability  {c(BOLD, f'{prob:6.1%}')}  "
          f"{c(DIM, '█' * filled + '·' * (30 - filled))}")
    print(f"  {c(DIM, f'decision threshold {threshold:.2f} — above this, flag for retention')}")

    if (m := bundle.get("metrics")):
        print(c(DIM, f"\n  model: ROC-AUC {m['roc_auc']:.3f}, catches "
                     f"{m['churn_recall']:.0%} of churners on {m['n_test']} held-out customers"))


if __name__ == "__main__":
    main()
