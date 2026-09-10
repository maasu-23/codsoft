#!/usr/bin/env python3
"""Score a credit card transaction for fraud risk.

Usage:
    # a suspicious transaction: large, online, 2am
    python predict.py --amount 950 --category shopping_net --hour 2 --age 55

    # an ordinary one
    python predict.py --amount 42 --category grocery_pos --hour 14 --age 38

    # see how the same purchase scores at every hour of the day
    python predict.py --amount 950 --category shopping_net --sweep-hours

Every flag has a sensible default, so you can vary one at a time. Run train.py first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "fraud_model.pkl"
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
]

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m")


def build_row(args, hour: int) -> dict:
    return {
        "amt": args.amount,
        "category": args.category,
        "hour": hour,
        "is_night": int(hour in NIGHT_HOURS),
        "dayofweek": args.dayofweek,
        "age": args.age,
        "gender": args.gender,
        "city_pop": args.city_pop,
        "distance_km": args.distance,
    }


def band(p: float, threshold: float) -> tuple[str, str]:
    if p >= threshold:
        return ("BLOCK — high fraud risk", RED) if p >= threshold * 2 else ("REVIEW — elevated risk", YELLOW)
    return "APPROVE — looks legitimate", GREEN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amount", type=float, default=100.0, help="transaction amount in $")
    ap.add_argument("--category", choices=CATEGORIES, default="shopping_net")
    ap.add_argument("--hour", type=int, default=14, choices=range(24), metavar="0-23")
    ap.add_argument("--dayofweek", type=int, default=2, choices=range(7), metavar="0-6",
                    help="0=Monday")
    ap.add_argument("--age", type=float, default=45.0, help="cardholder age in years")
    ap.add_argument("--gender", choices=["M", "F"], default="F")
    ap.add_argument("--city-pop", type=int, default=2408,
                    help="cardholder city population (default 2408 = dataset median; "
                         "note 50000 is NOT a value that occurs in this data)")
    ap.add_argument("--distance", type=float, default=75.0,
                    help="km from home to merchant (contributes nothing — see README)")
    ap.add_argument("--sweep-hours", action="store_true",
                    help="score the same transaction at all 24 hours")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nTrain it first:  python train.py")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.5)

    hours = range(24) if args.sweep_hours else [args.hour]
    rows = [build_row(args, h) for h in hours]
    frame = pd.DataFrame(rows)[bundle["features"]]
    probs = pipeline.predict_proba(frame)[:, 1]

    if args.json:
        print(json.dumps({
            "threshold": round(threshold, 4),
            "results": [{"hour": int(r["hour"]), "fraud_probability": round(float(p), 4),
                         "decision": "block" if p >= threshold else "approve"}
                        for r, p in zip(rows, probs)],
        }, indent=2))
        return

    tty = sys.stdout.isatty()
    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if tty else s

    if args.sweep_hours:
        print(f"\n  ${args.amount:,.2f} · {args.category} · age {args.age:.0f}"
              f"  —  scored at every hour\n")
        for h, p in zip(hours, probs):
            bar = "█" * round(p * 34) + "·" * (34 - round(p * 34))
            night = c(DIM, " night") if h in NIGHT_HOURS else "      "
            flag = c(RED, "BLOCK") if p >= threshold else c(GREEN, "ok   ")
            print(f"  {h:02d}:00{night}  {p:6.1%}  {flag}  {c(DIM, bar)}")
        print(c(DIM, f"\n  threshold {threshold:.2f} — the 22:00 jump is the model's clearest signal."))
        print(c(DIM, "  Caveat: it under-rates 00:00-03:59, which the data says is riskier"))
        print(c(DIM, "  than the afternoon. See 'A limitation worth naming' in the README."))
    else:
        label, colour = band(float(probs[0]), threshold)
        print("\n  Transaction")
        for k, v in frame.iloc[0].items():
            print(f"    {k:<13} {v}")
        p = float(probs[0])
        filled = round(p * 30)
        print(f"\n  {c(BOLD + colour, label)}")
        print(f"  fraud probability  {c(BOLD, f'{p:6.2%}')}  "
              f"{c(DIM, '█' * filled + '·' * (30 - filled))}")
        print(c(DIM, f"  decision threshold {threshold:.2f}"))

    if (m := bundle.get("metrics")):
        print(c(DIM, f"\n  model: PR-AUC {m['pr_auc']:.3f}, catches {m['recall']:.0%} of fraud "
                     f"on {m['n_test']:,} held-out transactions ({m['fraud_rate']:.2%} fraud)"))


if __name__ == "__main__":
    main()
