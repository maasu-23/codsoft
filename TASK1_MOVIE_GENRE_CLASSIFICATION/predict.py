#!/usr/bin/env python3
"""Predict a film's genre from its plot summary.

Prints the top 3 genres with probabilities — the model is genuinely uncertain between
overlapping genres (a biography is also a documentary), so a single label hides most of
what it knows.

Usage:
    python predict.py "A gunslinger rides into a lawless frontier town and faces down
                       the cattle baron who murdered his brother."
    python predict.py --top 5 "A young couple fall in love over one summer in Paris."
    python predict.py --title "Blood Feast" "A caterer murders women for an ancient rite."
    cat plot.txt | python predict.py

Run train.py first to create models/genre_classifier.pkl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "genre_classifier.pkl"
BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plot", nargs="*", help="the plot summary")
    ap.add_argument("--title", default="", help="film title, if you have it (it helps a little)")
    ap.add_argument("--top", type=int, default=3, help="how many genres to show (default 3)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    plot = " ".join(args.plot).strip() or sys.stdin.read().strip()
    if not plot:
        raise SystemExit("No plot summary given. Pass one as an argument or pipe it on stdin.")

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nTrain it first:  python train.py")
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]

    # Must match how train.py builds its input.
    text = f"{args.title} . {plot}" if args.title else plot

    proba = pipeline.predict_proba([text])[0]
    genres = pipeline.classes_
    top = np.argsort(proba)[::-1][:max(1, args.top)]

    if args.json:
        print(json.dumps({
            "title": args.title or None,
            "plot": plot,
            "predictions": [{"genre": str(genres[i]), "probability": round(float(proba[i]), 4)}
                            for i in top],
        }, indent=2))
        return

    tty = sys.stdout.isatty()
    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if tty else s

    if args.title:
        print(f"\n  {c(BOLD, args.title)}")
    print(f'  {c(DIM, plot[:200] + ("..." if len(plot) > 200 else ""))}\n')

    for rank, i in enumerate(top, 1):
        p = float(proba[i])
        bar = "█" * round(p * 28) + "·" * (28 - round(p * 28))
        marker = c(BOLD + CYAN, f"{rank}.") if rank == 1 else f"{rank}."
        print(f"  {marker} {str(genres[i]):<13s} {p:6.1%}  {c(DIM, bar)}")

    if (m := bundle.get("metrics")):
        n_genres = m.get("n_genres", len(bundle.get("genres", pipeline.classes_)))
        print(c(DIM, f"\n  model: macro-F1 {m['macro_f1']:.3f}, accuracy {m['accuracy']:.3f} "
                     f"across {n_genres} genres on {m['n_test']:,} held-out films"))
        print(c(DIM, "  genres overlap heavily — treat the top 3 as the answer, not the top 1"))


if __name__ == "__main__":
    main()
