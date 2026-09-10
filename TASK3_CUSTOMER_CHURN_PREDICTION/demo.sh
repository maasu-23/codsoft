#!/usr/bin/env bash
# Self-narrating demo for screen recording — the terminal prints its own
# captions, so a silent recording still explains itself. No voiceover needed.
#
#   ./demo.sh          paced for recording
#   ./demo.sh --fast   no pauses, for checking it works
#
# Run it from inside this task folder. Requires train.py to have been run once.
set -uo pipefail
cd "$(dirname "$0")"

FAST=0; [[ "${1:-}" == "--fast" ]] && FAST=1

if [[ -t 1 ]]; then
  B=$'\e[1m'; D=$'\e[2m'; C=$'\e[36m'; R=$'\e[0m'
else B=""; D=""; C=""; R=""; fi

PY=python3
for cand in ../.venv/bin/python3 .venv/bin/python3; do
  if [[ -x "$cand" ]]; then
    # absolute path: a relative one makes Python emit sys.prefix warnings on screen
    PY="$(cd "$(dirname "$cand")" && pwd)/python3"; break
  fi
done

W=66
pause() { (( FAST )) || sleep "${1:-2}"; }
line()  { printf "  ${D}%s${R}\n" "$(printf '─%.0s' $(seq $W))"; }
thick() { printf "  ${B}%s${R}\n" "$(printf '═%.0s' $(seq $W))"; }
title() { printf "\n"; thick; printf "  ${B}%s${R}\n" "$1"; printf "  ${D}%s${R}\n" "$2"; thick; printf "\n"; pause 3; }
step()  { printf "\n"; line; printf "  ${B}%s${R}\n" "$1"; line; pause 1.5; }
say()   { printf "  %s\n" "$1"; pause "${2:-1.4}"; }
run()   { printf "\n  ${C}\$ python predict.py %s${R}\n" "$1"; pause 1.2; eval "$PY predict.py $1"; pause "${2:-3.5}"; }
close() { printf "\n"; thick; for l in "$@"; do printf "  ${B}%s${R}\n" "$l"; done; thick; printf "\n"; pause 3; }

title "TASK 3 — CUSTOMER CHURN PREDICTION" "GradientBoosting + tuned threshold   ·   ROC-AUC 0.871"

say "10,000 bank customers. 20% of them left."
say "So predicting \"nobody leaves\" is already 79.6% accurate"
say "— and finds not a single customer worth calling."
say ""
say "The bank's real question is not \"how often is the model right?\""
say "It is \"who is about to leave, while there is still time?\"" 2.5

step "1 · A customer the bank should call today"
run "--age 52 --geography Germany --gender Female --credit-score 600 --tenure 2 --balance 120000 --num-products 1 --is-active-member 0"

step "2 · A customer who is going nowhere"
run "--age 28 --geography France --num-products 2 --is-active-member 1"

step "3 · The single strangest finding in the data"
run "--num-products 4"

say "  Customers holding 4 products churn at 100% in this dataset."
say "  1 product: 28%.  2 products: 8%.  3 products: 83%.  4: 100%."
say ""
say "  That V-shape is why the tree models beat logistic regression."
say "  A straight line cannot go down and then back up." 3

close "ROC-AUC 0.871 · recall 0.678 at a threshold of 0.28" \
      "" \
      "The finding that mattered: class_weight=\"balanced\" left ROC-AUC" \
      "UNCHANGED. It never improved the model — it just moved the" \
      "operating point. Tuning the threshold instead reaches the same" \
      "recall with better precision."
