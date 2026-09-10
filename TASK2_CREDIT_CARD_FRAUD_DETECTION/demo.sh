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

title "TASK 2 — CREDIT CARD FRAUD DETECTION" "RandomForest + engineered features   ·   PR-AUC 0.879"

say "1.85 million transactions. Fraud is 0.39% of them."
say ""
say "So \"never fraud\" scores 99.61% accuracy."
say "Accuracy is not just misleading here — it is absurd."
say ""
say "It is also the one task where ROC-AUC breaks. It ranked the WORST"
say "model above a far better one. PR-AUC is the honest metric." 3

step "1 · Large, online, 2am"
run "--amount 950 --category shopping_net --hour 2 --age 55"

step "2 · A normal weekday grocery run"
run "--amount 42 --category grocery_pos --hour 14 --age 38"

step "3 · The same purchase, scored at every hour of the day"
run "--amount 950 --category shopping_net --age 55 --sweep-hours" 6

say "  Fraud is 18x more likely between 10pm and 4am."
say "  But that signal is not a column in the data — it had to be built"
say "  from the timestamp. And raw \"hour\" correlates with fraud at just"
say "  +0.014, because the risky window wraps around midnight." 3

close "PR-AUC 0.879 · recall 92.4% · threshold chosen by cost, not by F1" \
      "" \
      "Catches 1,981 of 2,145 frauds. Missed fraud costs the transaction;" \
      "a false alarm costs one declined card. Priced out, the model saves" \
      "\$1,109,040 over six months versus approving everything."
