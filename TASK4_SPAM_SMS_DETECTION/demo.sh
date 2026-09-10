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

title "TASK 4 — SPAM SMS DETECTION" "character-level TF-IDF + LinearSVC   ·   spam F1 0.977"

say "5,572 real SMS messages. Only 13% are spam."
say "So a model that answers \"not spam\" every single time"
say "is 87% accurate — and catches nothing."
say ""
say "That is why this project is scored on F1 for the spam class,"
say "not on accuracy." 2.5

step "1 · A scam message the model has never seen"
run "\"Congratulations! You've won a free iPhone, click here to claim\""

step "2 · An ordinary text from a friend"
run "\"hey are you coming to class tomorrow\""

step "3 · A harder one — real spam, no obvious keywords"
run "\"Your free ringtone is waiting to be collected. Simply text the password to 85069\""

close "On 1,034 held-out messages:" \
      "  spam F1 0.977   ·   recall 0.954   ·   ZERO false positives" \
      "" \
      "Character n-grams beat word n-grams here, because spammers" \
      "write FR33 and c-l-a-i-m to dodge keyword filters."
