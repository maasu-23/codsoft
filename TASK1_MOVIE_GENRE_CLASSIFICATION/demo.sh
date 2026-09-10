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

title "TASK 1 — MOVIE GENRE CLASSIFICATION" "word TF-IDF + LinearSVC   ·   27 genres   ·   macro-F1 0.411"

say "54,000 films, 27 genres — and the data is wildly skewed."
say "Drama has 13,613 films. War has 132. A ratio of 103 to 1."
say ""
say "Guessing \"drama\" every time scores 25% accuracy."
say "So the metric here is macro-F1, which weights all 27 genres"
say "equally — the 132 war films count as much as the dramas." 2.5

step "1 · A plot with a very distinctive vocabulary"
run "\"A gunslinger rides into a lawless frontier town and faces down the cattle baron who murdered his brother.\""

say "  Westerns talk about sheriffs, outlaws and saloons — words that"
say "  appear nowhere else. Best-classified genre of all 27." 2.5

step "2 · A genuinely ambiguous plot"
run "\"A young couple meet in Paris, fall deeply in love over one summer, and must decide whether to stay together.\""

say "  This is why it prints the top 3, not one label."
say "  Romance and drama overlap — and the model knows it." 2.5

step "3 · Horror"
run "--title \"Blood Feast\" \"A caterer murders young women to harvest body parts for an ancient Egyptian rite.\""

close "macro-F1 0.411 · accuracy 0.603 · baseline 0.251" \
      "" \
      "The ceiling is low for a reason worth naming: IMDb films have" \
      "SEVERAL genres, but this dataset keeps only one. When the model" \
      "calls a biography a \"documentary\" it is marked wrong for an" \
      "answer that is also right."
