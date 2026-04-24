#!/usr/bin/env bash
# Student v2 autonomous experiment loop via Hermes Agent.
#
# Launches a tmux session that keeps hermes chat running (auto-restart on exit
# so one max-turns limit does not terminate the whole experiment run).

set -u

SESSION="autoresearch_student_v2"
WORKDIR="/home/whiskey/workspace/project/central/v2/training/autoresearch/student_finetune"
VENV_PY="/home/whiskey/workspace/project/central/v2/training/autoresearch/.venv/bin/python"

PROMPT="Read student_finetune/program_final_student_v2.md for your full instructions. You are starting an autonomous V2 experiment run. Read results_v2.tsv for V2 history and results.tsv for V1 reference, then begin the experiment loop. NEVER stop. IMPORTANT: edit ONLY train_v2.py and results_v2.tsv -- NEVER touch train.py, train_final.py, results.tsv, prepare.py, or workspace/output/distill_final_lcnet050/ (those are V1). Run python train_v2.py SYNCHRONOUSLY, use ${VENV_PY} as python path."

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new -d -s "${SESSION}" "cd ${WORKDIR} && while true; do hermes chat --yolo --accept-hooks --max-turns 9999 -q \"${PROMPT}\"; echo '[run_v2.sh] hermes exited, restarting in 10s'; sleep 10; done"

echo "Started tmux session '${SESSION}'. Attach with:"
echo "  tmux attach -t ${SESSION}"
