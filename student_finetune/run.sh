tmux kill-session -t autoresearch 2>/dev/null; tmux new -s autoresearch
cd /home/whiskey/workspace/project/central/v2/training/autoresearch/student_finetune
claude --dangerously-skip-permissions "Read student_finetune/program_student.md for your full instructions. You are starting an autonomous experiment run. Read results.tsv for history, then begin the experiment loop. NEVER stop. IMPORTANT: run python train.py SYNCHRONOUSLY, use /home/whiskey/workspace/project/central/v2/training/autoresearch/.venv/bin/python as python path."
