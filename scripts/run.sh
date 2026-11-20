#!/bin/bash
# Consolidated rescore after all training fits completed. This is restored to
# scripts/reproduce.sh after the hardening evidence has been reviewed.
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 - <<'PY'
from pathlib import Path
root=Path('hardening/final_methods')
for group in ('ref','l1','weaker_short','better_long'):
    for seed in range(3):
        for experiment in ('synthetic_chat_specificity','synthetic_faint_chat_concepts'):
            for world in ('dev','heldout'):
                directory=root/f'{group}_seed{seed}'/experiment/world
                assert (directory/'train_info.json').is_file(),directory
                assert (directory/'crosscoder.npz').is_file(),directory
print('All 48 completed fits available; this job only constructs and scores saved outputs.',flush=True)
PY
python3 hardening/final_audit.py
