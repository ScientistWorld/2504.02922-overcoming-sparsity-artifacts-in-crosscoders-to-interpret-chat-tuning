#!/bin/bash
# Hardening job: build the public dev worlds and the hidden held-out worlds,
# train the reference (BatchTopK) and baseline (L1) crosscoders, freeze them
# into eval/, write the train/test scores, then run the hardening audit
# (seeds, better/weaker variants, baselines, shortcuts, entry-point attacks).

set -e

cd /home/user

stamp() { echo "== $(date '+%F %T') $*"; }

stamp env
nvidia-smi -L || true
python3 -c "import os, torch, numpy; print('cpus', len(os.sched_getaffinity(0)), 'cuda', torch.cuda.is_available(), 'torch', torch.__version__, 'numpy', numpy.__version__)"

stamp data
bash scripts/download.sh
python3 -m eval.test.build_heldout

stamp reference
WORKERS=4 bash scripts/method.sh
stamp baseline
WORKERS=4 bash scripts/baseline.sh

stamp freeze
for m in batchtopk l1; do
  for exp in synthetic_chat_specificity synthetic_faint_chat_concepts; do
    for pair in dev:eval/train heldout:eval/test; do
      world="${pair%%:*}"
      dst="${pair#*:}/reference_methods/$m/$exp/$world"
      if [ ! -f "$dst/crosscoder.npz" ]; then
        mkdir -p "$dst"
        cp "checkpoints/methods/$m/$exp/$world/crosscoder.npz" "checkpoints/methods/$m/$exp/$world/train_info.json" "$dst/"
      fi
    done
  done
done

stamp scores
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

stamp audit
python3 hardening/audit.py

stamp validate
python3 validate.py || true
stamp done
