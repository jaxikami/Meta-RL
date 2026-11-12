#!/bin/bash
# Meta retrain on Rorqual: one H100 and its share of a GPU node (64 cores and
# 4 H100s per node, so 16 cores per GPU). The 16 cores run the 16 held-out
# tasks in one wave and the training-episode workers.
# Submit from the repository root:  sbatch cluster/rorqual/train_meta.sbatch
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=metarl-meta
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# ~5 h on the workstation. There is no resume, so this leaves headroom.
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
VENV="${METARL_VENV:-$HOME/venvs/metarl}"

module load StdEnv/2023 python/3.12
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg
# The main process steps the networks on the GPU, and each rollout worker
# pins itself to one thread, so few threads are left for the main process.
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# The smoke test (~2 min) stops the job before a broken environment can
# spend the allocation. It runs before the run tag is set.
python tests/smoke_test.py

# Outputs go to policy/<tag>/ and results/{log,plot}/<tag>/. The warm start
# reads saferl from the same policy/<tag>/, so the copies are placed there.
TAG="${PLANT_RUN_TAG:-rorqual_$SLURM_JOB_ID}"
export PLANT_RUN_TAG="$TAG"
mkdir -p "policy/$TAG"
cp -n policy/saferl.pth policy/saferl_best.pth "policy/$TAG/"
# Evaluation redraws the training curves of both arms.
mkdir -p "results/log/$TAG"
cp -n results/log/saferl_log.json "results/log/$TAG/"

python - <<'EOF'
import os, torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| cpus usable:", len(os.sched_getaffinity(0)),
      "| run tag:", os.environ["PLANT_RUN_TAG"])
EOF

python -m metarl train_meta
python -m metarl evaluate
python -m metarl z_informativeness
echo "done: policy/$TAG, results/log/$TAG, results/plot/$TAG"
