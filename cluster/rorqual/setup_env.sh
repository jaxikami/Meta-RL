#!/bin/bash
# One-time setup on a Rorqual LOGIN node: compute nodes have no internet, so
# the virtualenv is built here from the Alliance wheelhouse (--no-index).
#   bash cluster/rorqual/setup_env.sh
set -euo pipefail

VENV="${METARL_VENV:-$HOME/venvs/metarl}"

module load StdEnv/2023 python/3.12
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy numba matplotlib tqdm ruff

python - <<'EOF'
import torch, numpy, numba, matplotlib
print("torch", torch.__version__, "numpy", numpy.__version__,
      "numba", numba.__version__, "matplotlib", matplotlib.__version__)
EOF
echo "virtualenv ready: $VENV"
