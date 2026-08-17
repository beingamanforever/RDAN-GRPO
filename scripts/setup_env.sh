#!/usr/bin/env bash
# Build the training environment on a fresh CUDA host.
#
# Each step here exists because its absence produced a real failure:
#   python3.12-dev  - Triton compiles a C extension at import, so vLLM will not start without
#                     Python headers, and the error surfaces as an opaque InductorError.
#   flash-attn      - ROLL's fsdp2 strategy imports flash_attn at module scope regardless of
#                     attn_implementation, so the actor worker cannot start without it. The
#                     prebuilt wheel is used because building from source takes ~40 minutes.
#   wandb >= 0.28   - earlier releases reject the current wandb_v1_ API key format outright.
#
# usage: scripts/setup_env.sh [rtt_root] [model_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTT_ROOT="${1:-${RTT_ROOT:-/workspace/Rubrics-To-Tokens}}"
MODEL_DIR="${2:-${RDAN_MODEL_SNAPSHOT:-/workspace/models/Qwen3-4B-Instruct-2507}}"
FLASH_ATTN_VERSION="2.8.3"
cd "$ROOT"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12-dev build-essential

echo "==> python environment"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
# flash-attn publishes no wheel for this torch build and its sdist imports torch at build
# time, which uv's isolated build environment does not have. Everything else installs first;
# the matching prebuilt wheel goes in below.
grep -v '^flash-attn' requirements.txt > "${TMPDIR:-/tmp}/rdan-requirements.txt"
uv pip install -q --python .venv/bin/python -r "${TMPDIR:-/tmp}/rdan-requirements.txt"
uv pip install -q --python .venv/bin/python -e .

echo "==> flash-attn wheel matching this torch build"
read -r TORCH_MM ABI CUDA_MM PYTAG <<<"$(.venv/bin/python - <<'PY'
import sys, torch
print(".".join(torch.__version__.split("+")[0].split(".")[:2]),
      "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE",
      torch.version.cuda.split(".")[0],
      f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"
WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_MM}torch${TORCH_MM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
if ! .venv/bin/python -c "import flash_attn" 2>/dev/null; then
  curl -fsSL -o "/tmp/${WHEEL}" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL}"
  uv pip install -q --python .venv/bin/python "/tmp/${WHEEL}"
fi

echo "==> model"
[ -d "$MODEL_DIR" ] || .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='$MODEL_DIR')
"

echo "==> credentials"
if [ ! -f .env ]; then
  cat > .env <<EOF
OPENROUTER_API_KEY=
WANDB_API_KEY=
RTT_ROOT=$RTT_ROOT
RDAN_MODEL_SNAPSHOT=$MODEL_DIR
EOF
  chmod 600 .env
  echo "    wrote .env, fill in the two API keys"
fi

echo "==> verification"
.venv/bin/python - <<PY
import torch, vllm, ray, wandb, flash_attn, transformers
assert torch.cuda.is_available(), "CUDA is not visible"
from triton.backends.nvidia.driver import CudaUtils; CudaUtils()
assert tuple(int(p) for p in wandb.__version__.split(".")[:2]) >= (0, 28), "wandb is too old for wandb_v1_ keys"
print(f"torch {torch.__version__} | vllm {vllm.__version__} | ray {ray.__version__}")
print(f"flash-attn {flash_attn.__version__} | transformers {transformers.__version__} | wandb {wandb.__version__}")
print(f"gpus {torch.cuda.device_count()} | triton cuda shim ok")
PY
echo "==> ready"
