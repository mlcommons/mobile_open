#!/usr/bin/bash
set -euo pipefail

########################################
# Config — edit these as needed
########################################
AI_EDGE_TORCH_VERSION="0.6.0"                 # <-- customize me
MODEL_REPO="meta-llama/Llama-3.1-8B-Instruct"
VENV_DIR=".venv"
HF_CACHE_ROOT="${PWD}/.hf_custom_cache"       # custom cache location
HF_TOKEN="<huggingface-token>"
NUM_THREADS=4
USE_LOCAL_SCRIPTS=1 # NOTE If this is set, you need to make sure to have a 'convert_llama3_1b.py' file present.
arguments=(
  --output_path "."
  --model_size 8b
 # --prefill_seq_lens 8
 # --prefill_seq_lens 64
 # --prefill_seq_lens 128
 # --prefill_seq_lens 256
 # --prefill_seq_lens 512
  --prefill_seq_lens 1024
 # --prefill_seq_lens 2048
  --kv_cache_max_len 3072
 # --mask_as_input true
  --quantize dynamic_int8
)
########################################

# Export Hugging Face related variables
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export OPENBLAS_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUM_THREADS}
export HF_TOKEN
export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers" # Deprecated

for d in "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"; do
  if [[ -d "$d" ]]; then
    echo "INFO: Detected existing $d, reusing..."
  else
    mkdir -p "$d"
  fi
done

# Find Python
PY=""
for cand in python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [[ -z "${PY}" ]]; then
  echo "ERROR: Python not found." >&2
  exit 127
fi

PY_VER="$("$PY" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:2])))
PY
)"
if [[ "${PY_VER}" != "3.10" ]]; then
  echo "WARNING: Detected Python ${PY_VER}; this script expects Python 3.10.x. Proceeding anyway." >&2
fi

if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/activate" ]]; then
  echo "INFO: Detected existing ${VENV_DIR}, resusing..."
else
  "$PY" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install \
  "transformers==4.46.3" \
  "accelerate==0.26.0" \
  "ai-edge-torch==${AI_EDGE_TORCH_VERSION}"

# Ensure curl exists
if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: 'curl' not found. Please install curl and re-run." >&2
    exit 127
fi

# Run the download script
"$PY" - <<PY
import torch
from transformers import pipeline
import os

model_id = "$MODEL_REPO"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

res = pipe("The key to life is")
PY

if (( USE_LOCAL_SCRIPTS )); then
    if [[ -f "convert_llama3_1b.py" ]]; then
        echo "Using local ai-edge-torch conversion script."
    else
        echo "ERROR: local ai-edge-torch conversion script could not be found. Please either have 'convert_llama3_1b.py' in your working directory, or disable 'USE_LOCAL_SCRIPTS'."
        exit 2
    fi
else
    CONVERT_URL="https://raw.githubusercontent.com/google-ai-edge/ai-edge-torch/refs/tags/v${AI_EDGE_TORCH_VERSION}/ai_edge_torch/generative/examples/llama/convert_to_tflite.py"
    curl -L -o convert_llama3_1b.py "${CONVERT_URL}"
fi

# Resolve the local snapshot path for the model
SNAPSHOT_DIR="$("$PY" - <<PY
from huggingface_hub import snapshot_download
repo = "$MODEL_REPO"
try:
    p = snapshot_download(repo, local_files_only=True)
except Exception:
    # Fallback: allow a download if not present yet
    p = snapshot_download(repo)
print(p)
PY
)"
arguments+=( --checkpoint_path "$SNAPSHOT_DIR" )

echo "Using checkpoint path: ${SNAPSHOT_DIR}"
echo "HF cache root: ${HF_CACHE_ROOT}"
echo "ai-edge-torch version: ${AI_EDGE_TORCH_VERSION}"
echo '========================================================================='
printf '%q ' "$PY" convert_llama3_1b.py "${arguments[@]}"; echo
echo '========================================================================='

# Run conversion script
"$PY" convert_llama3_1b.py "${arguments[@]}"
