#!/usr/bin/bash
set -euo pipefail

########################################
# Config — edit these as needed
########################################
AI_EDGE_TORCH_VERSION="0.9.4"                 # <-- customize me
MODEL_REPO="google/gemma-4-E2B-it"
VENV_DIR=".venv"
HF_CACHE_ROOT="${PWD}/.hf_custom_cache"       # custom cache location
HF_TOKEN="<HF-TOKEN>"
NUM_THREADS=4
arguments=(
  --output_dir "./out"
 # --output_path "./out"
 # --model_size 1b
 # --prefill_seq_lens 8
 # --prefill_seq_lens 64
 # --prefill_seq_lens 128
 # --prefill_seq_lens 256
 # --prefill_seq_lens 512
 # --prefill_seq_lens 1024
 # --prefill_seq_lens 2048
  --prefill_lengths "[1024]"
 # --kv_cache_max_len 3072
  --cache_length 3072
 # --mask_as_input true
 # --transpose_kv_cache=True
 # --quantize dynamic_int8
  --quantization_recipe dynamic_wi8_afp32
  --externalize_embedder True
  --bundle_litert_lm False
)
########################################

########################################
# Color helpers
########################################
infop() {
    printf '\033[36m'
    printf "$@"
    printf '\033[0m\n'
}

successp() {
    printf '\033[32m'
    printf "$@"
    printf '\033[0m\n'
}

warningp() {
    printf '\033[33m'
    printf "$@"
    printf '\033[0m\n'
}

errorp() {
    printf '\033[31m'
    printf "$@"
    printf '\033[0m\n'
}

runnerp() {
    printf '\033[34m'
    printf "$@"
    printf '\033[0m\n'
}
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
    infop "INFO: Detected existing $d, reusing..."
  else
    mkdir -p "$d"
  fi
done

# Find Python
PY=""
for cand in python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [[ -z "${PY}" ]]; then
  errorp "ERROR: Python not found." >&2
  exit 127
fi

PY_VER="$("$PY" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:2])))
PY
)"
if [[ "${PY_VER}" != "3.11" ]]; then
  warningp "WARNING: Detected Python ${PY_VER}; this script expects Python 3.11.x. Proceeding anyway." >&2
fi

if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/activate" ]]; then
  infop "INFO: Detected existing ${VENV_DIR}, resusing..."
else
  "$PY" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install \
  "transformers>=5.5.0" \
  "accelerate==1.1.0" \
  "litert-torch==${AI_EDGE_TORCH_VERSION}" \
  "ai-edge-tensorflow==2.21.0.dev20260126"

# Fix for Lora Still importing old Tensorflow
LTDIR="$("$PY" -c 'import litert_torch, os; print(os.path.dirname(litert_torch.__file__))')"
sed -i \
  's|from tensorflow.lite.python import schema_py_generated as schema_fb|from ai_edge_litert import schema_py_generated as schema_fb|' \
  "${LTDIR}/generative/layers/lora.py"


# Ensure curl exists
if ! command -v curl >/dev/null 2>&1; then
    errorp "ERROR: 'curl' not found. Please install curl and re-run." >&2
    exit 127
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
arguments+=( --model "$SNAPSHOT_DIR" )

runnerp "Using checkpoint path: ${SNAPSHOT_DIR}"
runnerp "HF cache root: ${HF_CACHE_ROOT}"
runnerp "ai-edge-torch version: ${AI_EDGE_TORCH_VERSION}"
runnerp '========================================================================='
runnerp '%q ' "$PY" -m litert_torch.generative.export_hf "${arguments[@]}"; echo
runnerp '========================================================================='

# Run conversion script
"$PY" -m litert_torch.generative.export_hf "${arguments[@]}"
