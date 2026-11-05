#!/usr/bin/env bash
# 2^71 project — fresh-server setup (Ubuntu 24.04 validated)
#
# One command from a clean checkout:
#   bash scripts/setup.sh
#
# What it does:
#   1. Creates an isolated Python virtual environment (.venv) — never
#      touches the system Python — and installs requirements.txt
#      (fixes the observed 'ModuleNotFoundError: No module named ecdsa'
#      on fresh servers).
#   2. Detects the CUDA toolkit (nvcc) and NVIDIA driver. Drivers are
#      detected only — this script never installs or reinstalls them.
#   3. Verifies a supported GPU architecture is compilable (sm_75+;
#      sm_120 for Blackwell).
#   4. Builds the project (make).
#   5. Runs preflight checks: ecdsa import + the CPU reference test suite.
#
# Exit codes: 0 success; non-zero with a clear message on each failure.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { printf '[setup] %s\n' "$*"; }
fail() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. Python environment (isolated, reproducible) ----------------------
if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found. Install: sudo apt-get install -y python3 python3-venv python3-pip"
fi
if [ ! -d .venv ]; then
  log "creating virtual environment .venv"
  python3 -m venv .venv || fail "python3 -m venv failed (install python3-venv)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
log "python: $(python3 --version) ($(command -v python3))"
log "installing requirements.txt"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt || fail "pip install failed (network? see requirements.txt)"

# --- 2. CUDA toolkit + driver detection (no driver installation) ---------
if ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc not found. Install the CUDA Toolkit (>= 12.8; 13.x recommended):
        sudo apt install cuda-toolkit    # or use the NVIDIA runfile / repo"
fi
CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)"
log "CUDA toolkit: ${CUDA_VER:-unknown} ($(command -v nvcc))"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "WARNING: nvidia-smi not found — no NVIDIA driver detected."
  log "         Build will still compile; runtime requires a driver (R570+ for Blackwell)."
else
  log "driver/GPU: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -n1)"
fi

# --- 3. Supported architecture check --------------------------------------
NVCC_ARCHS="$(nvcc --list-gpu-arch 2>/dev/null || true)"
if [ -n "$NVCC_ARCHS" ]; then
  if grep -q "compute_120" <<<"$NVCC_ARCHS"; then
    log "nvcc supports Blackwell (compute_120)"
  elif grep -q "compute_75" <<<"$NVCC_ARCHS"; then
    log "nvcc supports >= compute_75 (Blackwell sm_120 NOT supported by this toolkit)"
  else
    fail "nvcc supports no usable architecture (need compute_75+; found: $NVCC_ARCHS)"
  fi
else
  log "NOTE: nvcc --list-gpu-arch unavailable; skipping arch pre-check"
fi

# --- 4. Build --------------------------------------------------------------
log "building project (make)"
make || fail "build failed"

# --- 5. Preflight ----------------------------------------------------------
log "preflight: ecdsa import"
python3 -c "import ecdsa; print(f'  ecdsa {ecdsa.__version__} OK')" \
  || fail "ecdsa import failed — the dependency fix did not apply"
log "preflight: CPU reference test suite"
python3 -m unittest discover -s tests -v 2>&1 | tail -3
log "preflight complete."
log ""
log "GPU validation (requires hardware):"
log "  python3 -m unittest discover -s tests -v"
log "  python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512"
log "Benchmarks: bash benchmarks/sweep/grid_sweep.sh --repeat 3"
