#!/usr/bin/env bash
# 2^71 project — Nsight Compute profiling workflow (experiment E1)
#
# Collects the metrics that classify the main kernel's bottleneck
# (local-memory vs compute vs occupancy) on the canonical known-answer run.
#
# Common failure mode (observed on the RTX 5090 rig):
#   ERR_NVGPUCTRPERM — the NVIDIA driver does not grant GPU performance
#   counter access to this user. ncu writes an output file but collects
#   NO valid metrics. This script detects that, reports the limitation,
#   preserves the diagnostic log, marks the output invalid (.failed), and
#   exits non-zero — invalid profiler output is never published as data.
#
# Remedies for ERR_NVGPUCTRPERM (choose per your security policy):
#   sudo modprobe -r nvidia && sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
#   (persist via /etc/modprobe.d/nvidia.conf), or run ncu via sudo, or
#   grant the profiling user CAP_SYS_ADMIN.
#
# Usage:
#   bash benchmarks/profile/ncu_profile.sh              # full ncu profile
#   bash benchmarks/profile/ncu_profile.sh --timing     # timing-only fallback
#
# Outputs (docs/benchmarks/profile/):
#   env_<stamp>.txt          environment report (always)
#   ptxas_sm120_<stamp>.txt  compiler resource report for sm_120 (always)
#   ncu_<stamp>.csv          valid metrics (only when profiling succeeded)
#   ncu_<stamp>.failed.csv   invalid output (renamed, never published)
#   ncu_<stamp>.stderr       preserved diagnostics
#   timing_<stamp>.txt       timing-fallback results

set -uo pipefail   # NOT -e: errors are handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="ncu"
[ "${1:-}" = "--timing" ] && MODE="timing"

OUT_DIR="docs/benchmarks/profile"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

RANGE="2000000000:3FFFFFFFFF"
ADDRESS="1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
GRID="512,64"

# --- environment report (always) ------------------------------------------
{
  echo "=== profiling environment ==="
  echo "utc        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit     : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "gpu        : $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n1)"
  echo "cuda       : $(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)"
  echo "ncu        : $(ncu --version 2>/dev/null | sed -n 's/.*Version: \([0-9.]\+\).*/\1/p' | head -n1)"
  echo "mode       : $MODE"
} | tee "$OUT_DIR/env_${STAMP}.txt"

# --- compiler resource usage for sm_120 (always available, no counters) ---
{
  echo "=== ptxas resource report (sm_120) ==="
  nvcc -O3 -rdc=true --ptxas-options=-O3 -std=c++17 \
       -gencode arch=compute_120,code=sm_120 -Xptxas -v \
       -c project_2_71.cu -o /dev/null 2>&1 \
       | grep -E "Compiling entry|stack frame|spill|registers" || true
  echo "(resource observations are HYPOTHESES for investigation until"
  echo " counter-based profiling works — not measured bottlenecks)"
} > "$OUT_DIR/ptxas_sm120_${STAMP}.txt" 2>&1
echo "ptxas report: $OUT_DIR/ptxas_sm120_${STAMP}.txt"

if [ "$MODE" = "timing" ]; then
  # --- timing-only fallback (no hardware counters required) ---------------
  echo "mode: timing-only fallback (no ncu counters)"
  {
    for G in 512,64 512,128 128,64; do
      echo "--- grid $G (3 repetitions, best-observed, early termination)"
      for R in 1 2 3; do
        ./project-2-71 --range "$RANGE" --address "$ADDRESS" --grid "$G" 2>&1 \
          | grep -E "Speed:|FOUND" | tail -2
      done
    done
  } | tee "$OUT_DIR/timing_${STAMP}.txt"
  echo "timing results: $OUT_DIR/timing_${STAMP}.txt"
  echo "NOTE: timing-only results cannot classify bottlenecks; use them for"
  echo "      A/B comparisons only (see docs/optimization/EXPERIMENTS.md)."
  exit 0
fi

# --- full ncu profile ------------------------------------------------------
if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not installed (Nsight Compute required for profiling)" >&2
  exit 2
fi

METRICS=(
  l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
  l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
  sm__inst_executed_pipe_lsu.sum
  sm__inst_executed_pipe_alu.sum
  sm__inst_executed_pipe_fma.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
)
METRIC_CSV="$(IFS=,; echo "${METRICS[*]}")"

echo "running ncu (this may fail if GPU counters are restricted)..."
ncu --kernel-name regex:kernel_point_add_and_check_oneinv \
    --metrics "$METRIC_CSV" \
    --launch-count 3 \
    --csv \
    ./project-2-71 --range "$RANGE" --address "$ADDRESS" --grid "$GRID" \
    > "$OUT_DIR/ncu_${STAMP}.csv" 2> "$OUT_DIR/ncu_${STAMP}.stderr"
NCU_RC=$?

DIAG="$OUT_DIR/ncu_${STAMP}.stderr"
if grep -qi "ERR_NVGPUCTRPERM" "$DIAG" 2>/dev/null || [ "$NCU_RC" -ne 0 ]; then
  mv "$OUT_DIR/ncu_${STAMP}.csv" "$OUT_DIR/ncu_${STAMP}.failed.csv" 2>/dev/null
  echo "" >&2
  echo "PROFILING FAILED (ncu exit $NCU_RC)." >&2
  if grep -qi "ERR_NVGPUCTRPERM" "$DIAG" 2>/dev/null; then
    echo "  Cause: ERR_NVGPUCTRPERM — this user lacks NVIDIA GPU performance" >&2
    echo "  counter permission. The output contained NO valid metrics; it has" >&2
    echo "  been renamed .failed.csv and will never be published as measured" >&2
    echo "  data. Remedies (pick one per your security policy):" >&2
    echo "    sudo modprobe -r nvidia && sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0" >&2
    echo "    (persist via /etc/modprobe.d/nvidia.conf), or run ncu with sudo." >&2
  fi
  echo "  Diagnostics preserved: $DIAG" >&2
  echo "  Fallback (no counters): bash benchmarks/profile/ncu_profile.sh --timing" >&2
  exit 1
fi

# sanity: a valid profile must contain metric rows for the kernel
if ! grep -q "kernel_point_add" "$OUT_DIR/ncu_${STAMP}.csv"; then
  mv "$OUT_DIR/ncu_${STAMP}.csv" "$OUT_DIR/ncu_${STAMP}.failed.csv"
  echo "PROFILING OUTPUT INVALID (no kernel metric rows) — renamed .failed.csv" >&2
  echo "  Diagnostics preserved: $DIAG" >&2
  exit 1
fi

echo "profile written: $OUT_DIR/ncu_${STAMP}.csv"
echo "interpretation guide: docs/optimization/PERFORMANCE_BASELINE.md (E-series)"
