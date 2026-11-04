#!/usr/bin/env bash
# 2^71 project — Nsight Compute profiling workflow (experiment E1)
#
# Collects the metrics that classify the main kernel's bottleneck
# (local-memory vs compute vs occupancy) on the canonical known-answer run.
#
# Requirements: CUDA toolkit with Nsight Compute (ncu) on PATH, and
# permission for hardware performance counters (on many systems:
# sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0, or run as
# root). If counters are restricted, ncu fails — that limitation is
# expected; record it and fall back to the repeatable timing sweeps
# (benchmarks/sweep/grid_sweep.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="docs/benchmarks/profile"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

METRICS=(
  # local-memory traffic (H1 / investigation E1 decision gate)
  l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
  l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
  l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
  # occupancy
  sm__warps_active.avg.pct_of_peak_sustained_active
  # stall reasons
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
  # instruction mix / pipes
  sm__inst_executed_pipe_lsu.sum
  sm__inst_executed_pipe_alu.sum
  sm__inst_executed_pipe_fma.sum
  # achieved throughput vs peak
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
)

ARGS=(--range 2000000000:3FFFFFFFFF
      --address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2
      --grid 512,64)

ncu --kernel-name regex:kernel_point_add_and_check_oneinv \
    --metrics "$(IFS=,; echo "${METRICS[*]}")" \
    --launch-count 3 \
    --csv \
    ./project-2-71 "${ARGS[@]}" \
    > "$OUT_DIR/ncu_${STAMP}.csv" 2> "$OUT_DIR/ncu_${STAMP}.stderr"

echo "profile written: $OUT_DIR/ncu_${STAMP}.csv"
echo "interpretation guide: docs/optimization/PERFORMANCE_BASELINE.md (E-series)"
