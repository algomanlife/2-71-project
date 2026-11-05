#!/usr/bin/env bash
# 2^71 project — repeatable grid sweep (Phase 7 benchmark methodology)
#
# For each --grid configuration: run the known-answer search N times,
# parse the best speed line per run, and report MEDIAN and MAX separately.
# Captures environment + dmon power telemetry, RETAINS raw run logs, and
# validates the JSON record schema after writing.
#
# Usage:
#   bash benchmarks/sweep/grid_sweep.sh [--repeat 3]
#                                       [--grids "128,64 512,64 ..."]
#                                       [--range 2000000000:3FFFFFFFFF]
#                                       [--address 1HBtAp...]
#                                       [--allow-failed]
#
# Output: docs/benchmarks/data/sweep-<UTC>.json
#         docs/benchmarks/data/logs-sweep-<UTC>/run_*.log[.exit]
#
# Exit codes: 0 all runs valid; 1 some runs failed (see WARNING lines);
#             2 schema/parse failure. Failed runs are never silently
#             treated as valid benchmarks ('|| true' masking removed).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPEAT=3
GRIDS="128,64 128,128 128,256 256,64 256,128 256,256 512,64 512,128 512,256 1024,64 1024,128 1024,256"
RANGE="2000000000:3FFFFFFFFF"
ADDRESS="1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
SLICES=""
ALLOW_FAILED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)  REPEAT="$2"; shift 2 ;;
    --grids)   GRIDS="$2"; shift 2 ;;
    --range)   RANGE="$2"; shift 2 ;;
    --address) ADDRESS="$2"; shift 2 ;;
    --slices)  SLICES="$2"; shift 2 ;;
    --allow-failed) ALLOW_FAILED=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="docs/benchmarks/data"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$OUT_DIR/logs-sweep-$STAMP"
mkdir -p "$OUT_DIR" "$LOG_DIR"
DMON_LOG="$LOG_DIR/dmon.log"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GPU_NAME="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n1)"
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)"
NCU_VER="$(command -v ncu >/dev/null 2>&1 && ncu --version | sed -n 's/.*Version: \([0-9.]\+\).*/\1/p' | head -n1 || echo none)"
ENV_INFO="make-info: $(make info 2>/dev/null | tr '\n' ' ' | cut -c1-400); ncu: $NCU_VER"

# power telemetry at 1 Hz for the whole sweep (retained with the logs)
nvidia-smi dmon -s pucm -d 1 > "$DMON_LOG" 2>/dev/null &
DMON_PID=$!
trap '[[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true' EXIT

echo "sweep: commit=$COMMIT gpu=[$GPU_NAME] cuda=$NVCC_VER repeat=$REPEAT"
echo "logs:  $LOG_DIR"

SWEEP_RC=0
for G in $GRIDS; do
  echo "--- grid $G"
  for R in $(seq 1 "$REPEAT"); do
    ARGS=(--range "$RANGE" --address "$ADDRESS" --grid "$G")
    [[ -n "$SLICES" ]] && ARGS+=(--slices "$SLICES")
    # Run WITHOUT '|| true': capture the real exit code; a failed run is
    # recorded as invalid (exit code + log retained) and fails the sweep.
    set +e
    ./project-2-71 "${ARGS[@]}" > "$LOG_DIR/run_${G}_${R}.log" 2>&1
    RC=$?
    set -e
    echo "$RC" > "$LOG_DIR/run_${G}_${R}.log.exit"
    if [ "$RC" -ne 0 ]; then
      echo "    run $R FAILED (exit $RC) — see $LOG_DIR/run_${G}_${R}.log" >&2
      SWEEP_RC=1
    fi
  done
done

[[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true

set +e
python3 benchmarks/sweep/parse_runs.py \
  --log-dir "$LOG_DIR" --out "$OUT_DIR/sweep-$STAMP.json" \
  --commit "$COMMIT" --gpu "$GPU_NAME" --cuda "$NVCC_VER" \
  --repeat "$REPEAT" --range "$RANGE" --address "$ADDRESS" \
  --slices "$SLICES" --power-log "$DMON_LOG" --env-info "$ENV_INFO"
PARSE_RC=$?
set -e

if [ "$PARSE_RC" -ne 0 ]; then exit "$PARSE_RC"; fi
if [ "$SWEEP_RC" -ne 0 ] && [ "$ALLOW_FAILED" -ne 1 ]; then
  echo "sweep completed WITH FAILED RUNS (record marked; exit 1)" >&2
  exit 1
fi
exit 0
