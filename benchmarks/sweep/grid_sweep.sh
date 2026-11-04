#!/usr/bin/env bash
# 2^71 project — repeatable grid sweep (Phase 7 benchmark methodology)
#
# For each --grid configuration: run the known-answer search N times,
# parse the best speed line per run, and report the MEDIAN with variance.
# Captures environment + dmon power telemetry. Writes a JSON record.
#
# Usage:
#   bash benchmarks/sweep/grid_sweep.sh [--repeat 3]
#                                       [--grids "128,64 512,64 ..."]
#                                       [--range 2000000000:3FFFFFFFFF]
#                                       [--address 1HBtAp...]
#
# Output: docs/benchmarks/data/sweep-<UTC>.json
#
# Notes:
# - Runs are early-termination known-answer tests. The JSON marks this;
#   do not report medians as sustained throughput.
# - Missing speed lines (like the observed 256,128 case) are recorded as
#   null — never as zero.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPEAT=3
GRIDS="128,64 128,128 128,256 256,64 256,128 256,256 512,64 512,128 512,256 1024,64 1024,128 1024,256"
RANGE="2000000000:3FFFFFFFFF"
ADDRESS="1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
SLICES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)  REPEAT="$2"; shift 2 ;;
    --grids)   GRIDS="$2"; shift 2 ;;
    --range)   RANGE="$2"; shift 2 ;;
    --address) ADDRESS="$2"; shift 2 ;;
    --slices)  SLICES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="docs/benchmarks/data"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"; [[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true' EXIT

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GPU_NAME="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n1)"
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)"

# power telemetry at 1 Hz for the whole sweep
nvidia-smi dmon -s pucm -d 1 > "$TMPDIR_RUN/dmon.log" 2>/dev/null &
DMON_PID=$!

echo "sweep: commit=$COMMIT gpu=[$GPU_NAME] cuda=$NVCC_VER repeat=$REPEAT"

for G in $GRIDS; do
  echo "--- grid $G"
  for R in $(seq 1 "$REPEAT"); do
    ARGS=(--range "$RANGE" --address "$ADDRESS" --grid "$G")
    [[ -n "$SLICES" ]] && ARGS+=(--slices "$SLICES")
    ./project-2-71 "${ARGS[@]}" > "$TMPDIR_RUN/run_${G}_${R}.log" 2>&1 || true
  done
done

[[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true

python3 - "$TMPDIR_RUN" "$OUT_DIR/$STAMP.json" "$COMMIT" "$GPU_NAME" \
    "$NVCC_VER" "$REPEAT" "$RANGE" "$ADDRESS" "$SLICES" <<'PYEOF'
import json, re, sys, glob, os, statistics

tmpdir, out_path, commit, gpu, nvcc, repeat, rng, addr, slices = sys.argv[1:10]

speeds = {}
for f in sorted(glob.glob(os.path.join(tmpdir, 'run_*.log'))):
    m = re.match(r'run_(.+)_(\d+)\.log', os.path.basename(f))
    grid, rep = m.group(1), int(m.group(2))
    text = open(f).read()
    vals = [float(x) for x in re.findall(r'Speed:\s*([0-9.]+)\s*Mkeys/s', text)]
    found = 'Private Key' in text
    speeds.setdefault(grid, []).append({
        'run': rep, 'best_mkeys_per_s': (max(vals) if vals else None),
        'found_reference_key': found,
    })

pwr = []
try:
    for line in open(os.path.join(tmpdir, 'dmon.log')):
        parts = line.split()
        if len(parts) >= 4 and parts[0].lstrip('-').isdigit():
            try: pwr.append(float(parts[2]))
            except ValueError: pass
except FileNotFoundError:
    pass

record = {
    'utc': os.path.basename(out_path).removesuffix('.json'),
    'commit': commit,
    'gpu': gpu, 'cuda_toolkit': nvcc or None,
    'repeat_per_grid': int(repeat),
    'test_vector': {'range': rng, 'address': addr, 'slices': slices or None},
    'termination': 'early termination on known-answer match (NOT sustained throughput)',
    'grids': {
        g: {
            'runs': v,
            'median_best_mkeys_per_s': (
                statistics.median([r['best_mkeys_per_s'] for r in v
                                   if r['best_mkeys_per_s'] is not None])
                if any(r['best_mkeys_per_s'] is not None for r in v) else None),
            'n_valid_runs': sum(1 for r in v if r['best_mkeys_per_s'] is not None),
            'all_found_reference_key': all(r['found_reference_key'] for r in v),
        } for g, v in speeds.items()
    },
    'power_w': {'samples': len(pwr),
                'avg': (round(sum(pwr)/len(pwr), 1) if pwr else None),
                'max': (max(pwr) if pwr else None)},
    'notes': 'median of per-run best-observed speeds; null when no speed line (never zero)',
}
json.dump(record, open(out_path, 'w'), indent=2)
print(f"record written: {out_path}")
bad = [g for g, d in record['grids'].items() if not d['all_found_reference_key']]
if bad:
    print(f"WARNING: grids missing reference key: {bad}", file=sys.stderr)
    sys.exit(1)
PYEOF
