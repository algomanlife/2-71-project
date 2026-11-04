#!/usr/bin/env bash
# 2^71 project — baseline benchmark runner
# Captures the full environment, runs the canonical known-answer search,
# parses throughput, and appends a JSON record.
#
# Usage:  bash benchmarks/baseline/run_baseline.sh [--grid 512,256] [--slices 16]
# Output: docs/benchmarks/data/<UTC-timestamp>.json  (one record per run)
#
# NEVER fabricates numbers: if anything cannot be measured, the field is
# written as null and the record is marked incomplete.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GRID="512,256"
SLICES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid)   GRID="$2"; shift 2 ;;
    --slices) SLICES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="docs/benchmarks/data"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$(mktemp)"
DMON_LOG="$(mktemp)"
trap 'rm -f "$RUN_LOG" "$DMON_LOG"; [[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true' EXIT

# --- environment capture --------------------------------------------------
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)"
CLOCKS_BEFORE="$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader | head -n1)"

# power/clock sampling during run (1 Hz)
nvidia-smi dmon -s pucm -d 1 > "$DMON_LOG" 2>/dev/null &
DMON_PID=$!

# --- canonical run ---------------------------------------------------------
# README known-answer vector: range 2^33 keys, target 1HBtApA... at key
# 0x22382FACD0. The run prints periodic "Time/Speed/Count/Progress" lines.
ARGS=(--range 2000000000:3FFFFFFFFF
      --address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2
      --grid "$GRID")
[[ -n "$SLICES" ]] && ARGS+=(--slices "$SLICES")

T0=$(date +%s.%N)
./project-2-71 "${ARGS[@]}" | tee "$RUN_LOG"
T1=$(date +%s.%N)

[[ -n "${DMON_PID:-}" ]] && kill "$DMON_PID" 2>/dev/null || true

CLOCKS_AFTER="$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader | head -n1)"

# --- parse -----------------------------------------------------------------
python3 - "$RUN_LOG" "$DMON_LOG" "$OUT_DIR/$STAMP.json" "$COMMIT" "$GPU_NAME" \
    "$GPU_CC" "$DRIVER" "$NVCC_VER" "$CLOCKS_BEFORE" "$CLOCKS_AFTER" \
    "$GRID" "${SLICES:-}" "$T0" "$T1" <<'PYEOF'
import json, re, sys

(run_log, dmon_log, out_path, commit, gpu, cc, driver, nvcc,
 before, after, grid, slices, t0, t1) = sys.argv[1:15]

text = open(run_log).read()

speeds = [float(m) for m in re.findall(r"Speed:\s*([0-9.]+)\s*Mkeys/s", text)]
times  = [float(m) for m in re.findall(r"Time:\s*([0-9.]+)\s*s", text)]
counts = [int(m) for m in re.findall(r"Count:\s*(\d+)", text)]

found = "Private Key" in text
priv = None
m = re.search(r"Private Key\s*:\s*([0-9A-Fa-f]+)", text)
if m:
    priv = m.group(1).upper()

peak = max(speeds) if speeds else None
final = speeds[-1] if speeds else None
last_count = counts[-1] if counts else None
wall_s = float(t1) - float(t0)

# power samples from dmon: cols sm clk, mem clk, pwr, temp...
pwr = []
try:
    for line in open(dmon_log):
        parts = line.split()
        if len(parts) >= 4 and parts[0].lstrip("-").isdigit():
            try:
                pwr.append(float(parts[2]))
            except ValueError:
                pass
except FileNotFoundError:
    pass

record = {
    "commit": commit,
    "utc": out_path.split("/")[-1].removesuffix(".json"),
    "gpu": gpu.strip() or None,
    "compute_capability": cc.strip() or None,
    "driver": driver.strip() or None,
    "cuda_toolkit": nvcc or None,
    "grid": grid,
    "slices": slices or None,
    "measured": {
        "wall_seconds": round(wall_s, 2),
        "peak_mkeys_per_s": peak,
        "final_mkeys_per_s": final,
        "total_keys_hashed": last_count,
        "found": found,
        "found_private_key": priv,
        "power_w_samples_avg": (round(sum(pwr) / len(pwr), 1) if pwr else None),
        "power_w_samples_max": (max(pwr) if pwr else None),
    },
    "clocks_before": before,
    "clocks_after": after,
    "incomplete": peak is None or not found,
    "notes": "baseline runner; all values machine-measured, null when not measurable",
}
if priv is not None and priv != "0" * 52 + "22382FACD0".rjust(12, "0"):
    record["notes"] = "UNEXPECTED key found — vector mismatch, investigate"

json.dump(record, open(out_path, "w"), indent=2)
print(f"record written: {out_path}")
if record["incomplete"]:
    print("WARNING: record incomplete (no speed parsed or target not found)", file=sys.stderr)
    sys.exit(1)
PYEOF
