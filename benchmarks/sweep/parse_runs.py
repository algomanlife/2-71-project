#!/usr/bin/env python3
"""Benchmark run-log parsing, statistics, and JSON schema validation.

Extracted from benchmarks/sweep/grid_sweep.sh so the parsing logic is
unit-testable (tests/test_benchmark_parsing.py uses synthetic fixture
logs — never real measurements).

Semantics (binding):
- A run is VALID only if: process exit code == 0 (known-answer found),
  AND a speed line was parsed, AND the reference key was found.
- Failed runs are recorded with their exit code and stderr tail, never
  silently treated as valid, and never counted in median/max.
- Missing speed lines are None (never zero).
- Median and maximum are computed and reported as SEPARATE fields.
- The output JSON is validated against the schema after writing.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

SPEED_RE = re.compile(r"Speed:\s*([0-9.]+)\s*Mkeys/s")
PRIV_RE = re.compile(r"Private Key\s*:\s*([0-9A-Fa-f]+)")


def parse_run_log(path):
    """Parse one solver run log -> run dict (exit_code from sidecar file)."""
    name = os.path.basename(path)
    m = re.match(r"run_(.+)_(\d+)\.log", name)
    grid, run_no = (m.group(1), int(m.group(2))) if m else (None, None)
    text = open(path, errors="replace").read()
    speeds = [float(x) for x in SPEED_RE.findall(text)]
    found = bool(PRIV_RE.search(text))
    exit_code = None
    code_file = path + ".exit"
    if os.path.exists(code_file):
        exit_code = int(open(code_file).read().strip() or "-1")
    valid = (exit_code == 0) and bool(speeds) and found
    return {
        "run": run_no,
        "grid": grid,
        "exit_code": exit_code,
        "found_reference_key": found,
        "best_mkeys_per_s": (max(speeds) if speeds else None),
        "n_speed_lines": len(speeds),
        "valid": valid,
        "failure_reason": None if valid else _failure_reason(exit_code, speeds, found),
    }


def _failure_reason(exit_code, speeds, found):
    if exit_code not in (0, None):
        return f"non-zero exit code {exit_code}"
    if not found:
        return "reference key not found"
    if not speeds:
        return "no speed line produced"
    return None


def summarize(runs):
    """Group run dicts by grid -> per-grid statistics."""
    grids = {}
    for r in runs:
        grids.setdefault(r["grid"], []).append(r)
    out = {}
    for g, rs in grids.items():
        valid = [r for r in rs if r["valid"]]
        speeds = [r["best_mkeys_per_s"] for r in valid]
        out[g] = {
            "runs": rs,
            "n_runs": len(rs),
            "n_valid_runs": len(valid),
            "all_found_reference_key": all(r["found_reference_key"] for r in rs),
            "median_best_mkeys_per_s": (statistics.median(speeds) if speeds else None),
            "max_best_mkeys_per_s": (max(speeds) if speeds else None),
            "min_best_mkeys_per_s": (min(speeds) if speeds else None),
            "speed_spread_mkeys_per_s": (
                round(max(speeds) - min(speeds), 1) if speeds else None),
            "any_run_failed": any(not r["valid"] for r in rs),
        }
    return out


def validate_schema(record):
    """Return a list of schema violations (empty list = valid)."""
    errors = []
    for key in ("utc", "commit", "gpu", "cuda_toolkit", "repeat_per_grid",
                "termination", "grids", "notes"):
        if key not in record:
            errors.append(f"missing top-level key: {key}")
    for g, d in record.get("grids", {}).items():
        for key in ("runs", "n_valid_runs", "median_best_mkeys_per_s",
                    "max_best_mkeys_per_s", "all_found_reference_key"):
            if key not in d:
                errors.append(f"grid {g}: missing key {key}")
        for r in d.get("runs", []):
            if "exit_code" not in r or "valid" not in r:
                errors.append(f"grid {g} run {r.get('run')}: missing exit_code/valid")
            if r.get("best_mkeys_per_s") is None and r.get("valid"):
                errors.append(f"grid {g} run {r.get('run')}: valid run with no speed")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--range", required=True)
    ap.add_argument("--address", required=True)
    ap.add_argument("--slices", default="")
    ap.add_argument("--power-log", default="")
    ap.add_argument("--env-info", default="")
    args = ap.parse_args()

    runs = [parse_run_log(p) for p in
            sorted(glob.glob(os.path.join(args.log_dir, "run_*.log")))]
    grids = summarize(runs)

    pwr = []
    if args.power_log and os.path.exists(args.power_log):
        for line in open(args.power_log, errors="replace"):
            parts = line.split()
            if len(parts) >= 4 and parts[0].lstrip("-").isdigit():
                try:
                    pwr.append(float(parts[2]))
                except ValueError:
                    pass

    record = {
        "utc": os.path.basename(args.out).removesuffix(".json"),
        "commit": args.commit,
        "gpu": args.gpu,
        "cuda_toolkit": args.cuda or None,
        "repeat_per_grid": args.repeat,
        "test_vector": {"range": args.range, "address": args.address,
                        "slices": args.slices or None},
        "termination": ("early termination on known-answer match "
                        "(median of per-run best-observed speeds; "
                        "NOT sustained throughput)"),
        "grids": grids,
        "power_w": {"samples": len(pwr),
                    "avg": (round(sum(pwr) / len(pwr), 1) if pwr else None),
                    "max": (max(pwr) if pwr else None)},
        "environment": args.env_info or None,
        "notes": ("missing speed lines are null (never zero); failed runs "
                  "recorded with exit codes and excluded from statistics"),
    }

    errors = validate_schema(record)
    if errors:
        print("SCHEMA ERRORS:", *errors, sep="\n  ", file=sys.stderr)
        sys.exit(2)

    json.dump(record, open(args.out, "w"), indent=2)
    print(f"record written: {args.out}")

    failed = [g for g, d in grids.items() if d["any_run_failed"]]
    if failed:
        print(f"WARNING: failed runs in grids: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
