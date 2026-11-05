#!/usr/bin/env python3
"""Automated experiment runner for the CUDA research lab (Phase 6).

Enumerates the standalone benchmark cases, runs each with repetitions,
computes median/min/max/variability, captures hardware and toolchain
metadata, validates correctness first, and writes JSON + CSV + Markdown
summaries into research/cuda-lab/results/.

Baseline comparison:
    python3 scripts/run_experiments.py --compare results/A.json results/B.json
Improvements are reported only when the median delta exceeds the pooled
run-to-run variability (guard against single-run noise); otherwise the
comparison is labeled inconclusive.

A failed correctness check or benchmark failure aborts the run — failed
processes are never recorded as results.
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
BIN = os.path.join(LAB, "build", "cuda-lab-bench")
TESTS = os.path.join(LAB, "build", "cuda-lab-tests")
RESULTS = os.path.join(LAB, "results")
REPORTS = os.path.join(LAB, "reports")


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def capture_env():
    gpu = sh(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
              "--format=csv,noheader"]).stdout.strip() or "unknown"
    nvcc = ""
    m = sh(["nvcc", "--version"])
    mm = re.search(r"release ([0-9.]+)", m.stdout)
    if mm:
        nvcc = mm.group(1)
    return {
        "gpu": gpu,
        "cuda_toolkit": nvcc or "unknown",
        "commit": sh(["git", "rev-parse", "HEAD"]).stdout.strip() or "unknown",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run_suite(benches, reps, extra_args):
    if not os.path.isfile(TESTS):
        sys.exit("cuda-lab-tests not built; run cmake --build first")
    t = sh([TESTS])
    if t.returncode != 0:
        sys.exit(f"correctness FAILED — no benchmarks run:\n{t.stdout}{t.stderr}")

    if not os.path.isfile(BIN):
        sys.exit("cuda-lab-bench not built")

    records = {}
    for b in benches:
        r = sh([BIN, "--bench", b, "--reps", str(reps), "--json"] + extra_args)
        if r.returncode != 0:
            sys.exit(f"benchmark {b} failed (exit {r.returncode}):\n{r.stderr}")
        for line in r.stdout.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec["bench"]
            # re-run per-rep timings ourselves? the binary reports one JSON
            # object with reps count; collect ms from a --reps run
            records[name] = rec
        print(f"  {b}: median {records[b]['median_ms']} ms"
              if b in records else f"  {b}: no record")

    return records


def verify_benches(benches):
    """Phase-3 validity gate: each bench's executed work is cross-checked
    against the host reference (--verify-only). A false result means the
    benchmark's reported numbers would NOT represent executed workloads —
    abort before any timing."""
    if not os.path.isfile(BIN):
        sys.exit("cuda-lab-bench not built")
    verified = {}
    for b in benches:
        r = sh([BIN, "--bench", b, "--verify-only", "--json"])
        if r.returncode != 0:
            sys.exit(f"verification run failed for {b} (exit {r.returncode}):\n{r.stderr}")
        rec = None
        for line in r.stdout.splitlines():
            try:
                obj = json.loads(line)
                if obj.get("bench") == b:
                    rec = obj
            except json.JSONDecodeError:
                continue
        if rec is None or not rec.get("verified"):
            sys.exit(f"VERIFICATION FAILED for {b}: reported timings would "
                     f"NOT represent executed workloads — aborting")
        verified[b] = True
        print(f"  verify {b}: PASS (matches host reference)")
    return verified


def run_suite_full(benches, reps):
    """Run each bench `reps` times as separate invocations for genuine
    per-repetition timing samples."""
    if not os.path.isfile(TESTS):
        sys.exit("cuda-lab-tests not built")
    t = sh([TESTS])
    if t.returncode != 0:
        sys.exit(f"correctness FAILED — no benchmarks run:\n{t.stdout}{t.stderr}")
    if not os.path.isfile(BIN):
        sys.exit("cuda-lab-bench not built")

    records = {}
    for b in benches:
        samples_ms = []
        ops = None
        op_unit = None
        cfg = {}
        for _ in range(reps):
            r = sh([BIN, "--bench", b, "--reps", "1", "--json"])
            if r.returncode != 0:
                sys.exit(f"benchmark {b} failed (exit {r.returncode}):\n{r.stderr}")
            for line in r.stdout.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                samples_ms.append(rec["ms"])
                ops = rec["ops"]
                op_unit = rec["ops_unit"]
                cfg = {k: rec[k] for k in ("blocks", "threads", "iters")}
        med = statistics.median(samples_ms)
        records[b] = {
            "config": cfg,
            "ops_unit": op_unit,
            "ops": ops,
            "reps": reps,
            "samples_ms": [round(x, 4) for x in samples_ms],
            "median_ms": round(med, 4),
            "min_ms": round(min(samples_ms), 4),
            "max_ms": round(max(samples_ms), 4),
            "variability_ms": round(max(samples_ms) - min(samples_ms), 4),
            "stdev_ms": round(statistics.stdev(samples_ms), 4) if reps > 1 else 0.0,
            "throughput": round(ops / (med / 1000.0), 1),
        }
    return records


def write_outputs(record):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS, f"lab-{stamp}.json")
    csv_path = os.path.join(RESULTS, f"lab-{stamp}.csv")
    md_path = os.path.join(REPORTS, f"lab-{stamp}.md")
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "ops_unit", "median_ms", "min_ms", "max_ms",
                    "stdev_ms", "reps", "throughput"])
        for name, r in record["benches"].items():
            w.writerow([name, r["ops_unit"], r["median_ms"], r["min_ms"],
                        r["max_ms"], r["stdev_ms"], r["reps"], r["throughput"]])

    lines = [f"# CUDA Lab Run {record['utc']}", "",
             f"- GPU: {record['gpu']}  CUDA: {record['cuda_toolkit']}",
             f"- commit: `{record['commit']}`  reps: {record['reps']}", "",
             "| Bench | op unit | median ms | stdev ms | throughput |",
             "|---|---|---:|---:|---:|"]
    for name, r in record["benches"].items():
        lines.append(f"| `{name}` | {r['ops_unit']} | {r['median_ms']} | "
                     f"{r['stdev_ms']} | {r['throughput']:,} ops/s |")
    open(md_path, "w").write("\n".join(lines) + "\n")
    print(f"record: {json_path}\ncsv:    {csv_path}\nreport: {md_path}")


def compare(baseline_path, candidate_path):
    with open(baseline_path) as f:
        base = json.load(f)
    with open(candidate_path) as f:
        cand = json.load(f)
    print(f"baseline: {baseline_path} ({base['utc']})")
    print(f"candidate: {candidate_path} ({cand['utc']})\n")
    print("| Bench | base median | cand median | delta | verdict |")
    print("|---|---:|---:|---:|---|")
    for name, cr in cand["benches"].items():
        br = base["benches"].get(name)
        if not br:
            print(f"| `{name}` | — | {cr['median_ms']} | — | new |")
            continue
        pooled = (br.get("stdev_ms", 0) + cr.get("stdev_ms", 0)) or 1e-9
        delta = (br["median_ms"] - cr["median_ms"])  # negative = faster
        improvement_pct = delta / br["median_ms"] * 100
        # noise guard: improvement must exceed 2x pooled stdev
        if abs(delta) > 2 * pooled:
            verdict = (f"improvement +{improvement_pct:.1f}%" if delta > 0
                       else f"regression {improvement_pct:.1f}%")
        else:
            verdict = "inconclusive (within noise)"
        print(f"| `{name}` | {br['median_ms']} | {cr['median_ms']} | "
              f"{delta:+.3f} ms | {verdict} |")
    print("\nverdicts use the 2x-pooled-stdev guard; smaller medians are faster.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="all")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    listing = sh([BIN, "--list"]).stdout if os.path.isfile(BIN) else ""
    all_benches = re.findall(r"^(\S+)", listing, re.M)
    if not all_benches:
        sys.exit("cuda-lab-bench not built (cmake --build build)")
    benches = all_benches if args.benches == "all" else args.benches.split()

    env = capture_env()
    print(f"lab run: {env['gpu']} cuda={env['cuda_toolkit']} commit={env['commit']}")
    verified = verify_benches(benches)
    records = run_suite_full(benches, args.reps)
    for name in records:
        records[name]["verified"] = verified.get(name, False)
    record = {"utc": env["utc"], "gpu": env["gpu"],
              "cuda_toolkit": env["cuda_toolkit"], "commit": env["commit"],
              "reps": args.reps, "benches": records,
              "notes": "median of repetitions; every bench verified against "
                       "the host reference before timing; failed runs abort; "
                       "single measurements are never reported"}
    write_outputs(record)


if __name__ == "__main__":
    main()
