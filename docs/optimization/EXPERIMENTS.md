# Optimization Experiment Log — 2^71 project

Protocol (binding, see `PERFORMANCE_BASELINE.md`): one mechanism per
experiment; every row records commit, environment, resource usage,
repetitions, **median** throughput, variance, correctness, decision.
Never claim a gain from a single noisy measurement. Status values:
`prepared` (implemented, runtime pending), `measured-keep`,
`measured-revert`, `rejected`.

## Log

| ID | Mechanism | Commit | GPU / driver / CUDA | Grid | ptxas resources | Reps | Median Mkeys/s | Variance | Correctness | Decision | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-01 | Build flag A/B: relocatable device code off (`make RDC=0`) | — pending run | — | — | record at run | ≥3 | — | — | — | — | prepared |

## EXP-01 — rdc A/B (prepared)

- **Hypothesis:** `-rdc=true` limits whole-program codegen (audit H4);
  compiling without rdc may improve scheduling and removes device-link
  overhead. Expected: 0–5% (low confidence).
- **Setup:** `make clean && make RDC=0 GPU_ARCH=120 SM_ARCHS=120` (link
  drops `-lcudadevrt`); confirm the binary still passes the 19-test suite
  and `proof.py`; then `bash benchmarks/sweep/grid_sweep.sh --grids "512,64"
  --repeat 5` vs the identical rdc=on sweep.
- **Baseline commit for comparison:** `b3d13be` (rdc on, best observed
  8,133.0 Mkeys/s on RTX 5090, grid 512,64 — early-termination values;
  compare medians only against a fresh rdc=on sweep, not the historical
  best-observed).
- **Risk:** low (build-level; correctness fully re-verified).
- **Decision rule:** keep only if median improves beyond run-to-run
  variance and all correctness gates pass; otherwise revert.

## Candidate queue (from roadmap, gated on E1/E2 profiling)

- O-05 `subp[]` local-memory restructuring — gated on E1 evidence
- O-06 launch-bounds/threadsPerBlock sweep
- E5 `512,128` dip verification (is it real?)
- O-03 host polling zero-copy flag
