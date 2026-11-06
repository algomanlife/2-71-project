# Multi-GPU Setup & Test Runbook (2× RTX 5090)

Complete runbook for a fresh 2-GPU server (CUDA 13, driver R570+). Copy
each block in order.

## 0) Pre-flight — verify the hardware you are paying for

```bash
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
```

Expect **two rows**, both `NVIDIA GeForce RTX 5090`, compute `12.0`.
If the cards are mixed (e.g. one 5090 + something else), pin the 5090
with `--device 0` and report the `nvidia-smi` output — heterogeneous
fleets are supported (the range is split across ALL GPUs by default),
but speed is reported in aggregate.

```bash
nvcc --version | tail -2        # CUDA 13.x (13.4 validated)
```

## 1) Install (one command)

```bash
git clone https://github.com/algomanlife/2-71-project.git
cd 2-71-project
bash scripts/setup.sh            # venv + ecdsa oracle + full build
source .venv/bin/activate
git log --oneline -1             # confirm the multi-GPU merge
```

## 2) Correctness gates (must be green before any speed test)

```bash
python3 -m unittest discover -s tests -v          # 33 tests -> OK
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512 | tail -2
# expect: Successes=848 Failures=0  (no-skip key coverage)
```

## 3) Multi-GPU known-answer searches (no flag — all GPUs by default)

```bash
# Short run (~11 s on 2 cards): second historical vector
./project-2-71 --range 8000000000:ffffffffff \
  --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
  --grid 512,64

# Long run (~160 s on 2 cards): third historical vector
./project-2-71 --range 100000000000:1fffffffffff \
  --address 1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk \
  --grid 512,64
```

What to verify in the output:
- banner line `======== 2^71 project (multi-GPU: 2 devices) =======`
- a per-GPU banner for each card (`GPU 0 of 2`, `GPU 1 of 2`)
- ONE aggregated progress line (`Speed:` is the fleet total — expected
  **~17-19 Gkeys/s sustained** on 2× 5090 at `512,64`)
- `FOUND MATCH!` banner with `Found by GPU : k` and the fleet-wide
  `Keys hashed` total
- exit code `0`

## 4) Reproducible speed record (same workflow as single-GPU)

```bash
bash benchmarks/sweep/grid_sweep.sh --repeat 5 --grids "512,64"
grep -E "median_best|max_best" docs/benchmarks/data/sweep-*.json | tail -2
```

## 5) Single-card sanity (optional — classic engine on one card)

```bash
./project-2-71 --range 8000000000:ffffffffff \
  --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
  --grid 512,64 --device 0
```

## Troubleshooting

- `invalid P2PKH address` → paste corruption of the address string
  (verify 34 chars); see TROUBLESHOOTING.md
- Only one GPU used → check `nvidia-smi`; the split requires each
  sub-range to hold at least one full batch (auto-shrinks to a power
  of two); mixed cards are all used by default
- Tests fail → paste the failing block; do not speed-test a build that
  has not passed section 2
