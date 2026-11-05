# Usage — `project-2-71` CLI Reference

Verified against `project_2_71.cu` (argument parser and validation at
commit `ad3b5cb`; every option, default, and constraint below was
checked against the source, not inferred).

## Synopsis

```text
project-2-71 --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hex40>)
             [--grid A,B] [--slices N]
```

`--range` and exactly one of `--address` / `--target-hash160` are
required. There is no `--help` flag; running without valid arguments
prints the usage line to stderr and exits non-zero.

## Options

| Option | Format / values | Default | Purpose |
|---|---|---|---|
| `--range` | `START:END`, hexadecimal, `:`-separated | required | scalar search interval |
| `--address` | base58 P2PKH address (starts with `1`) | — | target; decoded to hash160 with a checksum verify |
| `--target-hash160` | exactly 40 hex chars | — | target hash160 directly (equivalent to `--address`) |
| `--grid` | `A,B` — positive integers | `128,8` | `A` = points per batch, `B` = batches per SM |
| `--slices` | integer 1 … 2^20 | `64` | batches processed per thread per kernel launch |

### Validation constraints (all enforced before any GPU work)

| Constraint | Error message (stderr) |
|---|---|
| `A,B` both positive integers ≤ 2^20 | `--grid expects "A,B" (positive integers).` |
| `--slices` in 1..2^20 | `--slices must be in 1..1048576` |
| not both `--address` and `--target-hash160` | `provide either --address or --target-hash160, not both.` |
| range has a `:` | `range format must be start:end` |
| valid hex, ≤ 64 hex chars per bound | `invalid range hex` |
| address decodes (base58 + double-SHA checksum) | `invalid P2PKH address` |
| hash160 is 40 hex chars | `invalid target hash160 hex` |
| batch `A` even power of two | `batch size must be even and a power of two.` |
| batch `A` ≤ 1024 | `batch size must be <= 1024 (kernel limit).` |
| range length (end−start+1) is a power of two | `range length (end - start + 1) must be a power of two.` |
| start aligned to the range length | `start must be aligned to the range length.` |
| range length divisible by batch size | `range length must be divisible by batch size (A).` |

Ranges smaller than one full thread block are supported (partial final
block; see [SEARCH_WORKFLOW.md](SEARCH_WORKFLOW.md)).

## Example (neutral placeholders)

```bash
# structure of a run — substitute your own range and target:
./project-2-71 --range <start_hex>:<end_hex> \
               --address <your_target_p2pkh_address> \
               --grid 512,64 --slices 16
```

A ready-to-run correctness example with a public test vector exists in
the test suite (`tests/test_gpu_validation.py`, `TestKnownAnswer`);
the benchmark methodology uses the same canonical vector
(`docs/benchmarks/GRID_RESULTS.md`). This document deliberately uses
placeholders.

## Output

1. Startup banner: project name, device name/compute capability, SM
   count, threads per block, blocks, points batch size, batches/SM,
   batches/launch, memory utilization, total threads.
2. Progress line (≈1/s, `\r`-updated):
   `Time: … s | Speed: … Mkeys/s | Count: … | Progress: … %`
3. Terminal banner + result:

| Banner | Meaning | Exit code |
|---|---|---|
| `FOUND MATCH!` + private key + compressed public key | target found | `0` |
| `KEY NOT FOUND (exhaustive)` | range fully searched, no match | `1` |
| `INTERRUPTED (Ctrl+C)` | user interrupt; current slice finishes first | `130` |
| `CUDA ERROR` / `TERMINATED` | runtime failure (error name + description on stderr) | `2` |

Invalid arguments exit `1` with `Error: …` on stderr **before any GPU
work**. CUDA API failures are reported with
`CUDA Error (<name>) in <context>: <description>` and exit `2` — a CUDA
failure is never converted into a successful exit.

## Build variants

```bash
make                                   # default (rdc on, auto arch list)
make GPU_ARCH=120 SM_ARCHS="120"       # sm_120 only
make RDC=0                             # whole-program device code (EXP-01)
make NVCC_EXTRA="-DMAX_BATCH_SIZE=512" # compile-time experiments
make ptxas-info                        # register/spill report
```

## Differences from the original CUDACyclone executable

| Aspect | CUDACyclone (original) | `project-2-71` (this project) |
|---|---|---|
| Executable name | `CUDACyclone` | `project-2-71` |
| Exit codes | found `0`; most failures unspecified/single code | `0` FOUND · `1` NOT_FOUND / invalid args · `130` interrupted · `2` CUDA error |
| CUDA error handling | several unchecked calls; stream errors could exit `0` | all CUDA calls checked; errors named + described; never exit `0` on failure |
| Small ranges | aborted ("failed to pick threadsTotal") for ranges with < 1 block of batches | partial final block supported; small ranges execute correctly |
| Startup banner | (none) | `======== 2^71 project =====` header |
| Test tooling CLI | `proof.py --cyclone-path` (default `./CUDACyclone`) | `proof.py --project-path` (default `./project-2-71`) |
| Build | archs `75 86 89` + detected GPU | CUDA-13-aware arch selection incl. `sm_120` + PTX fallback; `RDC` / `NVCC_EXTRA` knobs |
| Search math, CLI semantics of `--range/--grid/--slices/--address` | same | same (preserved by design) |
