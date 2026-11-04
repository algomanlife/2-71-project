#!/usr/bin/env python3
"""GPU correctness validation for 2^71 project.

Rewritten after GPU validation round 1 (RTX 5090 session, see
docs/testing/GPU_VALIDATION_ROUND1_FIXES.md): all search ranges are small,
bounded, and aligned; random keys use fixed seeds; every solver call is
checked for return code, output, and timeout with full diagnostics.

Two test families:

  * TestArgumentValidation — runs WITHOUT a GPU: invalid configurations are
    rejected before any CUDA call (argument validation precedes device
    init in main()). Requires only the compiled binary.
  * Test* (GPU) — require an NVIDIA GPU; skip visibly otherwise. A skip is
    NEVER reported as a pass or as hardware validation.

Run from repository root:
    python3 -m unittest discover -s tests -v
    python3 tests/test_gpu_validation.py --binary ./project-2-71
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ecdsa import SECP256k1, SigningKey
    HAVE_ECDSA = True
except ImportError:
    HAVE_ECDSA = False

BINARY = os.environ.get("PROJECT_2_71_BIN", os.path.join(REPO_ROOT, "project-2-71"))
SOLVER_TIMEOUT = int(os.environ.get("PROJECT_2_71_TIMEOUT", "120"))

# Fixed seeds -> reproducible randomized fixtures.
SEED = 20261105

# README known-answer vector (historical, target documented upstream).
KNOWN_RANGE = "2000000000:3FFFFFFFFF"
KNOWN_ADDRESS = "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
KNOWN_PRIV = "00000000000000000000000000000000000000000000000000000022382FACD0"
KNOWN_PUB = "03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6"

# All fixtures live inside this small region (2^16 keys max) so every GPU
# test finishes in well under a second of search work.
FIXTURE_LO = 0x2000000000
FIXTURE_HI = 0x200000FFFF       # inclusive; 65536 keys total
BATCH = 128                      # default batch (first number of --grid)


def gpu_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def binary_available() -> bool:
    return os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)


class _TimeoutResult:
    def __init__(self, args, e):
        self.args = args
        self.returncode = None
        self.stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        self.stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        self.timed_out = True


def run_solver(range_arg, address, grid="128,8", slices=None, timeout=SOLVER_TIMEOUT):
    """Run the solver; never swallows failures or timeouts."""
    args = [BINARY, "--range", range_arg, "--address", address, "--grid", grid]
    if slices:
        args += ["--slices", str(slices)]
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return _TimeoutResult(args, e)


def parse_found(output: str):
    priv = pub = None
    for line in output.splitlines():
        if "Private Key" in line and ":" in line:
            priv = line.split(":", 1)[1].strip()
        if "Public Key" in line and ":" in line:
            pub = line.split(":", 1)[1].strip()
    return priv, pub


def make_case(priv_int):
    """(priv_hex, address, pubkey_hex) from the independent ecdsa reference."""
    import proof
    priv_hex = priv_int.to_bytes(32, "big").hex()
    pub = proof.compressed_pubkey_from_priv32(priv_int.to_bytes(32, "big"))
    addr = proof.p2pkh_from_pubkey_compressed(pub)
    return priv_hex, addr, pub.hex().upper()


def aligned_range(k, size=4096):
    """Power-of-two, aligned, batch-divisible range of `size` keys containing k."""
    assert size % BATCH == 0
    start = k - (k % size)
    return f"{start:0X}:{start + size - 1:0X}"


def diag(r):
    return (f"cmd={getattr(r, 'args', '?')}\nrc={r.returncode}\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


class _BinaryTest(unittest.TestCase):
    """Needs the compiled binary (no GPU required)."""

    def setUp(self):
        if not binary_available():
            self.skipTest(f"solver binary not found/executable: {BINARY}")


class _GpuTest(_BinaryTest):
    """Needs the compiled binary AND an NVIDIA GPU."""

    def setUp(self):
        super().setUp()
        if not gpu_available():
            self.skipTest("no NVIDIA GPU detected (nvidia-smi absent)")


class TestArgumentValidation(_BinaryTest):
    """Invalid inputs must exit non-zero with a clear stderr message.
    These are rejected before any CUDA call, so they run on GPU-less CI."""

    BAD_CASES = [
        # (range, extra args, expected stderr fragment)
        ("2000000000:2000000FFE", [], "power of two"),            # length 4095
        ("2000000001:2000000800", [], "aligned"),                 # pow2 length, misaligned start
        ("2000000000:2000000FFF", ["--target-hash160", "00" * 20], "not both"),
        ("2000000000:2000000FFF", ["--grid", "130,8"], "power of two"),  # odd batch
        ("2000000000:2000000FFF", ["--grid", "2048,8"], "1024"),  # batch > MAX
    ]

    def test_invalid_configs_rejected(self):
        if not HAVE_ECDSA:
            self.skipTest("ecdsa package not installed (needed for a valid address)")
        addr = make_case(FIXTURE_LO + 1)[1]
        for rng, extra, want in self.BAD_CASES:
            with self.subTest(range=rng, extra=extra):
                cmd = [BINARY, "--range", rng, "--address", addr, "--grid", "128,8"] + extra
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(r.returncode, 0, msg=diag(r))
                self.assertIn(want, r.stderr, msg=diag(r))

    def test_missing_arguments(self):
        r = subprocess.run([BINARY], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stderr)


class TestKnownAnswer(_GpuTest):
    def test_readme_vector(self):
        r = run_solver(KNOWN_RANGE, KNOWN_ADDRESS)
        self.assertEqual(r.returncode, 0, msg=diag(r))
        priv, pub = parse_found(r.stdout)
        self.assertIsNotNone(priv, msg=diag(r))
        self.assertEqual(priv.upper(), KNOWN_PRIV)
        self.assertEqual((pub or "").upper(), KNOWN_PUB)


class TestSmallRanges(_GpuTest):
    """Regression matrix for the round-1 scheduler failure.

    Cases: range smaller than one block (4 threads), one full block
    (256 threads), multiple blocks (512 threads); first/last key; keys at
    batch boundaries; different grids and slices."""

    RANGE_CASES = [
        # (range_size_keys, key_offset, label)
        (512, 0, "first key, sub-block range"),
        (512, 511, "last key, sub-block range"),
        (4096, 0, "first key, sub-block range 4096"),
        (4096, 4095, "last key, sub-block range 4096"),
        (4096, 127, "batch boundary -1"),
        (4096, 128, "batch boundary"),
        (4096, 255, "batch 2 -1"),
        (4096, 3968, "last batch start"),
        (32768, 0, "first key, one full block"),
        (32768, 32767, "last key, one full block"),
        (32768, 16384, "mid key, one full block"),
        (65536, 0, "first key, two blocks"),
        (65536, 65535, "last key, two blocks"),
    ]

    GRID_CASES = ["128,8", "512,8", "128,64"]
    SLICE_CASES = [None, 1, 4, 64]

    def _run_case(self, size, off, grid, slices=None):
        k = FIXTURE_LO + off
        priv_hex, addr, pub_ref = make_case(k)
        rng = f"{FIXTURE_LO:0X}:{FIXTURE_LO + size - 1:0X}"
        # batch must divide range length (512|512, 128|4096, ...)
        if size % int(grid.split(",")[0]) != 0:
            self.skipTest(f"batch {grid} does not divide range {size}")
        r = run_solver(rng, addr, grid=grid, slices=slices)
        self.assertEqual(r.returncode, 0, msg=diag(r))
        priv, pub = parse_found(r.stdout)
        self.assertIsNotNone(priv, msg=diag(r))
        self.assertEqual(priv.upper(), priv_hex.upper(), msg=diag(r))
        self.assertEqual((pub or "").upper(), pub_ref.upper(), msg=diag(r))

    def test_range_matrix_default_grid(self):
        for size, off, label in self.RANGE_CASES:
            with self.subTest(label=label):
                self._run_case(size, off, "128,8")

    def test_grid_configurations(self):
        for grid in self.GRID_CASES:
            for off in (0, 1, 2048, 4095):
                with self.subTest(grid=grid, off=off):
                    self._run_case(4096, off, grid)

    def test_slice_configurations(self):
        for slices in self.SLICE_CASES:
            with self.subTest(slices=slices):
                self._run_case(4096, 2048, "128,8", slices=slices)


class TestParityCoverage(_GpuTest):
    """Fixtures must include both compressed-pubkey parities (02/03)."""

    def test_both_parities_found(self):
        rng = random.Random(SEED)
        keys = [FIXTURE_LO, FIXTURE_LO + 1]
        pubs = [make_case(k)[2] for k in keys]
        while not (any(p.startswith("02") for p in pubs) and any(p.startswith("03") for p in pubs)):
            k = rng.randrange(FIXTURE_LO, FIXTURE_LO + 4096)
            keys.append(k)
            pubs.append(make_case(k)[2])
        self.assertTrue(any(p.startswith("02") for p in pubs))
        self.assertTrue(any(p.startswith("03") for p in pubs))
        for k, pub_ref in zip(keys, pubs):
            with self.subTest(key=hex(k), parity=pub_ref[:2]):
                priv_hex, addr, _ = make_case(k)
                r = run_solver(aligned_range(k, 4096), addr, grid="128,8")
                self.assertEqual(r.returncode, 0, msg=diag(r))
                priv, pub = parse_found(r.stdout)
                self.assertEqual(priv.upper(), priv_hex.upper(), msg=diag(r))
                self.assertEqual((pub or "").upper(), pub_ref.upper(), msg=diag(r))


class TestRandomizedSeeded(_GpuTest):
    """Randomized scalars with reproducible seeds, bounded aligned ranges."""

    def test_seeded_random_keys(self):
        rng = random.Random(SEED)
        keys = [rng.randrange(FIXTURE_LO, FIXTURE_LO + 8192) for _ in range(8)]
        keys.append(rng.randrange(FIXTURE_LO + 8192, FIXTURE_HI - 4096))
        for k in keys:
            with self.subTest(key=hex(k)):
                priv_hex, addr, pub_ref = make_case(k)
                r = run_solver(aligned_range(k, 8192), addr, grid="128,8", slices=4)
                self.assertEqual(r.returncode, 0, msg=diag(r))
                priv, pub = parse_found(r.stdout)
                self.assertEqual(priv.upper(), priv_hex.upper(), msg=diag(r))
                self.assertEqual((pub or "").upper(), pub_ref.upper(), msg=diag(r))


class TestConfigInvariance(_GpuTest):
    """No skipped/duplicated keys under different execution configurations."""

    def test_config_invariance(self):
        rng = random.Random(SEED + 1)
        keys = [FIXTURE_LO, FIXTURE_LO + 1, FIXTURE_LO + 4095,
                rng.randrange(FIXTURE_LO, FIXTURE_LO + 4096)]
        for k in keys:
            priv_hex, addr, _ = make_case(k)
            with self.subTest(key=hex(k)):
                for grid, slices in (("128,8", "4"), ("512,8", None), ("128,8", "64")):
                    with self.subTest(grid=grid, slices=slices):
                        r = run_solver(f"{FIXTURE_LO:0X}:{FIXTURE_LO + 4095:0X}",
                                       addr, grid=grid, slices=slices)
                        self.assertEqual(r.returncode, 0, msg=diag(r))
                        priv, _ = parse_found(r.stdout)
                        self.assertIsNotNone(priv, msg=diag(r))
                        self.assertEqual(priv.upper(), priv_hex.upper(), msg=diag(r))


class TestRepeatability(_GpuTest):
    def test_same_run_twice(self):
        k = FIXTURE_LO + 2048
        priv_hex, addr, pub_ref = make_case(k)
        results = []
        for _ in range(2):
            r = run_solver(f"{FIXTURE_LO:0X}:{FIXTURE_LO + 4095:0X}", addr,
                           grid="128,8", slices=4)
            self.assertEqual(r.returncode, 0, msg=diag(r))
            results.append(parse_found(r.stdout))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0].upper(), priv_hex.upper())
        self.assertEqual(results[0][1].upper(), pub_ref.upper())


class TestExhaustiveNoMatch(_GpuTest):
    def test_no_match_exits_nonzero(self):
        # A key just outside the searched range -> exhaustive completion.
        k_out = FIXTURE_HI + 100
        _, addr, _ = make_case(k_out)
        r = run_solver(f"{FIXTURE_LO:0X}:{FIXTURE_LO + 4095:0X}", addr, grid="128,8")
        self.assertEqual(r.returncode, 1, msg=diag(r))
        self.assertIn("KEY NOT FOUND (exhaustive)", r.stdout, msg=diag(r))


class TestExitCodes(_GpuTest):
    def test_found_exit_zero(self):
        k = FIXTURE_LO + 100
        _, addr, _ = make_case(k)
        r = run_solver(aligned_range(k, 512), addr, grid="128,8")
        self.assertEqual(r.returncode, 0, msg=diag(r))
        self.assertIn("FOUND MATCH", r.stdout, msg=diag(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    ns, remaining = parser.parse_known_args()
    BINARY = ns.binary
    sys.argv = [sys.argv[0]] + remaining
    unittest.main()
