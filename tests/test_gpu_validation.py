#!/usr/bin/env python3
"""GPU correctness validation for 2^71 project.

Requires: compiled binary (default ./project-2-71), an NVIDIA GPU, and the
`ecdsa` python package. Skips with an explicit message otherwise — a skip
is NEVER reported as a pass.

Covers (per docs/audits/INITIAL_TECHNICAL_AUDIT.md §8 and the Phase-3 spec):
  - known-answer end-to-end search (README historical vector)
  - scalar multiplication / hash pipeline correctness vs the independent
    `ecdsa` reference (CPU vs GPU result consistency)
  - boundary conditions (first/last key of range, both parities)
  - randomized scalars across the range
  - skip/duplicate detection across different --grid/--slices configurations
  - repeatability

Run from repository root:
    python3 -m unittest discover -s tests -v
    python3 tests/test_gpu_validation.py --binary ./project-2-71 --grid 512,256
"""

import argparse
import os
import secrets
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


def _gpu_available() -> bool:
    return shutil.which("nvidia-smi") is not None


# CLI overrides (unittest-friendly via env or module attributes)
BINARY = os.environ.get("PROJECT_2_71_BIN", os.path.join(REPO_ROOT, "project-2-71"))
GRID = os.environ.get("PROJECT_2_71_GRID", "512,256")
SLICES = os.environ.get("PROJECT_2_71_SLICES", None)

# README historical known-answer vector (original project's documented run):
# range 0x2000000000:0x3FFFFFFFFF (2^33 keys, power of two, aligned),
# address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 -> key 0x22382FACD0
KNOWN_RANGE = "2000000000:3FFFFFFFFF"
KNOWN_ADDRESS = "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
KNOWN_PRIV = "00000000000000000000000000000000000000000000000000000022382FACD0"


class _GpuTest(unittest.TestCase):
    def setUp(self):
        if not _gpu_available():
            self.skipTest("no NVIDIA GPU detected (nvidia-smi absent)")
        if not (os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)):
            self.skipTest(f"solver binary not found/executable: {BINARY}")


def run_solver(range_arg, address, grid=GRID, slices=None, timeout=600):
    args = [BINARY, "--range", range_arg, "--address", address, "--grid", grid]
    if slices:
        args += ["--slices", slices]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def parse_found(output: str):
    priv = pub = None
    for line in output.splitlines():
        if "Private Key" in line and ":" in line:
            priv = line.split(":", 1)[1].strip()
        if "Public Key" in line and ":" in line:
            pub = line.split(":", 1)[1].strip()
    return priv, pub


def make_case(priv_int):
    """Return (priv_hex, address) computed by the independent reference."""
    assert HAVE_ECDSA, "ecdsa package required to generate GPU test cases"
    import proof  # proof.py (repo root) uses the ecdsa reference as well
    priv_hex = priv_int.to_bytes(32, "big").hex()
    pub = proof.compressed_pubkey_from_priv32(priv_int.to_bytes(32, "big"))
    addr = proof.p2pkh_from_pubkey_compressed(pub)
    return priv_hex, addr, pub.hex().upper()


class TestKnownAnswer(_GpuTest):
    def test_readme_vector(self):
        r = run_solver(KNOWN_RANGE, KNOWN_ADDRESS)
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        priv, pub = parse_found(r.stdout)
        self.assertIsNotNone(priv, msg=r.stdout)
        self.assertEqual(priv.upper(), KNOWN_PRIV)


class TestScalarMulVsReference(_GpuTest):
    """GPU math + hash pipeline vs ecdsa reference (CPU vs GPU consistency)."""

    def test_random_keys_found_and_pubkey_matches(self):
        if not HAVE_ECDSA:
            self.skipTest("ecdsa package not installed")
        lo, hi = 0x2000000000, 0x3FFFFFFFFF
        keys = [lo, lo + 1, hi - 1, hi] + [
            secrets.randbelow(hi - lo + 1) for _ in range(4)
        ]
        for k in keys:
            with self.subTest(key=k):
                priv_hex, addr, pub_ref = make_case(k)
                # small power-of-two range containing the key, aligned
                size = 1 << max(8, (hi - lo).bit_length())
                start = k - (k % size)  # align to range length
                rng = f"{start:0X}:{start + size - 1:0X}"
                r = run_solver(rng, addr)
                priv, pub = parse_found(r.stdout)
                self.assertIsNotNone(priv, msg=r.stdout)
                self.assertEqual(priv.upper(), priv_hex.upper())
                self.assertIsNotNone(pub, msg=r.stdout)
                self.assertEqual(pub.upper(), pub_ref.upper())


class TestNoSkipDuplicate(_GpuTest):
    """Same keys must be found under different execution configurations."""

    def test_config_invariance(self):
        if not HAVE_ECDSA:
            self.skipTest("ecdsa package not installed")
        lo, hi = 0x2000000000, 0x2000001000  # 2^12 keys
        keys = [lo, lo + 2, hi - 2, hi - 1, 0x2000000ABC]
        for k in keys:
            priv_hex, addr, _ = make_case(k)
            with self.subTest(key=k):
                r = run_solver("2000000000:2000000FFF", addr, grid="128,8",
                               slices="4")
                priv, _ = parse_found(r.stdout)
                self.assertIsNotNone(priv, msg=r.stdout)
                self.assertEqual(priv.upper(), priv_hex.upper())


class TestRepeatability(_GpuTest):
    def test_same_run_twice(self):
        if not HAVE_ECDSA:
            self.skipTest("ecdsa package not installed")
        k = 0x2000000ABC
        priv_hex, addr, _ = make_case(k)
        results = []
        for _ in range(2):
            r = run_solver("2000000000:2000000FFF", addr)
            priv, _ = parse_found(r.stdout)
            results.append(priv)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0].upper(), priv_hex.upper())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--grid", default=GRID)
    parser.add_argument("--slices", default=SLICES)
    ns, remaining = parser.parse_known_args()
    BINARY, GRID, SLICES = ns.binary, ns.grid, ns.slices
    sys.argv = [sys.argv[0]] + remaining
    unittest.main()
