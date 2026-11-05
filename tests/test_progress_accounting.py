#!/usr/bin/env python3
"""Progress-accounting and traversal-order regression tests (Phase 4).

Root-cause context (docs/research/SEARCH_DISCREPANCY_INVESTIGATION.md):
two grid configurations observed with different Progress values and an
apparently missing match. These tests pin down the actual invariants:

1. On exhaustive completion the reported candidate count equals the
   range length EXACTLY (each key hashed exactly once — no missed,
   duplicated, or miscounted positions). Regression for the count
   reporting: previously the final count was only observable via the
   1 Hz progress line (stale by up to one print interval), which is why
   the "reported count == completed work" invariant was unverifiable.
2. Different --grid configurations traverse candidates in different
   orders (deterministic per configuration) — the same target is found
   by every configuration, at a different keys-hashed count.
3. Progress % and Count use the same counter (consistent definitions).
4. Range-boundary diagnostics: ranges extending to or beyond the
   secp256k1 group order produce a warning (scalars >= n wrap).

GPU tests skip visibly without hardware; the boundary-warning test runs
wherever the binary exists (warning is emitted before device init).
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BINARY = os.environ.get("PROJECT_2_71_BIN", os.path.join(REPO_ROOT, "project-2-71"))

# canonical public test vector (documented in tests/test_gpu_validation.py)
KNOWN_START = 0x2000000000
KNOWN_ADDRESS = "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"
KNOWN_PRIV = 0x22382FACD0

SECP_ORDER_HEX = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"


def gpu_available():
    import shutil
    return shutil.which("nvidia-smi") is not None


def binary_available():
    return os.path.isfile(BINARY) and os.access(BINARY, os.X_OK)


def run_solver(range_arg, address, grid, timeout=300):
    cmd = [BINARY, "--range", range_arg, "--address", address, "--grid", grid]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse(output):
    """Return (priv, keys_hashed, last_count_line, last_progress_pct)."""
    priv = keys = count = prog = None
    for line in output.splitlines():
        if "Private Key" in line and ":" in line:
            priv = line.split(":", 1)[1].strip()
        if "Keys hashed" in line and ":" in line:
            keys = int(line.split(":", 1)[1].strip())
        if "Count:" in line:
            count = int(line.split("Count:")[1].split("|")[0].strip())
        if "Progress:" in line:
            prog = float(line.split("Progress:")[1].replace("%", "").strip())
    return priv, keys, count, prog


class _Bin(unittest.TestCase):
    def setUp(self):
        if not binary_available():
            self.skipTest(f"solver binary not found: {BINARY}")


class TestRangeOrderWarning(_Bin):
    """Boundary diagnostic — runs without a GPU (warning precedes device
    init). Regression for requirement 7 of the investigation."""

    def test_range_beyond_curve_order_warns(self):
        # range ending at the group order n itself (n is even; length
        # n - 2^0... use a power-of-two length ending at n):
        # simplest: end = n, start = n - 2^32 (length 2^32, aligned)
        n = int(SECP_ORDER_HEX, 16)
        start = n - (1 << 32)
        rng = f"{start:X}:{n:X}"
        r = subprocess.run([BINARY, "--range", rng, "--address", KNOWN_ADDRESS],
                           capture_output=True, text=True, timeout=60)
        self.assertIn("Warning: range extends to or beyond the secp256k1 group order",
                      r.stderr, msg=r.stderr)
        # warning is non-fatal: the run proceeds (fails later without GPU,
        # or runs with one) — the warning itself is the regression target.

    def test_normal_range_no_warning(self):
        r = subprocess.run([BINARY, "--range", "2000000000:3FFFFFFFFF",
                            "--address", KNOWN_ADDRESS],
                           capture_output=True, text=True, timeout=60)
        self.assertNotIn("group order", r.stderr, msg=r.stderr)


def _make_case(k):
    import proof
    priv_hex = k.to_bytes(32, "big").hex()
    pub = proof.compressed_pubkey_from_priv32(k.to_bytes(32, "big"))
    addr = proof.p2pkh_from_pubkey_compressed(pub)
    return priv_hex, addr, pub.hex().upper()


class TestProgressAccountingGPU(_Bin):
    """GPU invariants (skips visibly without hardware)."""

    def setUp(self):
        super().setUp()
        if not gpu_available():
            self.skipTest("no NVIDIA GPU detected")
        try:
            import proof  # noqa: F401
        except ImportError:
            self.skipTest("ecdsa package not installed")

    def test_exhaustive_count_equals_range_length(self):
        # 2^30-key range, target OUTSIDE it -> exhaustive completion.
        size = 1 << 30
        start = KNOWN_START
        _, addr_out, _ = _make_case(start + 10 * size)  # far outside
        r = run_solver(f"{start:X}:{start + size - 1:X}", addr_out, "128,4")
        self.assertEqual(r.returncode, 1, msg=r.stdout + r.stderr)
        self.assertIn("KEY NOT FOUND (exhaustive)", r.stdout)
        priv, keys, _, _ = parse(r.stdout)
        self.assertEqual(keys, size,
                         "reported count must equal range length exactly "
                         "(each key hashed exactly once)")

    def test_found_count_within_range_and_correct(self):
        # target INSIDE a 2^30-key range at ~47% position
        size = 1 << 30
        start = KNOWN_START
        k = start + size // 2 + 12345
        priv_hex, addr, _ = _make_case(k)
        r = run_solver(f"{start:X}:{start + size - 1:X}", addr, "128,4")
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        priv, keys, _, _ = parse(r.stdout)
        self.assertEqual(priv.upper(), priv_hex.upper())
        self.assertIsNotNone(keys)
        self.assertGreater(keys, 0)
        self.assertLessEqual(keys, size, "count cannot exceed range length")

    def test_grid_configs_find_same_key_at_different_counts(self):
        """Traversal-order evidence: two grid configurations both find the
        same target with the correct key, but at DIFFERENT keys-hashed
        counts (deterministic per configuration). This is the mechanism
        behind the reported 'different progress, no match yet at the
        observation point' observation."""
        size = 1 << 30
        start = KNOWN_START
        k = start + size // 3
        priv_hex, addr, _ = _make_case(k)
        results = {}
        for grid in ("128,4", "512,4"):
            with self.subTest(grid=grid):
                r = run_solver(f"{start:X}:{start + size - 1:X}", addr, grid)
                self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
                priv, keys, _, _ = parse(r.stdout)
                self.assertEqual(priv.upper(), priv_hex.upper())
                self.assertIsNotNone(keys)
                results[grid] = keys
        self.assertNotEqual(results["128,4"], results["512,4"],
                            "different grid geometry must yield a different "
                            "traversal order (different count at discovery)")
        for grid, keys in results.items():
            self.assertLessEqual(keys, size)

    def test_progress_percent_consistent_with_count(self):
        """Progress % and Count come from the same device counter: the last
        printed pair must agree to the print precision."""
        size = 1 << 32  # long enough for >= 1 progress line
        start = KNOWN_START
        k = start + size - size // 4
        priv_hex, addr, _ = _make_case(k)
        r = run_solver(f"{start:X}:{start + size - 1:X}", addr, "128,4")
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        _, keys, count, prog = parse(r.stdout)
        self.assertIsNotNone(keys)
        if prog is not None and count is not None:
            # same counter, printed together on one line
            self.assertAlmostEqual(prog, count / size * 100.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
