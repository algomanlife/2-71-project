#!/usr/bin/env python3
"""Regression tests for the benchmark run-log parser (CPU-only).

Fixture logs are SYNTHETIC (clearly labeled) — they test parsing and
failure-detection semantics, never real measurements.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks", "sweep"))

from parse_runs import parse_run_log, summarize, validate_schema  # noqa: E402

import tempfile

GOOD_LOG = """======== 2^71 project =================================
======== PrePhase: GPU Information ====================
Device               : FIXTURE GPU (compute 12.0)
======== Phase-1: BruteForce ==========================
Time: 1.0 s | Speed: 100.0 Mkeys/s | Count: 100 | Progress: 1.00 %
Time: 2.0 s | Speed: 200.0 Mkeys/s | Count: 400 | Progress: 4.00 %
======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
Public Key    : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
"""

NO_SPEED_LOG = """======== 2^71 project =================================
======== Phase-1: BruteForce ==========================
======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
"""

NOT_FOUND_LOG = """======== 2^71 project =================================
======== KEY NOT FOUND (exhaustive) ===================
Target hash160 was not found within the specified range.
"""


class TestParseRunLog(unittest.TestCase):
    def _write(self, text, exit_code=None):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "run_512,64_1.log")
        with open(path, "w") as f:
            f.write(text)
        if exit_code is not None:
            with open(path + ".exit", "w") as f:
                f.write(str(exit_code))
        return path

    def test_valid_run(self):
        r = parse_run_log(self._write(GOOD_LOG, exit_code=0))
        self.assertTrue(r["valid"])
        self.assertEqual(r["best_mkeys_per_s"], 200.0)
        self.assertEqual(r["n_speed_lines"], 2)
        self.assertTrue(r["found_reference_key"])
        self.assertEqual(r["grid"], "512,64")
        self.assertEqual(r["run"], 1)

    def test_missing_speed_is_none_not_zero(self):
        r = parse_run_log(self._write(NO_SPEED_LOG, exit_code=0))
        self.assertIsNone(r["best_mkeys_per_s"])
        self.assertFalse(r["valid"])
        self.assertIn("no speed line", r["failure_reason"])

    def test_nonzero_exit_code_invalid(self):
        # previously '|| true' masked this: solver failed but the run
        # looked like a benchmark
        r = parse_run_log(self._write(GOOD_LOG, exit_code=2))
        self.assertFalse(r["valid"])
        self.assertIn("exit code 2", r["failure_reason"])

    def test_not_found_invalid(self):
        # exhaustive not-found exits 1 (documented exit-code semantics)
        r = parse_run_log(self._write(NOT_FOUND_LOG, exit_code=1))
        self.assertFalse(r["valid"])
        self.assertIn("exit code 1", r["failure_reason"])
        self.assertFalse(r["found_reference_key"])


class TestSummarize(unittest.TestCase):
    def test_median_and_max_separate(self):
        runs = []
        speeds = [100.0, 200.0, 300.0]
        for i, s in enumerate(speeds, 1):
            d = tempfile.mkdtemp()
            path = os.path.join(d, f"run_512,64_{i}.log")
            with open(path, "w") as f:
                f.write(GOOD_LOG.replace("200.0", str(s)))
            with open(path + ".exit", "w") as f:
                f.write("0")
            runs.append(parse_run_log(path))
        s = summarize(runs)["512,64"]
        self.assertEqual(s["median_best_mkeys_per_s"], 200.0)
        self.assertEqual(s["max_best_mkeys_per_s"], 300.0)
        self.assertEqual(s["n_valid_runs"], 3)
        self.assertTrue(s["all_found_reference_key"])

    def test_failed_run_excluded_from_stats(self):
        d = tempfile.mkdtemp()
        ok = os.path.join(d, "run_512,64_1.log")
        with open(ok, "w") as f:
            f.write(GOOD_LOG)
        with open(ok + ".exit", "w") as f:
            f.write("0")
        bad = os.path.join(d, "run_512,64_2.log")
        with open(bad, "w") as f:
            f.write(GOOD_LOG)
        with open(bad + ".exit", "w") as f:
            f.write("2")
        s = summarize([parse_run_log(ok), parse_run_log(bad)])["512,64"]
        self.assertEqual(s["n_valid_runs"], 1)
        self.assertEqual(s["n_runs"], 2)
        self.assertTrue(s["any_run_failed"])
        self.assertEqual(s["median_best_mkeys_per_s"], 200.0)


class TestSchema(unittest.TestCase):
    def test_valid_record(self):
        record = {
            "utc": "x", "commit": "c", "gpu": "g", "cuda_toolkit": "13.4",
            "repeat_per_grid": 3, "termination": "t",
            "grids": {"512,64": {
                "runs": [{"run": 1, "exit_code": 0, "valid": True,
                          "best_mkeys_per_s": 100.0}],
                "n_valid_runs": 1, "median_best_mkeys_per_s": 100.0,
                "max_best_mkeys_per_s": 100.0,
                "all_found_reference_key": True}},
            "notes": "n",
        }
        self.assertEqual(validate_schema(record), [])

    def test_missing_keys_detected(self):
        self.assertTrue(validate_schema({}))
        rec = {"utc": "x", "commit": "c", "gpu": "g", "cuda_toolkit": "1",
               "repeat_per_grid": 1, "termination": "t", "notes": "n",
               "grids": {"g": {"runs": [{"run": 1, "valid": True}]}}}
        errs = validate_schema(rec)
        self.assertTrue(any("exit_code" in e for e in errs))
        self.assertTrue(any("missing key" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
