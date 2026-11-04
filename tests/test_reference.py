#!/usr/bin/env python3
"""CPU-only reference tests for 2^71 project validation framework.

These tests run anywhere (no GPU, no CUDA). They validate:
  1. The pure-python RIPEMD-160 reference (official test vectors + hashlib
     cross-check when the platform provides ripemd160).
  2. SHA-256 / hash160 known-answer behavior.
  3. proof.py's address-generation helpers against the independent `ecdsa`
     package (skipped when ecdsa is not installed).
  4. The secp256k1 generator-point constants used across the project.

GPU-side correctness is covered by tests/test_gpu_validation.py (requires
hardware) and proof.py.

Run: python3 -m unittest discover -s tests -v
     (from the repository root)
"""

import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_ripenmd160 import ripemd160  # noqa: E402

# Official RIPEMD-160 test vectors (RIPEMD-160 specification, 1996)
RIPEMD_VECTORS = [
    (b"", "9c1185a5c5e9fc54612808977ee8f548b2258d31"),
    (b"abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"),
    (b"message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36"),
    (b"abcdefghijklmnopqrstuvwxyz", "f71c27109c692c1b56bbdceb5b9d2865b3708dbc"),
    (b"1234567890" * 8, "9b752e45573d4b39f4dbd3323cab82bf63326bfb"),
]

SHA256_VECTORS = [
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
]

# secp256k1 generator point (independent constant, cross-checked below
# against the ecdsa package when available)
G_COMPRESSED_HEX = "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"


class TestRipemd160Reference(unittest.TestCase):
    def test_official_vectors(self):
        for msg, want in RIPEMD_VECTORS:
            with self.subTest(msg=msg[:20]):
                self.assertEqual(ripemd160(msg).hex(), want)

    def test_hashlib_cross_check(self):
        try:
            hashlib.new("ripemd160", b"").digest()
        except Exception:
            self.skipTest("hashlib ripemd160 unavailable on this platform")
        import secrets
        for _ in range(8):
            data = secrets.token_bytes(64)
            self.assertEqual(
                ripemd160(data).hex(),
                hashlib.new("ripemd160", data).hexdigest(),
            )


class TestSha256(unittest.TestCase):
    def test_known_vectors(self):
        for msg, want in SHA256_VECTORS:
            with self.subTest(msg=msg[:20]):
                self.assertEqual(hashlib.sha256(msg).hexdigest(), want)


class TestSecp256k1Constants(unittest.TestCase):
    def test_generator_point(self):
        try:
            from ecdsa import SECP256k1, SigningKey
        except ImportError:
            self.skipTest("ecdsa package not installed")
        sk = SigningKey.from_secret_exponent(1, curve=SECP256k1)
        pub = sk.get_verifying_key().to_string("compressed")
        self.assertEqual(pub.hex().upper(), G_COMPRESSED_HEX)


class TestProofHelpers(unittest.TestCase):
    """Validate proof.py address generation against independent references."""

    def setUp(self):
        try:
            import proof  # noqa: F401
        except ImportError:
            self.skipTest("ecdsa package not installed (proof.py imports it)")
        import proof
        self.proof = proof

    def test_pubkey_matches_ecdsa(self):
        import secrets
        from ecdsa import SECP256k1, SigningKey
        for _ in range(5):
            k = secrets.randbelow(SECP256k1.order - 1) + 1
            want = SigningKey.from_secret_exponent(
                k, curve=SECP256k1
            ).get_verifying_key().to_string("compressed")
            priv32 = k.to_bytes(32, "big")
            got = self.proof.compressed_pubkey_from_priv32(priv32)
            self.assertEqual(got, want)

    def test_p2pkh_hash160(self):
        import secrets
        for _ in range(5):
            pub = secrets.token_bytes(33)
            want = hashlib.new("ripemd160", hashlib.sha256(pub).digest()).digest()
            got = self.proof.hash160(pub)
            self.assertEqual(got, want)

    def test_base58_roundtrip(self):
        import secrets
        for _ in range(5):
            raw = secrets.token_bytes(25)
            enc = self.proof.base58_encode(raw)
            dec = self._base58_decode(enc)
            self.assertEqual(dec, raw)

    @staticmethod
    def _base58_decode(s: str) -> bytes:
        alph = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        num = 0
        for ch in s:
            num = num * 58 + alph.index(ch)
        out = num.to_bytes((num.bit_length() + 7) // 8, "big")
        pad = 0
        for ch in s:
            if ch == "1":
                pad += 1
            else:
                break
        return b"\x00" * pad + out


if __name__ == "__main__":
    unittest.main()
