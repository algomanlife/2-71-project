#!/usr/bin/env python3
"""Pure-python RIPEMD-160 reference implementation for 2^71 project tests.

Round/rotation/message-order tables were transcribed from the project's own
CUDA implementation (CUDAHash.cu, R11..R51/R12..R52 unrolled sequence), which
is the production-validated implementation. Correctness of this transcription
is verified against the official RIPEMD-160 test vectors in
tests/test_reference.py (and against hashlib when the platform provides
ripemd160).

Used only by the test/benchmark framework. The GPU pipeline never runs this
code.
"""

import struct
from typing import List

# f-functions per round: left line uses f1..f5, right line f5..f1
def _f1(x, y, z): return x ^ y ^ z
def _f2(x, y, z): return (x & y) | (~x & z)
def _f3(x, y, z): return (x | ~y) ^ z
def _f4(x, y, z): return (x & z) | (~z & y)
def _f5(x, y, z): return x ^ (y | ~z)

_F = [_f1, _f2, _f3, _f4, _f5]

_MASK = 0xFFFFFFFF

# K constants
_K_LEFT  = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E]
_K_RIGHT = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000]

# (message_word_index, rotation) per step, 16 steps per round, 5 rounds.
# Left line:
_L_IDX = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8],
    [3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12],
    [1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2],
    [4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13],
]
_L_ROT = [
    [11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8],
    [7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12],
    [11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5],
    [11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12],
    [9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6],
]
# Right line:
_R_IDX = [
    [5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12],
    [6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2],
    [15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13],
    [8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14],
    [12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11],
]
_R_ROT = [
    [8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6],
    [9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11],
    [9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5],
    [15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8],
    [8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11],
]


def _rol(x: int, n: int) -> int:
    x &= _MASK
    return ((x << n) | (x >> (32 - n))) & _MASK


def _compress(h: List[int], block: bytes) -> List[int]:
    w = list(struct.unpack("<16I", block))

    a1, b1, c1, d1, e1 = h
    a2, b2, c2, d2, e2 = h

    for rnd in range(5):
        fl = _F[rnd]
        fr = _F[4 - rnd]
        kl = _K_LEFT[rnd]
        kr = _K_RIGHT[rnd]
        for step in range(16):
            # left line
            x = w[_L_IDX[rnd][step]]
            t = _rol((a1 + fl(b1, c1, d1) + x + kl) & _MASK, _L_ROT[rnd][step]) + e1
            t &= _MASK
            a1, b1, c1, d1, e1 = e1, t, b1, _rol(c1, 10), d1
            # right line
            x = w[_R_IDX[rnd][step]]
            t = _rol((a2 + fr(b2, c2, d2) + x + kr) & _MASK, _R_ROT[rnd][step]) + e2
            t &= _MASK
            a2, b2, c2, d2, e2 = e2, t, b2, _rol(c2, 10), d2

    h0, h1, h2, h3, h4 = h
    return [
        (h1 + c1 + d2) & _MASK,
        (h2 + d1 + e2) & _MASK,
        (h3 + e1 + a2) & _MASK,
        (h4 + a1 + b2) & _MASK,
        (h0 + b1 + c2) & _MASK,
    ]


def ripemd160(data: bytes) -> bytes:
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]

    # padding (MD-style, little-endian bit length)
    padded = data + b"\x80"
    padded += b"\x00" * ((56 - len(padded) % 64) % 64)
    padded += struct.pack("<Q", (len(data) * 8) & 0xFFFFFFFFFFFFFFFF)

    for off in range(0, len(padded), 64):
        h = _compress(h, padded[off:off + 64])

    return struct.pack("<5I", *h)


if __name__ == "__main__":
    import hashlib
    import sys
    vectors = [b"", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz"]
    ok = True
    for v in vectors:
        mine = ripemd160(v).hex()
        print(f"ripemd160({v!r}) = {mine}")
        try:
            ref = hashlib.new("ripemd160", v).hexdigest()
            if mine != ref:
                print(f"  MISMATCH vs hashlib: {ref}")
                ok = False
        except Exception:
            print("  (hashlib ripemd160 unavailable; skipping cross-check)")
    sys.exit(0 if ok else 1)
