"""Linearly homomorphic hash over secp256k1 (prime order n): h(v) = sum_k s_k * G_k.

Five coordinates are packed into one scalar at 50-bit lane spacing, s_k = sum_c v[5k+c] * 2^(50c).
Binding: if every lane of an integer vector D satisfies |D_c| < 2^49, its packed scalars are zero
iff D = 0 and lie in (-n/2, n/2), so h(D) = 0 with D != 0 yields a discrete-log relation among
the G_k. Generators are hashed to the curve (try-and-increment on x), so no discrete log is known.
"""

import hashlib

import numpy as np
from coincurve import PublicKey

N_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
LANES = 5
LANE_BITS = 50
LANE_BOUND = 1 << 49
POINT_LEN = 33
IDENTITY = b"\x00" * POINT_LEN
_TAG = b"ppfl/lhh/v1"
_WEIGHTS = np.array([1 << (LANE_BITS * c) for c in range(LANES)], dtype=object)


def _hash_to_point(k: int) -> PublicKey:
    ctr = 0
    while True:
        x = hashlib.sha256(_TAG + k.to_bytes(8, "big") + ctr.to_bytes(4, "big")).digest()
        try:
            return PublicKey(b"\x02" + x)
        except ValueError:
            ctr += 1


_GENS: list[PublicKey] = []


def generators(n_chunks: int) -> list[PublicKey]:
    while len(_GENS) < n_chunks:
        _GENS.append(_hash_to_point(len(_GENS)))
    return _GENS[:n_chunks]


def pack(v: np.ndarray) -> list[int]:
    """Pack an integer vector (|v_i| < 2^49) into scalars mod n."""
    v = np.asarray(v, dtype=np.int64)
    if v.size and int(np.max(np.abs(v))) >= LANE_BOUND:
        raise ValueError("lane overflow")
    pad = (-v.size) % LANES
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.int64)])
    rows = v.reshape(-1, LANES).astype(object)
    return [int(s) % N_ORDER for s in rows.dot(_WEIGHTS)]


def combine(points: list[bytes]) -> bytes:
    keys = [PublicKey(p) for p in points if p != IDENTITY]
    if not keys:
        return IDENTITY
    try:
        return PublicKey.combine_keys(keys).format(compressed=True)
    except ValueError:  # sum is the point at infinity
        return IDENTITY


def _add(a, b):
    """Sum of two points (None = identity); the point at infinity is also returned as None."""
    if a is None:
        return b
    if b is None:
        return a
    try:
        return PublicKey.combine_keys([a, b])
    except ValueError:
        return None


_TWO8 = (256).to_bytes(32, "big")


def _pippenger(scalars: list[int], gens: list[PublicKey]) -> bytes:
    """Bucket MSM with 8-bit windows (Pippenger): per window, generators are summed into 255 digit
    buckets with one libsecp256k1 multi-add each, then combined by running sums; no tables."""
    pts = np.empty(len(gens), dtype=object)
    pts[:] = gens
    raw = np.frombuffer(b"".join(s.to_bytes(32, "little") for s in scalars), dtype=np.uint8).reshape(-1, 32)
    acc = None
    for byte in range(31, -1, -1):
        if acc is not None:
            acc = acc.multiply(_TWO8)
        digits = raw[:, byte]
        if not digits.any():
            continue
        order = np.argsort(digits, kind="stable")
        bounds = np.searchsorted(digits[order], np.arange(1, 257))
        running = window = None
        for d in range(255, 0, -1):
            lo, hi = bounds[d - 1], bounds[d]
            if hi > lo:
                try:
                    bucket = PublicKey.combine_keys(pts[order[lo:hi]].tolist())
                except ValueError:
                    bucket = None
                running = _add(running, bucket)
            window = _add(window, running)
        acc = _add(acc, window)
    return IDENTITY if acc is None else acc.format(compressed=True)


PIPPENGER_MIN_CHUNKS = 1024  # below this the fixed cost of 32 windows x 255 buckets dominates


def hash_vector(v: np.ndarray) -> bytes:
    scalars = pack(v)
    if len(scalars) < PIPPENGER_MIN_CHUNKS:
        return hash_vector_reference(v)
    return _pippenger(scalars, generators(len(scalars)))


def hash_vector_pippenger(v: np.ndarray) -> bytes:
    scalars = pack(v)
    return _pippenger(scalars, generators(len(scalars)))


def hash_vector_reference(v: np.ndarray) -> bytes:
    """One scalar multiplication per packed scalar; used to test the bucket MSM."""
    scalars = pack(v)
    gens = generators(len(scalars))
    terms = [g.multiply(s.to_bytes(32, "big")) for g, s in zip(gens, scalars) if s]
    if not terms:
        return IDENTITY
    try:
        return PublicKey.combine_keys(terms).format(compressed=True)
    except ValueError:
        return IDENTITY
