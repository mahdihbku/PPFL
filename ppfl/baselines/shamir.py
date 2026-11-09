"""Vectorised Shamir secret sharing of 32-byte secrets over GF(2^31 - 1).

A secret is split into 9 limbs of 30 bits; each limb is shared with the same evaluation points,
so a share of one secret is 9 field elements (36 bytes).
"""

import os

import numpy as np

PRIME = (1 << 31) - 1
LIMB_BITS = 30
LIMBS = 9  # 270 bits >= 256
SHARE_LEN = 4 * LIMBS


def to_limbs(secret: bytes) -> np.ndarray:
    n = int.from_bytes(secret, "little")
    return np.array([(n >> (LIMB_BITS * k)) & ((1 << LIMB_BITS) - 1) for k in range(LIMBS)], dtype=np.int64)


def from_limbs(limbs) -> bytes:
    n = sum(int(v) << (LIMB_BITS * k) for k, v in enumerate(limbs))
    return (n % (1 << 256)).to_bytes(32, "little")  # garbage only if fewer than t shares


def _rand_field(shape) -> np.ndarray:
    count = int(np.prod(shape))
    raw = np.frombuffer(os.urandom(8 * count), dtype=np.uint64)
    return (raw % PRIME).astype(np.int64).reshape(shape)


def share(secrets: list[bytes], xs, t: int) -> np.ndarray:
    """Shares of every secret for every holder: array (len(xs), len(secrets) * LIMBS)."""
    xs = np.asarray(xs, dtype=np.int64)
    top = np.concatenate([to_limbs(s) for s in secrets])
    coeffs = _rand_field((t - 1, top.size))
    acc = np.zeros((xs.size, top.size), dtype=np.int64)
    for c in coeffs[::-1]:  # Horner from the highest-degree coefficient
        acc = (acc * xs[:, None] + c) % PRIME
    return (acc * xs[:, None] + top) % PRIME


def lagrange_at_zero(xs) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.int64) % PRIME
    t = xs.size
    num = np.ones(t, dtype=np.int64)
    den = np.ones(t, dtype=np.int64)
    for m in range(t):
        diff = (xs[m] - xs) % PRIME
        diff[m] = 1
        num = num * np.where(np.arange(t) == m, 1, xs[m]) % PRIME
        den = den * diff % PRIME
    inv = np.array([pow(int(d), PRIME - 2, PRIME) for d in den], dtype=np.int64)
    return num * inv % PRIME


def reconstruct(xs, shares: np.ndarray, n_secrets: int) -> list[bytes]:
    """Recover the secrets from exactly t shares (rows aligned with xs)."""
    lam = lagrange_at_zero(xs)
    limbs = (np.asarray(shares, dtype=np.int64) * lam[:, None] % PRIME).sum(axis=0) % PRIME
    return [from_limbs(limbs[k * LIMBS:(k + 1) * LIMBS]) for k in range(n_secrets)]
