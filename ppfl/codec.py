"""Fixed-point encoding of model vectors into the ring Z_{2^32}."""

import numpy as np

RING_BITS = 32
RING_MOD = 1 << RING_BITS


def max_abs_supported(frac_bits: int) -> float:
    """Largest |aggregate| value representable without wrap-around."""
    return float((1 << (RING_BITS - 1)) - 1) / (1 << frac_bits)


def encode(x: np.ndarray, frac_bits: int) -> np.ndarray:
    """Round x * 2^f to the nearest integer and reduce mod 2^32 (two's complement)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size and np.max(np.abs(x)) > max_abs_supported(frac_bits):
        raise OverflowError("value out of fixed-point range")
    q = np.rint(x * (1 << frac_bits)).astype(np.int64)
    return q.astype(np.uint32)  # wraps negatives mod 2^32


def decode(v: np.ndarray, frac_bits: int) -> np.ndarray:
    """Inverse of encode for a (possibly summed) ring vector; valid while |sum| < 2^31 / 2^f."""
    return np.asarray(v, dtype=np.uint32).view(np.int32).astype(np.float64) / (1 << frac_bits)
