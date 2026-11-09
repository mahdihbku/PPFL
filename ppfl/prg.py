"""Key derivation (HKDF-SHA256) and mask expansion (AES-128-CTR) into Z_{2^32} vectors."""

import struct

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_ZEROS = bytearray()


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def mask_key(shared: bytes, salt: bytes, t: int, attempt: int, i: int, j: int) -> bytes:
    """Per-round key for the sub-mask between client i and committee member j."""
    info = b"ppfl/mask/v1" + struct.pack(">IBII", t, attempt, i, j)
    return hkdf(shared, salt, info)


def commit_mask_key(seed: bytes, t: int, attempt: int, j: int) -> bytes:
    info = b"ppfl/commit-mask/v1" + struct.pack(">IBI", t, attempt, j)
    return hkdf(seed, b"", info)


def expand(key: bytes, length: int) -> np.ndarray:
    """Expand a 32-byte key (16 B AES key || 16 B IV) into `length` uniform uint32 values."""
    global _ZEROS
    nbytes = 4 * length
    if len(_ZEROS) < nbytes:
        _ZEROS = bytearray(nbytes)
    enc = Cipher(algorithms.AES(key[:16]), modes.CTR(key[16:32])).encryptor()
    out = enc.update(memoryview(_ZEROS)[:nbytes]) + enc.finalize()
    return np.frombuffer(out, dtype=np.uint32).copy()
