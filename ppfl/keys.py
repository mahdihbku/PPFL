"""Thin wrappers over Ed25519 signatures and X25519 key agreement (raw 32-byte encodings).

Ed25519 uses native libsodium when available (sodium.py), else OpenSSL via `cryptography`; both
implement RFC 8032, so keys and signatures are interchangeable. X25519 uses `cryptography`
(native libsodium is not faster for it on this machine).
"""

import ctypes
import hashlib
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature

from . import sodium

SIG_LEN = 64
PK_LEN = 32


def sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _raw(pub) -> bytes:
    return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)


class SigningKey:
    def __init__(self, raw: bytes | None = None):
        self._seed = raw if raw else os.urandom(32)
        if sodium.LIB is not None:
            pk, self._sk64 = ctypes.create_string_buffer(32), ctypes.create_string_buffer(64)
            sodium.LIB.crypto_sign_seed_keypair(pk, self._sk64, self._seed)
            self.pk = pk.raw
        else:
            self._sk = Ed25519PrivateKey.from_private_bytes(self._seed)
            self.pk = _raw(self._sk.public_key())

    def raw(self) -> bytes:
        return self._seed

    def sign(self, msg: bytes) -> bytes:
        if sodium.LIB is not None:
            sig = ctypes.create_string_buffer(SIG_LEN)
            sodium.LIB.crypto_sign_detached(sig, None, msg, ctypes.c_ulonglong(len(msg)), self._sk64)
            return sig.raw
        return self._sk.sign(msg)


_pk_cache: dict[bytes, Ed25519PublicKey] = {}


def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
    if sodium.LIB is not None:
        if len(sig) != SIG_LEN or len(pk) != PK_LEN:
            return False
        return sodium.LIB.crypto_sign_verify_detached(sig, msg, ctypes.c_ulonglong(len(msg)), pk) == 0
    key = _pk_cache.get(pk)
    if key is None:
        key = _pk_cache[pk] = Ed25519PublicKey.from_public_bytes(pk)
    try:
        key.verify(sig, msg)
        return True
    except InvalidSignature:
        return False


class KAKey:
    """X25519 key pair (used either as a long-term static key or as a per-round ephemeral key)."""

    def __init__(self, raw: bytes | None = None):
        self._sk = X25519PrivateKey.from_private_bytes(raw) if raw else X25519PrivateKey.generate()
        self.pk = _raw(self._sk.public_key())

    def raw(self) -> bytes:
        return self._sk.private_bytes_raw()

    def agree(self, peer_pk: bytes) -> bytes:
        return self._sk.exchange(X25519PublicKey.from_public_bytes(peer_pk))


def random_bytes(n: int) -> bytes:
    return os.urandom(n)
