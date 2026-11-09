"""ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381, suite 0x03).

Group operations run in libsodium: the native library (Homebrew build, loaded with ctypes) when
available, otherwise PyNaCl's bundled build, which is about 15x slower on Apple M3. Point
decoding (RFC 8032, Sec. 5.1.3) runs in Python because the try-and-increment hash must accept any
curve point; cofactor clearing uses point additions, which libsodium allows outside the
prime-order subgroup. Deviation from RFC 9381: a proof whose Gamma (or a public key) is not in the
prime-order subgroup is rejected; honest proofs always satisfy this.
"""

import ctypes
import hashlib
import os
from functools import lru_cache

from . import sodium

SUITE = b"\x03"
Q = 2**252 + 27742317777372353535851937790883648493
_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)
C_LEN = 16
PROOF_LEN = 80
BETA_LEN = 64
IDENTITY = b"\x01" + bytes(31)

class _Native:
    name = "libsodium-native"

    def __init__(self, lib, version):
        self.lib, self.version = lib, version

    def _call(self, fn, *args):
        out = ctypes.create_string_buffer(32)
        if fn(out, *args) != 0:
            raise ValueError("libsodium rejected the input")
        return out.raw

    def base(self, n: bytes) -> bytes:
        return self._call(self.lib.crypto_scalarmult_ed25519_base_noclamp, n)

    def smul(self, n: bytes, p: bytes) -> bytes:
        return self._call(self.lib.crypto_scalarmult_ed25519_noclamp, n, p)

    def add(self, p: bytes, q: bytes) -> bytes:
        return self._call(self.lib.crypto_core_ed25519_add, p, q)

    def sub(self, p: bytes, q: bytes) -> bytes:
        return self._call(self.lib.crypto_core_ed25519_sub, p, q)


class _PyNaCl:
    name = "pynacl-bundled"

    def __init__(self):
        import nacl.bindings as nb
        from nacl.exceptions import RuntimeError as NaclError
        self.nb, self.err = nb, NaclError
        self.version = "PyNaCl bundled"

    def _wrap(self, fn, *args):
        try:
            return fn(*args)
        except self.err as e:
            raise ValueError("libsodium rejected the input") from e

    def base(self, n):
        return self._wrap(self.nb.crypto_scalarmult_ed25519_base_noclamp, n)

    def smul(self, n, p):
        return self._wrap(self.nb.crypto_scalarmult_ed25519_noclamp, n, p)

    def add(self, p, q):
        return self._wrap(self.nb.crypto_core_ed25519_add, p, q)

    def sub(self, p, q):
        return self._wrap(self.nb.crypto_core_ed25519_sub, p, q)


def _load_backend():
    if sodium.LIB is not None and os.environ.get("PPFL_VRF_BACKEND") != "pynacl":
        return _Native(sodium.LIB, sodium.VERSION)
    return _PyNaCl()


BACKEND = _load_backend()


def _sha512(*parts: bytes) -> bytes:
    h = hashlib.sha512()
    for p in parts:
        h.update(p)
    return h.digest()


def _is_point(s: bytes) -> bool:
    """RFC 8032 decoding succeeds (canonical y < p, x recoverable, no negative zero)."""
    if len(s) != 32:
        return False
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        return False
    u = (y * y - 1) % _P
    v = (_D * y * y + 1) % _P
    x = (u * pow(v, 3, _P) * pow(u * pow(v, 7, _P), (_P - 5) // 8, _P)) % _P
    vx2 = v * x * x % _P
    if vx2 != u and vx2 != (-u) % _P:
        return False
    if vx2 != u:
        x = x * _SQRT_M1 % _P
    return not (x == 0 and sign == 1)


def _mul_cofactor(s: bytes) -> bytes:
    for _ in range(3):
        s = BACKEND.add(s, s)
    return s


def _scalar(k: int) -> bytes:
    return (k % Q).to_bytes(32, "little")


def _secret_scalar(sk: bytes) -> tuple[int, bytes]:
    h = _sha512(sk)
    a = bytearray(h[:32])
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(a, "little"), h


def encode_to_curve(pk: bytes, alpha: bytes) -> bytes:
    for ctr in range(256):
        h = _sha512(SUITE, b"\x01", pk, alpha, bytes([ctr]), b"\x00")[:32]
        if not _is_point(h):
            continue
        pt = _mul_cofactor(h)
        if pt != IDENTITY:
            return pt
    raise ValueError("encode_to_curve failed")


def _challenge(*points: bytes) -> int:
    return int.from_bytes(_sha512(SUITE, b"\x02", *points, b"\x00")[:C_LEN], "little")


def keygen(sk: bytes | None = None) -> tuple[bytes, bytes]:
    sk = sk if sk is not None else os.urandom(32)
    x, _ = _secret_scalar(sk)
    return sk, BACKEND.base(_scalar(x))


def prove(sk: bytes, alpha: bytes) -> bytes:
    x, h = _secret_scalar(sk)
    pk = BACKEND.base(_scalar(x))
    H = encode_to_curve(pk, alpha)
    gamma = BACKEND.smul(_scalar(x), H)
    k = int.from_bytes(_sha512(h[32:], H), "little") % Q
    c = _challenge(pk, H, gamma, BACKEND.base(_scalar(k)), BACKEND.smul(_scalar(k), H))
    s = (k + c * x) % Q
    return gamma + c.to_bytes(C_LEN, "little") + s.to_bytes(32, "little")


def output(sk: bytes, alpha: bytes) -> bytes:
    """beta = proof_to_hash(prove(sk, alpha)) without computing the proof (for simulations)."""
    x, _ = _secret_scalar(sk)
    pk = BACKEND.base(_scalar(x))
    gamma = BACKEND.smul(_scalar(x), encode_to_curve(pk, alpha))
    return _sha512(SUITE, b"\x03", _mul_cofactor(gamma), b"\x00")


def proof_to_hash(pi: bytes) -> bytes:
    if not _is_point(pi[:32]):
        raise ValueError("invalid proof")
    return _sha512(SUITE, b"\x03", _mul_cofactor(pi[:32]), b"\x00")


@lru_cache(maxsize=1 << 16)
def _valid_key(pk: bytes) -> bool:
    try:
        return _is_point(pk) and _mul_cofactor(pk) != IDENTITY
    except ValueError:
        return False


def verify(pk: bytes, alpha: bytes, pi: bytes) -> bytes | None:
    """Return beta if the proof is valid, else None (validate_key = TRUE)."""
    if len(pi) != PROOF_LEN or not _valid_key(pk):
        return None
    gamma, c, s = pi[:32], int.from_bytes(pi[32:48], "little"), int.from_bytes(pi[48:], "little")
    if s >= Q or not _is_point(gamma):
        return None
    try:
        H = encode_to_curve(pk, alpha)
        u = BACKEND.sub(BACKEND.base(_scalar(s)), BACKEND.smul(_scalar(c), pk))
        v = BACKEND.sub(BACKEND.smul(_scalar(s), H), BACKEND.smul(_scalar(c), gamma))
        if _challenge(pk, H, gamma, u, v) != c:
            return None
        return _sha512(SUITE, b"\x03", _mul_cofactor(gamma), b"\x00")
    except ValueError:
        return None


def leading_zero_bits(beta: bytes) -> int:
    n = int.from_bytes(beta, "big")
    return len(beta) * 8 - n.bit_length()


def selected(beta: bytes, L: int) -> bool:
    """Committee rule: at least L leading zero bits, so p = 2^-L."""
    return leading_zero_bits(beta) >= L


BETA_BITS = 8 * BETA_LEN


def threshold(p: float) -> int:
    """Selection threshold for probability p: selected iff int(beta) < p * 2^512."""
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    return int(p * (1 << BETA_BITS))


def threshold_for_L(L: int) -> int:
    """Equivalent threshold of the leading-zeros rule (at least L leading zero bits)."""
    return 1 << (BETA_BITS - L)


def below(beta: bytes, thr: int) -> bool:
    return int.from_bytes(beta, "big") < thr
