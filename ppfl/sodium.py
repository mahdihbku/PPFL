"""Native libsodium (Homebrew build) via ctypes, or None if unavailable.

PyNaCl's bundled libsodium is about 15x slower on Apple M3 (Ed25519 verify 494 us vs 33 us), so
the VRF and signatures use the native library when present. Set PPFL_NATIVE_SODIUM=0 to disable,
or PPFL_LIBSODIUM=/path/to/libsodium.dylib to point at another build.
"""

import ctypes
import os

_PATHS = [os.environ.get("PPFL_LIBSODIUM", ""), "/opt/homebrew/opt/libsodium/lib/libsodium.dylib",
          "/usr/local/opt/libsodium/lib/libsodium.dylib"]


def _load():
    if os.environ.get("PPFL_NATIVE_SODIUM", "1") == "0":
        return None
    for path in _PATHS:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(path)
            if lib.sodium_init() < 0:
                raise OSError("sodium_init failed")
            lib.sodium_version_string.restype = ctypes.c_char_p
            return lib
    return None


LIB = _load()
VERSION = LIB.sodium_version_string().decode() if LIB else None
