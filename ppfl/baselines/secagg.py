"""SecAgg (Bonawitz et al., CCS 2017) and SecAgg+ (Bell et al., CCS 2020), semi-honest variants.

Same primitives and accounting as ppfl.protocol: X25519, HKDF-SHA256, AES-128-CTR masks in
Z_2^32, AES-GCM for share transport, Shamir over GF(2^31 - 1) (shamir.py). SecAgg uses the
complete graph; SecAgg+ a randomly relabelled Harary graph H(k, n) with threshold t among the k
neighbours. Rounds: 0 AdvertiseKeys, 1 ShareKeys, 2 MaskedInputCollection, 3 Unmasking. There is
no consistency-check round and there are no signatures (semi-honest setting), so these are the
cheapest variants of both baselines.
"""

import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .. import codec, prg
from ..keys import KAKey
from ..ledger import Ledger
from . import shamir

HDR = struct.Struct(">II")  # idx, round
HDR_LEN = HDR.size  # 8
ID_LEN = 4
NONCE_LEN, TAG_LEN = 12, 16
PT_LEN = 2 * ID_LEN + 2 * shamir.SHARE_LEN  # u, v, share of s_u^SK, share of b_u
CT_LEN = NONCE_LEN + PT_LEN + TAG_LEN  # 108
SALT = b"ppfl/secagg"


def complete_graph(n):
    return [set(range(n)) - {u} for u in range(n)]


def harary_graph(n, k, rng):
    """k-regular Harary graph on a random relabelling of the n users (k even, k < n)."""
    if k % 2 or k >= n:
        raise ValueError("k must be even and < n")
    pos = rng.permutation(n)  # pos[u] = position of user u on the ring
    at = np.empty(n, dtype=np.int64)
    at[pos] = np.arange(n)
    return [{int(at[(pos[u] + d) % n]) for d in range(-k // 2, k // 2 + 1) if d} for u in range(n)]


def enc_key(shared: bytes) -> bytes:
    return prg.hkdf(shared, SALT, b"share-transport", 16)


def pair_key(shared: bytes, u: int, v: int) -> bytes:
    return prg.mask_key(shared, SALT, 0, 0, min(u, v), max(u, v))


def self_key(b: bytes) -> bytes:
    return prg.hkdf(b, SALT, b"self-mask")


# ----- per-party computations (timed by the benchmarks) ------------------------------------------

def client_share_keys(u, c_key, s_raw, b, nbr_cpk: dict, t):
    """Round 1: Shamir-share s_u^SK and b_u among the neighbours, encrypt each share pair."""
    nbrs = sorted(nbr_cpk)
    shares = shamir.share([s_raw, b], [v + 1 for v in nbrs], t).astype("<u4")
    out, keys = {}, {}
    for row, v in zip(shares, nbrs):
        keys[v] = enc_key(c_key.agree(nbr_cpk[v]))
        nonce = os.urandom(NONCE_LEN)
        out[v] = nonce + AESGCM(keys[v]).encrypt(nonce, struct.pack(">II", u, v) + row.tobytes(), None)
    return out, keys


def client_masked_input(u, x, s_key, b, nbr_spk: dict, P, f):
    """Round 2: y_u = enc(x_u) + PRG(b_u) + sum_v +-PRG(s_uv) over neighbours that shared keys."""
    y = codec.encode(x, f) + prg.expand(self_key(b), P)
    for v in sorted(nbr_spk):
        m = prg.expand(pair_key(s_key.agree(nbr_spk[v]), u, v), P)
        if u < v:
            y += m
        else:
            y -= m
    return y


def client_unmask(u, keys: dict, cts: dict, survivors: set):
    """Round 3: decrypt the received shares; release b-shares of survivors, s-shares of dropouts."""
    out = {}
    for v, ct in cts.items():
        pt = AESGCM(keys[v]).decrypt(ct[:NONCE_LEN], ct[NONCE_LEN:], None)
        body = pt[2 * ID_LEN:]
        out[v] = body[shamir.SHARE_LEN:] if v in survivors else body[:shamir.SHARE_LEN]
    return out


def _reconstruct_complete(owners, released, t):
    """Complete graph: every owner is reconstructed from a t-subset of the first t+1 responders.
    With lambda^B the Lagrange coefficients of that (t+1)-set B, the coefficients of B minus one
    holder o are lambda^B_j (x_o - x_j) / x_o, so one O(t^2) step serves all owners."""
    p = shamir.PRIME
    B = sorted(released)[:t + 1]
    x = np.array(B, dtype=np.int64) + 1
    lam = shamir.lagrange_at_zero(x)
    inv_x = np.array([pow(int(v), p - 2, p) for v in x], dtype=np.int64)
    Lam = lam[None, :] * ((x[:, None] - x[None, :]) % p) % p * inv_x[:, None] % p  # row r: B minus B[r]
    zero = bytes(shamir.SHARE_LEN)
    sh = np.stack([np.frombuffer(b"".join(released[h].get(o, zero) for o in owners), dtype="<u4")
                   .reshape(len(owners), shamir.LIMBS) for h in B]).astype(np.int64)  # (t+1, M, LIMBS)
    pos = {h: r for r, h in enumerate(B)}
    rows = Lam[[pos.get(o, t) for o in owners]]  # (M, t+1)
    limbs = (sh.transpose(1, 0, 2) * rows[:, :, None] % p).sum(axis=1) % p
    return {o: shamir.from_limbs(limbs[m]) for m, o in enumerate(owners)}


def _reconstruct_sparse(owners, released, graph, t):
    out, cache = {}, {}
    for o in owners:
        holders = tuple(sorted(h for h in graph[o] if h in released and o in released[h])[:t])
        if len(holders) < t:
            raise RuntimeError(f"not enough shares for user {o}")
        if holders not in cache:
            cache[holders] = shamir.lagrange_at_zero(np.array(holders) + 1)
        sh = np.frombuffer(b"".join(released[h][o] for h in holders), dtype="<u4").reshape(t, shamir.LIMBS)
        limbs = (sh.astype(np.int64) * cache[holders][:, None] % shamir.PRIME).sum(axis=0) % shamir.PRIME
        out[o] = shamir.from_limbs(limbs)
    return out


def reconstruct(owners, released, graph, t, complete: bool):
    if complete and len(released) > t:
        return _reconstruct_complete(sorted(owners), released, t)
    return _reconstruct_sparse(sorted(owners), released, graph, t)


def server_unmask(ys: dict, released: dict, graph, survivors, dropped, spk: dict, t, P, complete=False):
    """Reconstruct self masks of survivors and pairwise masks of dropped users; return the sum."""
    z = np.zeros(P, dtype=np.uint32)
    for y in ys.values():
        z += y
    secrets = reconstruct(set(survivors) | set(dropped), released, graph, t, complete)
    for u in survivors:
        z -= prg.expand(self_key(secrets[u]), P)
    for v in dropped:
        sk = KAKey(secrets[v])
        for u in graph[v]:
            if u in survivors:
                m = prg.expand(pair_key(sk.agree(spk[u]), u, v), P)
                if u < v:
                    z -= m
                else:
                    z += m
    return z


def _masked_input_job(args):
    u, x, s_raw, b, nbr_spk, P, f = args
    t0 = time.perf_counter()
    y = client_masked_input(u, x, KAKey(s_raw), b, nbr_spk, P, f)
    return u, y, time.perf_counter() - t0


# ----- orchestration ---------------------------------------------------------------------------

class SecAggSystem:
    def __init__(self, n_users, length, k=None, t=None, frac_bits=16, workers=1, seed=None):
        self.n, self.P, self.f, self.workers = n_users, length, frac_bits, workers
        rng = np.random.default_rng(seed)
        self.complete = k is None
        self.graph = complete_graph(n_users) if k is None else harary_graph(n_users, k, rng)
        deg = n_users - 1 if k is None else k
        self.t = t if t is not None else deg // 2 + 1
        self.ledger = Ledger()

    def run_round(self, inputs: dict, drop_before_share=(), drop_after_share=(), drop_after_mask=()):
        n, P, L, g = self.n, self.P, self.ledger, self.graph
        users = sorted(inputs)
        # Round 0: advertise keys
        c_keys, s_keys = {}, {}
        for u in users:
            with L.clock(u, "client", "r0_keygen"):
                c_keys[u], s_keys[u] = KAKey(), KAKey()
            L.send(u, "client", "S", "server", "server", "keys", HDR_LEN + 64)
        U0 = set(users)
        for u in users:
            L.send("S", "server", u, "client", "server", "key_list",
                   HDR_LEN + len(g[u] & U0) * (ID_LEN + 64))
        # Round 1: share keys
        U1 = U0 - set(drop_before_share)
        b_seeds, enc_keys, inbox = {}, {}, {u: {} for u in users}
        for u in sorted(U1):
            nbr_cpk = {v: c_keys[v].pk for v in g[u] & U0}
            with L.clock(u, "client", "r1_share"):
                b_seeds[u] = os.urandom(32)
                cts, enc_keys[u] = client_share_keys(u, c_keys[u], s_keys[u].raw(), b_seeds[u], nbr_cpk, self.t)
            L.send(u, "client", "S", "server", "server", "shares_ct", HDR_LEN + len(cts) * (ID_LEN + CT_LEN))
            for v, ct in cts.items():
                inbox[v][u] = ct
        for u in sorted(U1):
            got = {v: ct for v, ct in inbox[u].items() if v in U1}
            inbox[u] = got
            L.send("S", "server", u, "client", "server", "shares_fwd", HDR_LEN + len(got) * (ID_LEN + CT_LEN))
        # Round 2: masked inputs
        U2 = U1 - set(drop_after_share)
        jobs = [(u, inputs[u], s_keys[u].raw(), b_seeds[u], {v: s_keys[v].pk for v in g[u] & U1},
                 P, self.f) for u in sorted(U2)]
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(_masked_input_job, jobs, chunksize=max(1, len(jobs) // (4 * self.workers))))
        else:
            results = [_masked_input_job(j) for j in jobs]
        ys = {}
        for u, y, dt in results:
            L.charge([u], "client", "r2_mask", dt)
            L.send(u, "client", "S", "server", "server", "w_tilde", HDR_LEN + 4 * P)
            ys[u] = y
        dropped = U1 - U2
        for u in sorted(U2):
            L.send("S", "server", u, "client", "server", "survivors", HDR_LEN + 4 + 4 * len(dropped))
        # Round 3: unmasking
        U3 = U2 - set(drop_after_mask)
        released = {}
        for u in sorted(U3):
            cts = {v: ct for v, ct in inbox[u].items() if v in U2 or v in dropped}
            with L.clock(u, "client", "r3_unmask"):
                released[u] = client_unmask(u, enc_keys[u], cts, U2)
            L.send(u, "client", "S", "server", "server", "unmask_shares",
                   HDR_LEN + len(released[u]) * (ID_LEN + shamir.SHARE_LEN))
        with L.clock("S", "server", "r3_reconstruct"):
            z = server_unmask(ys, released, g, U2, dropped, {u: s_keys[u].pk for u in users}, self.t, P,
                              self.complete)
        for u in sorted(U3):
            L.send("S", "server", u, "client", "server", "model", 4 * P)
        return z, sorted(U2)


def round_bytes(P, n_users, deg):
    """Bytes per client (deg neighbours) and for the server in a round without dropouts."""
    keys, key_list = HDR_LEN + 64, HDR_LEN + deg * (ID_LEN + 64)
    shares = HDR_LEN + deg * (ID_LEN + CT_LEN)
    unmask, survivors = HDR_LEN + deg * (ID_LEN + shamir.SHARE_LEN), HDR_LEN + 4
    return {"client": {"up": keys + shares + unmask, "down": key_list + shares + survivors,
                       "up_baseline": HDR_LEN + 4 * P, "down_baseline": 4 * P},
            "server": {"up": n_users * (key_list + shares + survivors),
                       "down": n_users * (keys + shares + unmask),
                       "up_baseline": n_users * 4 * P, "down_baseline": n_users * (HDR_LEN + 4 * P)}}
