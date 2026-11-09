"""Verifiable committee-based secure aggregation, one FL round per System.run_round() call.

Message flow (revised Fig. 2); control messages go to the bulletin board (BB), bulk vectors via
the server:
  1. election   member -> BB    (idx, t, a, VRF proof, member epk)               committee >= c_min
  2. control    client -> BB    (idx, t, a, client epk, LHH(w~_i), [d_i], H(w~_i))
     w_tilde    client -> S     w~_i = enc(x_i) + sum_j PRG(s_ij)
  3. delta      S -> BB         (t, a, D, Delta_t); members refuse |Delta_t| > d_max
  4. mcontrol   member -> BB    (idx, t, a, commit(seed_j), LHH(r~^j), H(r~^j))
     subsum     member -> S     r~^j = sum_{i in I_t} PRG(s_ij) + PRG(seed_j)
  5. AK         S -> member     A = sum w~_i - sum r~^j mod 2^32, carries K (int16)
  6. reveal     member -> BB    seed_j, only if LHH(A + 2^32 K) + sum LHH(r~^j) == sum LHH(w~_i)
  7. cosign     member -> BB    H(aggregate); model S -> client, checked against all C co-signatures
Failures: missing/invalid member messages abort the attempt, put the member on cooldown and
re-elect (at most max_retries times); every election is counted for the multi-round bound.

The per-party computations are module-level functions so benchmarks time exactly this code.
"""

import struct
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import codec, lhh, prg, vrf
from .baselines import shamir
from .bb import BulletinBoard
from .keys import KAKey, SigningKey, random_bytes, sha256, verify
from .ledger import Ledger

U32 = np.uint32
HDR = struct.Struct(">IIB")  # idx, round, attempt
HDR_LEN = HDR.size  # 9
USER_INFO_LEN = 20  # connection metadata per enrolled user (address, port)


@dataclass
class Params:
    n_users: int
    length: int  # P, number of aggregated coordinates
    L: int  # selected iff beta has >= L leading zero bits (p = 2^-L), unless select_p is set
    c_min: int  # minimum accepted committee size
    frac_bits: int = 16
    fs: str = "ephemeral"  # "ephemeral" (per-round X25519, forward secrecy) or "static"
    lhh: bool = True  # committee-side aggregate integrity check
    cosign: bool = True  # committee co-signs the new model
    similarity: bool = False  # optional server-side consistency score (leaks <w^t, w_i>)
    max_retries: int = 3
    cooldown_rounds: int = 10
    tau: int = 0  # committee-member failures recovered per attempt (0: abort on any failure)
    select_p: float | None = None  # selection probability: selected iff int(beta) < p * 2^512
    d_max: int | None = None  # largest dropout announcement members accept (None: n_users // 2)

    @property
    def dropout_cap(self) -> int:
        return self.n_users // 2 if self.d_max is None else self.d_max

    @property
    def threshold(self) -> int:
        return vrf.threshold(self.select_p) if self.select_p is not None else vrf.threshold_for_L(self.L)

    def __post_init__(self):
        if self.fs not in ("ephemeral", "static"):
            raise ValueError("fs must be 'ephemeral' or 'static'")
        if self.tau and self.fs != "ephemeral":
            raise ValueError("recovery (tau > 0) shares per-round member keys; it needs fs='ephemeral'")
        if self.lhh and self.n_users >= (1 << 14):
            raise ValueError("LHH lane bound requires n_users < 2^14")


@dataclass
class Behavior:
    """Fault and attack injection for one round."""

    offline: set = field(default_factory=set)
    drop_before_submit: set = field(default_factory=set)  # Case 1
    drop_after_submit: set = field(default_factory=set)  # Case 2
    member_drop: dict = field(default_factory=dict)  # idx -> "before_shares" | "before_subsum" (Case 3) | "before_reveal" (Case 4)
    member_bad_seed: set = field(default_factory=set)
    client_bad_lhh: set = field(default_factory=set)
    server_delta: np.ndarray | None = None  # added to the aggregate by a cheating server
    server_equivocate: bool = False  # two different dropout announcements
    server_tamper_model_for: set = field(default_factory=set)
    server_exclude: set = field(default_factory=set)  # submitters the server falsely lists as dropped


@dataclass
class Attempt:
    attempt: int
    committee: list
    outcome: str
    culprits: list = field(default_factory=list)
    recovered: list = field(default_factory=list)  # members whose failure was recovered (tau > 0)


@dataclass
class RoundResult:
    status: str  # "ok" | "failed" | "detected"
    aggregate: np.ndarray | None  # decoded sum over the participation set
    encoded_sum: np.ndarray | None  # same, in Z_2^32
    participants: list
    attempts: list
    detections: list


class User:
    def __init__(self, idx: int, fs: str):
        self.idx = idx
        self.sig = SigningKey()
        self.vrf_sk, self.vrf_pk = vrf.keygen()
        self.static_ka = KAKey() if fs == "static" else None

    def enrollment_entry(self) -> bytes:
        return (struct.pack(">I", self.idx) + self.sig.pk + self.vrf_pk
                + (self.static_ka.pk if self.static_ka else b"") + bytes(USER_INFO_LEN))


# ----- per-party computations ------------------------------------------------------------------

def verify_committee(posts, sig_pks, vrf_pks, eligible, alpha, threshold):
    """Valid election posts: eligible author, valid signature, valid VRF proof meeting L."""
    out = {}
    for e in posts:
        i = e.author
        if i not in eligible or not verify(sig_pks[i], e.payload, e.sig):
            continue
        beta = vrf.verify(vrf_pks[i], alpha, e.payload[HDR_LEN:HDR_LEN + vrf.PROOF_LEN])
        if beta is not None and vrf.below(beta, threshold):
            out[i] = e
    return out


def client_mask(ka, i, x, member_pk: dict, D, t, a, P, f):
    r = np.zeros(P, dtype=U32)
    for j in sorted(member_pk):
        r += prg.expand(prg.mask_key(ka.agree(member_pk[j]), D, t, a, i, j), P)
    x_enc = codec.encode(x, f)
    return x_enc + r, x_enc


def member_derive(ka, j, client_pk: dict, D, t, a, P):
    acc = np.zeros(P, dtype=U32)
    for i in sorted(client_pk):
        acc += prg.expand(prg.mask_key(ka.agree(client_pk[i]), D, t, a, i, j), P)
    return acc


def commitment(t, a, j, seed) -> bytes:
    return sha256(b"ppfl/commit", struct.pack(">IBI", t, a, j), seed)


def member_commit(acc, j, t, a, P, seed=None):
    seed = random_bytes(32) if seed is None else seed
    return acc + prg.expand(prg.commit_mask_key(seed, t, a, j), P), seed, commitment(t, a, j, seed)


SHARE_CT_LEN = 12 + 2 * shamir.SHARE_LEN + 16  # nonce, shares of (esk_j, seed_j), GCM tag
SHARE_MSG_LEN = HDR_LEN + 4 + SHARE_CT_LEN  # + recipient index
RELEASE_MSG_LEN = HDR_LEN + 4 + 1 + shamir.SHARE_LEN  # + owner index, secret selector, share


def _share_key(ka, peer_pk, D, t, a, owner, holder) -> bytes:
    info = b"ppfl/member-share/v1" + struct.pack(">IBII", t, a, owner, holder)
    return prg.hkdf(ka.agree(peer_pk), D, info, 16)


def member_share_secrets(j, ka, seed, holder_pk: dict, t_s, D, t, a):
    """Shamir-share (esk_j, seed_j) among the other members (threshold t_s), one AES-GCM
    ciphertext per holder. Returns {holder: payload}."""
    holders = sorted(holder_pk)
    shares = shamir.share([ka.raw(), seed], [h + 1 for h in holders], t_s).astype("<u4")
    out = {}
    for row, h in zip(shares, holders):
        hdr = HDR.pack(j, t, a) + struct.pack(">I", h)
        nonce = random_bytes(12)
        out[h] = hdr + nonce + AESGCM(_share_key(ka, holder_pk[h], D, t, a, j, h)).encrypt(nonce, row.tobytes(), hdr)
    return out


def member_open_share(h, ka, owner_pk, payload, D, t, a):
    """Holder h decrypts its share bundle from the owner; returns (esk-share, seed-share) bytes."""
    j = HDR.unpack(payload[:HDR_LEN])[0]
    hdr, nonce, ct = payload[:HDR_LEN + 4], payload[HDR_LEN + 4:HDR_LEN + 16], payload[HDR_LEN + 16:]
    pt = AESGCM(_share_key(ka, owner_pk, D, t, a, j, h)).decrypt(nonce, ct, hdr)
    return pt[:shamir.SHARE_LEN], pt[shamir.SHARE_LEN:]


def reconstruct_secret(releases: dict, t_s) -> bytes:
    """releases: {holder: share bytes}; uses the t_s lowest holder indices."""
    holders = sorted(releases)[:t_s]
    sh = np.stack([np.frombuffer(releases[h], dtype="<u4").astype(np.int64) for h in holders])
    return shamir.reconstruct([h + 1 for h in holders], sh, 1)[0]


def server_combine(uploads, subsums, P, delta=None):
    """A = sum uploads - sum subsums mod 2^32 and the integer carries K (A + 2^32 K = exact)."""
    Z = np.zeros(P, dtype=np.int64)
    for w in uploads:
        Z += w
    for s in subsums:
        Z -= s
    if delta is not None:
        Z = Z + np.asarray(delta, dtype=np.int64)
    A = np.mod(Z, 1 << 32).astype(U32)
    K = (Z - A.astype(np.int64)) >> 32
    if np.abs(K).max(initial=0) >= (1 << 15):
        raise OverflowError("carry out of int16 range")
    return A, K.astype(np.int16)


def lhh_check(A, K, member_hashes, client_hashes) -> bool:
    lhs = lhh.combine([lhh.hash_vector(A.astype(np.int64) + (K.astype(np.int64) << 32))]
                      + list(member_hashes))
    return lhs == lhh.combine(list(client_hashes))


def open_commitments(reveals, commitments: dict, sig_pks: dict, t, a, P):
    bad, M = [], np.zeros(P, dtype=U32)
    for e in reveals:
        j, seed = e.author, e.payload[HDR_LEN:HDR_LEN + 32]
        if not verify(sig_pks[j], e.payload, e.sig) or commitment(t, a, j, seed) != commitments[j]:
            bad.append(j)
            continue
        M += prg.expand(prg.commit_mask_key(seed, t, a, j), P)
    return bad, M


def model_hash(t, a, enc_sum_bytes) -> bytes:
    return sha256(b"ppfl/model", struct.pack(">IB", t, a), enc_sum_bytes)


def client_job(args):
    """Everything client i computes after the committee is known (picklable for process pools)."""
    (i, sig_raw, static_raw, x, member_pk, D, t, a, P, f, eph, use_lhh, bad_lhh, g) = args
    signer = SigningKey(sig_raw)
    times = {}
    t0 = time.perf_counter()
    ka = KAKey() if eph else KAKey(static_raw)
    w, x_enc = client_mask(ka, i, x, member_pk, D, t, a, P, f)
    times["mask"] = time.perf_counter() - t0
    h = d = b""
    if use_lhh:
        t0 = time.perf_counter()
        v = w.astype(np.int64)
        h = lhh.hash_vector(v + 1 if bad_lhh else v)
        times["lhh"] = time.perf_counter() - t0
    if g is not None:
        t0 = time.perf_counter()
        d = struct.pack(">d", float(np.dot(g, codec.decode(x_enc, f))))
        times["similarity"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    payload = HDR.pack(i, t, a) + (ka.pk if eph else b"") + h + d + sha256(w.tobytes())
    sig = signer.sign(payload)
    times["control_post"] = time.perf_counter() - t0
    return i, w, x_enc, payload, sig, times


# ----- orchestration ---------------------------------------------------------------------------

class _Abort(Exception):
    def __init__(self, outcome, culprits=()):
        self.outcome, self.culprits = outcome, list(culprits)


class _Detected(Exception):
    pass


class System:
    def __init__(self, params: Params, workers: int = 1):
        self.p = params
        self.workers = workers
        self.users = [User(i, params.fs) for i in range(params.n_users)]
        self.sig_pks = {u.idx: u.sig.pk for u in self.users}
        self.vrf_pks = {u.idx: u.vrf_pk for u in self.users}
        self.server_sig = SigningKey()
        self.ledger = Ledger()
        self.bb = BulletinBoard(self.ledger)
        self.cooldown_until: dict[int, int] = {}
        self.elections = 0
        roster = b"".join(u.enrollment_entry() for u in self.users)
        self.digest = sha256(b"ppfl/list_of_users", roster)
        self._alpha_chain = [sha256(b"ppfl/alpha", self.digest)]
        for u in self.users:  # offline phase: enrollment upload, roster download
            entry = u.enrollment_entry()
            self.ledger.send(u.idx, "client", "S", "server", "offline", "enroll", len(entry) - 4)
            self.ledger.send("S", "server", u.idx, "client", "offline", "roster", len(roster))

    def _client_layout(self):
        off, lay = HDR_LEN, {}
        for name, n in (("epk", 32 if self.p.fs == "ephemeral" else 0),
                        ("h", lhh.POINT_LEN if self.p.lhh else 0),
                        ("d", 8 if self.p.similarity else 0),
                        ("hw", 32)):
            lay[name] = (off, off + n)
            off += n
        return lay

    def alpha(self, t: int, attempt: int) -> bytes:
        """VRF input: hash chain over rounds (list_of_users unchanged), re-randomised per retry."""
        while len(self._alpha_chain) <= t:
            self._alpha_chain.append(sha256(self._alpha_chain[-1]))
        a = self._alpha_chain[t]
        return a if attempt == 0 else sha256(a, b"retry", bytes([attempt]))

    def preview_committee(self, t: int, attempt: int, online=None) -> list[int]:
        """Users that would be elected (no side effects, no accounting); used by tests."""
        alpha = self.alpha(t, attempt)
        users = self.users if online is None else [self.users[i] for i in online]
        return sorted(u.idx for u in users
                      if self.cooldown_until.get(u.idx, -1) < t
                      and vrf.below(vrf.proof_to_hash(vrf.prove(u.vrf_sk, alpha)), self.p.threshold))

    # ----- round ---------------------------------------------------------------------------
    def run_round(self, t: int, inputs: dict, behavior: Behavior | None = None,
                  global_model: np.ndarray | None = None) -> RoundResult:
        b = behavior or Behavior()
        online = [u.idx for u in self.users if u.idx not in b.offline]
        member_drop = dict(b.member_drop)
        excluded: set[int] = set()
        attempts, detections = [], []

        def punish(culprits):
            for c in culprits:  # failed members sit out the cooldown
                self.cooldown_until[c] = t + self.p.cooldown_rounds
                member_drop.pop(c, None)

        for a in range(self.p.max_retries + 1):
            self.elections += 1
            elected, member_keys, posts = self._elect(t, a, online)
            if len(elected) < self.p.c_min:
                attempts.append(Attempt(a, elected, "small_committee"))
                continue
            state = self._share_secrets(t, a, elected, member_keys, posts, member_drop)
            try:
                enc_sum, participants, recovered = self._aggregate(
                    t, a, state["committee"], member_keys, posts, online, excluded, inputs, b,
                    member_drop, global_model, detections, state)
            except _Abort as ab:
                attempts.append(Attempt(a, elected, ab.outcome, ab.culprits))
                punish(ab.culprits)
                excluded.update(ab.culprits)  # and the rest of this round
                continue
            except _Detected as d:
                attempts.append(Attempt(a, elected, "detected"))
                detections.append(str(d))
                return RoundResult("detected", None, None, [], attempts, detections)
            attempts.append(Attempt(a, elected, "ok", recovered=state["dropped"] + recovered))
            punish(state["dropped"] + recovered)
            agg = codec.decode(enc_sum, self.p.frac_bits)
            return RoundResult("ok", agg, enc_sum, participants, attempts, detections)
        return RoundResult("failed", None, None, [], attempts, detections)

    def _share_secrets(self, t, a, elected, member_keys, posts, member_drop):
        """tau > 0: every elected member shares (esk_j, seed_j) among the other elected members
        (threshold C - tau). Members that miss this step leave the committee before clients mask;
        they count against the tau budget of the attempt."""
        p, L, bb, D = self.p, self.ledger, self.bb, self.digest
        state = {"committee": elected, "dropped": [], "budget": 0, "t_s": None, "seeds": {}, "bundles": {}}
        if not p.tau:
            return state
        off = HDR_LEN + vrf.PROOF_LEN
        epk = {j: posts[j].payload[off:off + 32] for j in elected}
        alive = [j for j in elected if member_drop.get(j) != "before_shares"]
        t_s = len(elected) - p.tau
        for j in alive:
            with L.clock(j, "member", "share_secrets"):
                seed = random_bytes(32)
                bundle = member_share_secrets(j, member_keys[j], seed, {h: epk[h] for h in elected if h != j},
                                              t_s, D, t, a)
                sigs = {h: self.users[j].sig.sign(pl) for h, pl in bundle.items()}
            for h, pl in bundle.items():
                e = bb.post(j, "member", "share", t, a, pl, sigs[h])
                if h in alive:
                    bb.charge_read([h], "member", [e], "share")
                    state["bundles"][(j, h)] = pl
            state["seeds"][j] = seed
        state.update(committee=alive, dropped=[j for j in elected if j not in alive],
                     budget=p.tau - (len(elected) - len(alive)), t_s=t_s, epk=epk)
        return state

    def _release(self, t, a, owner, which, holders, state, member_keys):
        """Surviving holders publish their share of the owner's esk (which=0) or seed (which=1);
        everyone reconstructs from t_s shares."""
        L, bb = self.ledger, self.bb
        releases, entries = {}, []
        for h in holders:
            with L.clock(h, "member", "release_share"):
                esk_sh, seed_sh = member_open_share(h, member_keys[h], state["epk"][owner],
                                                    state["bundles"][(owner, h)], self.digest, t, a)
                body = HDR.pack(h, t, a) + struct.pack(">IB", owner, which) + (seed_sh if which else esk_sh)
                sig = self.users[h].sig.sign(body)
            entries.append(bb.post(h, "member", "release", t, a, body, sig))
            releases[h] = body[HDR_LEN + 5:]
        bb.charge_read(holders, "member", entries, "release")
        bb.charge_read(["S"], "server", entries, "release")
        if len(releases) < state["t_s"]:
            raise _Abort("recovery_failed", [owner])
        return L.shared(holders, "member", "reconstruct", lambda: reconstruct_secret(releases, state["t_s"]),
                        also=[(["S"], "server")])

    def _elect(self, t, a, online):
        p, L, bb = self.p, self.ledger, self.bb
        alpha = self.alpha(t, a)
        eligible = {i for i in online if self.cooldown_until.get(i, -1) < t}
        member_keys = {}
        for i in sorted(eligible):
            u = self.users[i]
            with L.clock(i, "client", "vrf_prove"):
                pi = vrf.prove(u.vrf_sk, alpha)
                chosen = vrf.below(vrf.proof_to_hash(pi), p.threshold)
            if not chosen:
                continue
            with L.clock(i, "member", "election_post"):
                if p.fs == "ephemeral":
                    member_keys[i] = KAKey()
                    epk = member_keys[i].pk
                else:
                    member_keys[i], epk = u.static_ka, b""
                payload = HDR.pack(i, t, a) + pi + epk
                sig = u.sig.sign(payload)
            bb.post(i, "member", "election", t, a, payload, sig)
        posts = bb.query("election", t, a)
        valid = L.shared(online, "client", "verify_committee",
                         lambda: verify_committee(posts, self.sig_pks, self.vrf_pks, eligible, alpha, p.threshold),
                         also=[(["S"], "server")])
        bb.charge_read(online, "client", posts, "election")
        bb.charge_read(["S"], "server", posts, "election")
        committee = sorted(valid)
        return committee, {j: member_keys[j] for j in committee}, valid

    def _run_clients(self, jobs):
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                return list(ex.map(client_job, jobs, chunksize=max(1, len(jobs) // (4 * self.workers))))
        return [client_job(j) for j in jobs]

    def _aggregate(self, t, a, committee, member_keys, posts, online, excluded, inputs, b,
                   member_drop, global_model, detections, state):
        p, L, bb, D = self.p, self.ledger, self.bb, self.digest
        P, f, eph = p.length, p.frac_bits, p.fs == "ephemeral"
        lay = self._client_layout()
        if state["budget"] < 0:  # more than tau members failed before sharing: honest ones may be gone
            raise _Abort("member_dropout", state["dropped"])

        # 2. clients mask and submit
        if eph:
            off = HDR_LEN + vrf.PROOF_LEN
            member_pk = {j: posts[j].payload[off:off + 32] for j in committee}
        else:
            member_pk = {j: self.users[j].static_ka.pk for j in committee}
        submitters = [i for i in online
                      if i in inputs and i not in b.drop_before_submit and i not in excluded]
        jobs = [(i, self.users[i].sig.raw(),
                 self.users[i].static_ka.raw() if not eph else None,
                 inputs[i], member_pk, D, t, a, P, f, eph, p.lhh, i in b.client_bad_lhh,
                 global_model if p.similarity else None) for i in submitters]
        uploads, enc_inputs = {}, {}
        for i, w, x_enc, payload, sig, times in self._run_clients(jobs):
            for phase, dt in times.items():
                L.charge([i], "client", phase, dt)
            bb.post(i, "client", "control", t, a, payload, sig)
            L.send(i, "client", "S", "server", "server", "w_tilde", HDR_LEN + 4 * P)
            uploads[i], enc_inputs[i] = w, x_enc

        # 3. server: participation set and dropout announcement
        ctrl = {e.author: e for e in bb.query("control", t, a)}
        with L.clock("S", "server", "check_uploads"):
            participants = sorted(
                i for i in uploads
                if i in ctrl and sha256(uploads[i].tobytes()) == ctrl[i].payload[slice(*lay["hw"])])
        bb.charge_read(["S"], "server", list(ctrl.values()), "control")
        delta = sorted((set(range(p.n_users)) - set(participants)) | set(b.server_exclude))
        self._post_delta(t, a, delta)
        if b.server_equivocate and participants:
            self._post_delta(t, a, sorted(set(delta) | {participants[0]}))
        deltas = bb.query("delta", t, a)

        def read_delta():
            valid = {e.payload for e in deltas if verify(self.server_sig.pk, e.payload, e.sig)}
            if len(valid) != 1:
                return None
            body = next(iter(valid))
            n = struct.unpack(">I", body[37:41])[0]
            return set(struct.unpack(f">{n}I", body[41:41 + 4 * n]))

        delta_set = L.shared(online, "client", "read_delta", read_delta)
        bb.charge_read(online, "client", deltas, "delta")
        if delta_set is None:
            raise _Detected("equivocation:delta")
        if len(delta_set) > p.dropout_cap:  # members refuse: the aggregate would cover too few users
            raise _Abort("too_many_dropouts")
        for i in submitters:
            if i in delta_set:
                detections.append(f"excluded:{i}")
        I_t = [i for i in range(p.n_users) if i not in delta_set]

        # 4. committee members derive sub-masks, commit, and submit masked sub-sums
        active = [j for j in committee if member_drop.get(j) not in ("before_subsum", "before_shares")]
        ctrl_I = [ctrl[i] for i in I_t]
        L.shared(active, "member", "verify_controls",
                 lambda: all(verify(self.sig_pks[e.author], e.payload, e.sig) for e in ctrl_I))
        bb.charge_read(active, "member", ctrl_I, "control")
        client_pk = {i: (ctrl[i].payload[slice(*lay["epk"])] if eph else self.users[i].static_ka.pk)
                     for i in I_t}
        subs = {}
        for j in active:
            with L.clock(j, "member", "derive_submasks"):
                acc = member_derive(member_keys[j], j, client_pk, D, t, a, P)
            with L.clock(j, "member", "commit"):
                rt, seed, cj = member_commit(acc, j, t, a, P, state["seeds"].get(j))
            hr = b""
            if p.lhh:
                with L.clock(j, "member", "lhh"):
                    hr = lhh.hash_vector(rt.astype(np.int64))
            with L.clock(j, "member", "control_post"):
                payload = HDR.pack(j, t, a) + cj + hr + sha256(rt.tobytes())
                sig = self.users[j].sig.sign(payload)
            bb.post(j, "member", "mcontrol", t, a, payload, sig)
            L.send(j, "member", "S", "server", "server", "subsum", HDR_LEN + 4 * P)
            subs[j] = (rt, seed, cj)
        F3 = [j for j in committee if j not in subs]
        recovered_sums = {}
        if F3:
            if len(F3) > state["budget"]:
                raise _Abort("member_dropout", F3 + state["dropped"])
            for j in F3:  # Case 3 recovery: rebuild esk_j, recompute the unmasked sub-sum r^j
                esk = self._release(t, a, j, 0, active, state, member_keys)
                ka = KAKey(esk)
                if ka.pk != state["epk"][j]:
                    raise _Abort("recovery_failed", [j])
                recovered_sums[j] = L.shared(active, "member", "recover_subsum",
                                             lambda ka=ka, j=j: member_derive(ka, j, client_pk, D, t, a, P),
                                             also=[(["S"], "server")])

        # 5. server computes A and the carry vector K
        mpost = {e.author: e for e in bb.query("mcontrol", t, a)}
        bb.charge_read(["S"], "server", list(mpost.values()), "mcontrol")
        with L.clock("S", "server", "aggregate"):
            try:
                A, K = server_combine((uploads[i] for i in I_t),
                                      [subs[j][0] for j in active] + list(recovered_sums.values()),
                                      P, b.server_delta)
            except OverflowError:
                raise _Detected("carry_out_of_range")
        ak_len = HDR_LEN + 4 * P + (2 * P if p.lhh else 0)
        for j in active:
            L.send("S", "server", j, "member", "server", "AK", ak_len)

        # 6. committee checks aggregate integrity, then reveals
        bb.charge_read(active, "member", list(mpost.values()), "mcontrol")
        if p.lhh:
            hoff = HDR_LEN + 32
            rec_h = {j: L.shared(active, "member", "recover_lhh",
                                 lambda v=v: lhh.hash_vector(v.astype(np.int64)))
                     for j, v in recovered_sums.items()}
            ok = L.shared(active, "member", "lhh_check", lambda: lhh_check(
                A, K, [mpost[j].payload[hoff:hoff + lhh.POINT_LEN] for j in active] + list(rec_h.values()),
                [ctrl[i].payload[slice(*lay["h"])] for i in I_t]))
            if not ok:
                self._dispute(t, a, active, I_t, uploads, subs, ctrl, mpost, lay, b)
        revealers = [j for j in active if member_drop.get(j) != "before_reveal"]
        for j in revealers:
            seed = random_bytes(32) if j in b.member_bad_seed else subs[j][1]
            payload = HDR.pack(j, t, a) + seed
            with L.clock(j, "member", "reveal_post"):
                sig = self.users[j].sig.sign(payload)
            bb.post(j, "member", "reveal", t, a, payload, sig)
        F4 = [j for j in active if j not in revealers]
        if F4 and len(F3) + len(F4) > state["budget"]:
            raise _Abort("member_dropout", F3 + F4 + state["dropped"])
        reveals = bb.query("reveal", t, a)
        bb.charge_read(revealers, "member", reveals, "reveal")
        bb.charge_read(["S"], "server", reveals, "reveal")
        commitments = {j: mpost[j].payload[HDR_LEN:HDR_LEN + 32] for j in active}
        bad, M = L.shared(revealers, "member", "open_commitments",
                          lambda: open_commitments(reveals, commitments, self.sig_pks, t, a, P),
                          also=[(["S"], "server")])
        if bad:
            raise _Abort("bad_reveal", bad)
        for j in F4:  # Case 4 recovery: rebuild seed_j, check it against the commitment
            seed = self._release(t, a, j, 1, revealers, state, member_keys)
            if commitment(t, a, j, seed) != commitments[j]:
                raise _Abort("recovery_failed", [j])
            M += L.shared(revealers, "member", "recover_mask",
                          lambda seed=seed, j=j: prg.expand(prg.commit_mask_key(seed, t, a, j), P),
                          also=[(["S"], "server")])
        enc_sum = A + M

        # 7. co-signature on the new model, checked by every receiving client
        receivers = [i for i in online if i in uploads and i not in b.drop_after_submit]
        for i in receivers:
            L.send("S", "server", i, "client", "server", "model", 4 * P)
        if p.cosign:
            hm = model_hash(t, a, enc_sum.tobytes())
            for j in revealers:
                payload = HDR.pack(j, t, a) + hm
                with L.clock(j, "member", "cosign_post"):
                    sig = self.users[j].sig.sign(payload)
                bb.post(j, "member", "cosign", t, a, payload, sig)
            cos = bb.query("cosign", t, a)
            bb.charge_read(receivers, "client", cos, "cosign")

            def check_cosign(model_bytes):
                h = model_hash(t, a, model_bytes)
                return len(cos) == len(revealers) and all(
                    e.payload[HDR_LEN:] == h and verify(self.sig_pks[e.author], e.payload, e.sig)
                    for e in cos)

            honest_ok = L.shared([i for i in receivers if i not in b.server_tamper_model_for],
                                 "client", "check_cosign", lambda: check_cosign(enc_sum.tobytes()))
            assert honest_ok
            for i in receivers:
                if i in b.server_tamper_model_for:
                    tampered = enc_sum.copy()
                    tampered[0] ^= 1
                    with L.clock(i, "client", "check_cosign"):
                        if not check_cosign(tampered.tobytes()):
                            detections.append(f"model_tamper:{i}")

        if p.similarity:
            doff = slice(*lay["d"])
            with L.clock("S", "server", "similarity_check"):
                reported = sum(struct.unpack(">d", ctrl[i].payload[doff])[0] for i in I_t)
                actual = float(np.dot(global_model, codec.decode(enc_sum, f)))
                if not np.isclose(reported, actual, rtol=1e-9, atol=1e-6 * (1 + abs(actual))):
                    detections.append("similarity_mismatch")
        return enc_sum, I_t, F3 + F4

    def _post_delta(self, t, a, delta):
        body = (struct.pack(">IB", t, a) + self.digest + struct.pack(">I", len(delta))
                + b"".join(struct.pack(">I", i) for i in delta))
        with self.ledger.clock("S", "server", "delta_post"):
            sig = self.server_sig.sign(body)
        self.bb.post("S", "server", "delta", t, a, body, sig)

    def _dispute(self, t, a, committee, I_t, uploads, subs, ctrl, mpost, lay, b):
        """LHH check failed. An honest server identifies the parties whose posted LHH does not
        match their signed bulk vector and publishes that evidence; if there is none, the server
        itself tampered with the aggregate."""
        P, hs, hoff = self.p.length, slice(*lay["h"]), HDR_LEN + 32
        if b.server_delta is not None:  # a cheating server cannot produce valid evidence
            raise _Detected("aggregate_tamper")
        with self.ledger.clock("S", "server", "dispute"):
            bad_c = [i for i in I_t if lhh.hash_vector(uploads[i].astype(np.int64)) != ctrl[i].payload[hs]]
            bad_m = [j for j in committee
                     if lhh.hash_vector(subs[j][0].astype(np.int64)) != mpost[j].payload[hoff:hoff + lhh.POINT_LEN]]
        culprits = bad_c + bad_m
        if not culprits:
            raise _Detected("aggregate_tamper")
        for c in culprits:  # evidence: the signed bulk vector, re-hashed by every member
            self.ledger.send("S", "server", "BB", "bb", "bb", "evidence", HDR_LEN + 4 * P)
            for j in committee:
                self.ledger.send("BB", "bb", j, "member", "bb", "evidence", HDR_LEN + 4 * P)
        raise _Abort("lhh_dispute", culprits)
