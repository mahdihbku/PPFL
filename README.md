# PPFL: Verifiable Committee-Based Secure Aggregation for Federated Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation and the evaluation code of the protocol in

> E. Bentafat, N. Lasla, A. Albaseer, Q. Malluhi, M. Abdallah,
> **"A Verifiable Committee-Based Approach for Privacy-Preserving Federated Learning"**,
> submitted to *IEEE Transactions on Network and Service Management*.

The server learns only the sum of the clients' model updates, without a trusted third party. In every round a
small committee of clients **elects itself with a Verifiable Random Function (VRF)**, so the committee is
unpredictable and cannot be chosen or suppressed by the server. The committee helps unmask the aggregate and
**verifies it before releasing anything**, which gives clients verifiability of the aggregation.

> The previous proof-of-concept (CCNC 2025 poster) is kept under the tag
> [`ccnc2025-poc`](https://github.com/mahdihbku/PPFL/tree/ccnc2025-poc).

---

## Protocol in one picture

```
                Bulletin board BB (signed, append-only, read by all parties)
          ^                         ^                             ^
          | (0) keys                | (3) dropout list Delta_t    | (1) VRF proof, epk_j
          | (2) epk_i, LH(w~_i)     |                             | (4) commitment c_j, LH(r~_j)
          |                         |                             | (6) seed_j, co-signature
   +------------+                   |                    +-----------------+
   | Users (N)  |                   |                    | Committee (C),  |
   +------------+                   |                    | elected / round |
      |      ^                 +----------+               +-----------------+
      |      +--(7) w(t+1)-----|  Server  |<------(4) r~_j-----|      ^
      +---------(2) w~_i------>|          |-------(5) (A,K)----------+
                               +----------+
```

Each round (attempt `a` of round `t`):

1. **Election.** Every user evaluates ECVRF on the public round input; users with `int(beta) < p * 2^512` post a
   signed election message (VRF proof + fresh ephemeral X25519 key) on the bulletin board. The committee is
   accepted only if it has at least `C_min` valid members; otherwise it is re-elected.
2. **Masking.** Each user masks its fixed-point update with one pseudo-random sub-mask per committee member
   (X25519 + HKDF + AES-CTR) and uploads it; it posts a linearly homomorphic hash `LH(w~_i)` on the board.
   Sub-masks are never transmitted: each member derives its own locally.
3. **Dropout list.** The server posts the list of users that did not submit; members refuse lists longer
   than `d_max`, so the server cannot isolate one update.
4. **Sub-sums.** Each member uploads the sum of its sub-masks hidden under a fresh commitment mask `m_j`.
5. **Combination.** The server computes `A = sum(w~_i) - sum(r~_j)`, which still hides the aggregate under
   `sum(m_j)`.
6. **Verification and release.** Each member checks `LH(A + 2^32 K) + sum LH(r~_j) == sum LH(w~_i)` with the
   values on the board, and only then reveals the seed of `m_j`. Everyone unmasks; members co-sign the model.
7. **Model.** Clients accept the new model only with the members' co-signatures.

**Properties.** Privacy as long as at least `tau + 1` committee members are honest; detection of any
modification of the aggregate; forward secrecy (per-round ephemeral keys); recovery of up to `tau` failed
committee members per round (members Shamir-share their round secrets); abort + cooldown + re-election beyond
that.

## What is implemented

| Module | Content |
|---|---|
| `ppfl/vrf.py` | ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381), validated against the RFC test vectors; threshold selection |
| `ppfl/sodium.py` | ctypes loader for native libsodium (VRF group operations, Ed25519); PyNaCl fallback |
| `ppfl/prg.py`, `ppfl/codec.py` | HKDF-SHA256 key derivation, AES-128-CTR masks in `Z_2^32`, fixed-point codec (16 fractional bits) |
| `ppfl/lhh.py` | Linearly homomorphic hash over secp256k1 (Pippenger multi-scalar multiplication) |
| `ppfl/bb.py`, `ppfl/ledger.py` | Bulletin board; byte ledger and per-party CPU accounting |
| `ppfl/protocol.py` | The full protocol: election, masking, dropout list, sub-sums, aggregate check, commit-reveal, co-signatures, `tau` recovery, abort/cooldown/re-election, fault and attack injection |
| `ppfl/costmodel.py` | Exact per-party byte counts, checked byte for byte against end-to-end runs |
| `ppfl/security.py` | Committee-size analysis: `C_min` over a whole training run (union bound over all elections), recovery-aware tails, choice of `p` |
| `ppfl/fl.py` | FedAvg through the protocol (weighted, including batch-norm statistics) |
| `ppfl/baselines/` | SecAgg and SecAgg+ (semi-honest, with dropout recovery) on the same primitives; Bell et al.'s parameter procedure |

Everything runs **in-process**: all parties live in one Python process, every message is serialized and its
bytes are counted per party, and CPU time is measured per party and phase. There is no network code.

## Installation

Requires Python 3.12 and (recommended) native libsodium, which is about 15x faster than PyNaCl's bundled build
for the VRF.

```bash
brew install libsodium            # macOS; on Debian/Ubuntu: sudo apt install libsodium-dev
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[test,bench]"     # add ",fl" for the accuracy experiments (torch, scikit-learn)
.venv/bin/python -m pytest tests/                        # 83 tests
```

Set `PPFL_NATIVE_SODIUM=0` to force the PyNaCl fallback.

## Quick start

```python
import numpy as np
from ppfl import security
from ppfl.protocol import Behavior, Params, System

N, P = 200, 1000                                      # users, model size
c_min = security.c_min_for_tau(40, 0.10, 2, 4000)     # sigma=40, T/N=10%, tau=2, 4000 elections -> 20
p = security.choose_p(N, int(0.10 * N), c_min, 0.10)  # re-election <= 10% even if corrupted users withhold
system = System(Params(n_users=N, length=P, L=0, c_min=c_min, tau=2, select_p=p))

rng = np.random.default_rng(0)
updates = {i: rng.normal(0, 0.1, P) for i in range(N)}
result = system.run_round(1, updates, Behavior(drop_before_submit={3, 17}))
print(result.status, len(result.attempts[-1].committee), len(result.participants))   # ok 35 198
```

`Behavior` injects faults and attacks: dropped users and members (`member_drop`), a server that adds a delta
to the aggregate (`server_delta`), equivocates on the dropout list, tampers with the model, or falsely
declares users dropped (`server_exclude`), wrong hashes or seeds. See `tests/test_protocol.py` and
`tests/test_recovery.py`.

## Reproducing the paper's results

All scripts write to a new directory under `results/`. Default configuration: `T/N = 10%`, `sigma = 40`,
`E = 4000` elections, `tau = 2`, hence `C_min = 20` and an expected committee size of about 29;
`P = 64,811` unless stated otherwise.

| Result | Command |
|---|---|
| Security numerics, `C_min` table (Sec. VI) | `python bench/security_numerics.py --out results/<dir>` |
| Committee size with real ECVRF outputs (Fig. 4) | `python bench/fig4_threshold.py --rho 0.10 --out results/<dir>` |
| Communication grid (Fig. 8) | `python bench/comm_grid.py --tau 2 --out results/<dir>` |
| Per-role computation (Fig. 7) | `python bench/role_timing.py --roles vrf client member server --out results/<dir>` |
| SecAgg / SecAgg+ timing (Fig. 9) | `python bench/baseline_timing.py --out results/<dir>` and `--bell-gamma 0.1 --bell-deltas 0.05 0.1` |
| Recovery cost | `python bench/recovery_timing.py --out results/<dir>` |
| Liveness under member dropout (Fig. 10) | `python bench/liveness.py --timing results/<role-timing-dir> --out results/<dir>` |
| End-to-end byte check | `python bench/e2e_validate.py --n 1000 --c-min 10 --workers 10 --timing results/<role-timing-dir> --out results/<dir>` |
| Accuracy (Fig. 6) | `python accuracy/run_all.py --out results/<dir>` |
| All figures | `python figures/make_figures.py --out results/<dir>` |

**Timing methodology.** On machines with performance and efficiency cores, the same benchmark can differ by
2-3x between processes depending on core placement and background load. We run every timing benchmark in the
foreground, in 3-5 independent processes, and keep per configuration the minimum of the per-process medians
(`bench/combine_runs.py`); the same procedure is applied to all schemes. The paper's numbers were measured on
an Apple M3 Pro, single-threaded.

**Figures.** `figures/make_figures.py` reads the result directories named at the top of the script; point
them to your output directories.

**Flamingo baseline.** Flamingo is measured with its authors' implementation:

```bash
git clone https://github.com/cslyc/flamingo ../flamingo-ref && cd ../flamingo-ref
git checkout dccbfc4 && git apply ../PPFL/bench/flamingo_instrumentation.patch
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt   # see the patch notes below
cd pki_files && FLAMINGO_NUM_KEYS=2048 ../.venv/bin/python setup_pki.py && cd ../../PPFL
python bench/flamingo_timing.py --out results/<dir> --ns 128 256 512 1024 2048 --runs 3 --bytes
python bench/flamingo_summary.py results/<dir>
```

The patch only adds measurement hooks (per-agent timers, message sizes) and parameter plumbing (vector length,
simulated waiting times so that every client and decryptor participates, 2048 keys); it does not change the
protocol. Set `FLAMINGO_REF` if the clone is elsewhere. Some of Flamingo's pinned plotting packages do not
build on recent systems; the protocol only needs `numpy`, `pandas`, `dill`, `libnum`, `pynacl==1.5.0`, and
`pycryptodomex==3.17`.

**Datasets for the accuracy experiments.** Place the data under `data/` (or set `PPFL_DATA`):

- `data/nsl-kdd/KDDTrain+.txt`, `data/nsl-kdd/KDDTest+.txt`: [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html);
- `data/datasets/data_without_ioa.csv`: the IEC 60870-5-104 smart-grid dataset of Teryak et al. (2023).

The training code (model, splits, local training, plain FedAvg) is in `accuracy/fl_helpers.py`; only the
aggregation differs between the FedAvg and the secure runs.

## Repository layout

```
ppfl/          protocol, primitives, cost model, security analysis, baselines
tests/         unit and end-to-end tests (pytest)
bench/         benchmarks and simulations behind the paper's figures
accuracy/      FedAvg vs. secure aggregation on NSL-KDD and the SG dataset
figures/       figure generation (matplotlib)
```

## Notes

This is research code: it has not been audited and is not intended for production use. The bulletin board is
modeled as a consistent append-only log; its replication is not implemented.

## License

MIT, see [LICENSE](LICENSE).
