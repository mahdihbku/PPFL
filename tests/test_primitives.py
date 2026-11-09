import numpy as np
import pytest

from ppfl import codec, lhh, prg, vrf
from ppfl.keys import KAKey, SigningKey, verify

# RFC 9381, Appendix B.3 (ECVRF-EDWARDS25519-SHA512-TAI), Examples 16-18.
RFC9381_TAI = [
    dict(
        sk="9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        pk="d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        alpha="",
        H="91bbed02a99461df1ad4c6564a5f5d829d0b90cfc7903e7a5797bd658abf3318",
        pi="8657106690b5526245a92b003bb079ccd1a92130477671f6fc01ad16f26f723f26f8a57ccaed74ee1b190bed1f479d9727d2d0f9b005a6e456a35d4fb0daab1268a1b0db10836d9826a528ca76567805",
        beta="90cf1df3b703cce59e2a35b925d411164068269d7b2d29f3301c03dd757876ff66b71dda49d2de59d03450451af026798e8f81cd2e333de5cdf4f3e140fdd8ae",
    ),
    dict(
        sk="4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        pk="3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        alpha="72",
        H="5b659fc3d4e9263fd9a4ed1d022d75eaacc20df5e09f9ea937502396598dc551",
        pi="f3141cd382dc42909d19ec5110469e4feae18300e94f304590abdced48aed5933bf0864a62558b3ed7f2fea45c92a465301b3bbf5e3e54ddf2d935be3b67926da3ef39226bbc355bdc9850112c8f4b02",
        beta="eb4440665d3891d668e7e0fcaf587f1b4bd7fbfe99d0eb2211ccec90496310eb5e33821bc613efb94db5e5b54c70a848a0bef4553a41befc57663b56373a5031",
    ),
    dict(
        sk="c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        pk="fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        alpha="af82",
        H="bf4339376f5542811de615e3313d2b36f6f53c0acfebb482159711201192576a",
        pi="9bc0f79119cc5604bf02d23b4caede71393cedfbb191434dd016d30177ccbf8096bb474e53895c362d8628ee9f9ea3c0e52c7a5c691b6c18c9979866568add7a2d41b00b05081ed0f58ee5e31b3a970e",
        beta="645427e5d00c62a23fb703732fa5d892940935942101e456ecca7bb217c61c452118fec1219202a0edcf038bb6373241578be7217ba85a2687f7a0310b2df19f",
    ),
]


@pytest.mark.parametrize("tv", RFC9381_TAI)
def test_vrf_rfc9381_vectors(tv):
    sk, alpha = bytes.fromhex(tv["sk"]), bytes.fromhex(tv["alpha"])
    _, pk = vrf.keygen(sk)
    assert pk.hex() == tv["pk"]
    assert vrf.encode_to_curve(pk, alpha).hex() == tv["H"]
    pi = vrf.prove(sk, alpha)
    assert pi.hex() == tv["pi"]
    assert vrf.proof_to_hash(pi).hex() == tv["beta"]
    assert vrf.verify(pk, alpha, pi).hex() == tv["beta"]


def test_vrf_rejects_tampering():
    sk, pk = vrf.keygen()
    pi = vrf.prove(sk, b"round-1")
    assert vrf.verify(pk, b"round-1", pi) is not None
    assert vrf.verify(pk, b"round-2", pi) is None
    bad = bytearray(pi)
    bad[40] ^= 1
    assert vrf.verify(pk, b"round-1", bytes(bad)) is None
    _, other_pk = vrf.keygen()
    assert vrf.verify(other_pk, b"round-1", pi) is None


def test_selection_rule_at_least_L():
    assert vrf.selected(bytes([0x0F]) + bytes(63), 4)  # exactly 4 leading zeros
    assert not vrf.selected(bytes([0x0F]) + bytes(63), 5)


def test_codec_roundtrip_and_sum():
    rng = np.random.default_rng(0)
    xs = [rng.normal(0, 1, 1000) for _ in range(20)]
    f = 16
    enc_sum = np.zeros(1000, dtype=np.uint32)
    for x in xs:
        enc_sum += codec.encode(x, f)
    err = np.abs(codec.decode(enc_sum, f) - np.sum(xs, axis=0))
    assert err.max() <= 20 * 2 ** -(f + 1) + 1e-12
    with pytest.raises(OverflowError):
        codec.encode(np.array([1e6]), f)


def test_prg_deterministic_and_distinct():
    k1 = prg.mask_key(b"s" * 32, b"D" * 32, t=1, attempt=0, i=2, j=3)
    k2 = prg.mask_key(b"s" * 32, b"D" * 32, t=2, attempt=0, i=2, j=3)
    a, b = prg.expand(k1, 5000), prg.expand(k1, 5000)
    assert a.dtype == np.uint32 and np.array_equal(a, b)
    assert not np.array_equal(a, prg.expand(k2, 5000))


def test_x25519_and_signatures():
    a, b = KAKey(), KAKey()
    assert a.agree(b.pk) == b.agree(a.pk)
    s = SigningKey()
    sig = s.sign(b"msg")
    assert verify(s.pk, b"msg", sig) and not verify(s.pk, b"msg2", sig)


def test_lhh_homomorphic_and_binding_smoke():
    rng = np.random.default_rng(1)
    P = 1003  # not a multiple of the lane count
    vs = [rng.integers(0, 2**32, P, dtype=np.int64) for _ in range(4)]
    total = np.sum(vs, axis=0)
    assert lhh.combine([lhh.hash_vector(v) for v in vs]) == lhh.hash_vector(total)
    tampered = total.copy()
    tampered[17] += 1
    assert lhh.hash_vector(tampered) != lhh.hash_vector(total)
    # signed lanes: h(a) + h(-a) = identity
    assert lhh.combine([lhh.hash_vector(vs[0]), lhh.hash_vector(-vs[0])]) == lhh.IDENTITY
    with pytest.raises(ValueError):
        lhh.pack(np.array([2**49]))


def test_lhh_bucket_msm_matches_reference():
    rng = np.random.default_rng(5)
    for v in (rng.integers(0, 2**32, 4001, dtype=np.int64),
              rng.integers(0, 2**32, 997, dtype=np.int64) - (rng.integers(-30, 3000, 997) << 32),
              np.zeros(50, dtype=np.int64),
              np.concatenate([np.zeros(500, dtype=np.int64), rng.integers(0, 9, 13)])):
        assert lhh.hash_vector_pippenger(v) == lhh.hash_vector_reference(v)


def test_vrf_output_shortcut_matches_proof():
    sk, _ = vrf.keygen()
    for alpha in (b"", b"a", b"round-7"):
        assert vrf.output(sk, alpha) == vrf.proof_to_hash(vrf.prove(sk, alpha))


def test_threshold_rule_generalises_leading_zeros():
    import os
    for _ in range(2000):
        beta = os.urandom(64)
        for L in (0, 1, 3, 8):
            assert vrf.selected(beta, L) == vrf.below(beta, vrf.threshold_for_L(L)) == vrf.below(beta, vrf.threshold(2.0 ** -L))
    assert vrf.below(bytes(64), vrf.threshold(1e-9)) and not vrf.below(b"\xff" * 64, vrf.threshold(0.999))
