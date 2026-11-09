import numpy as np
import torch
import torch.nn as nn

from ppfl.fl import FixedPointFedAvg, SecureFedAvg
from ppfl.protocol import Behavior


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 16)
        self.bn = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 3)


def plain_fedavg(states, counts):
    total = sum(counts[c] for c in states)
    out = {}
    first = next(iter(states.values()))
    for k, v in first.items():
        if v.is_floating_point():
            out[k] = sum(states[c][k].float() * (counts[c] / total) for c in states)
        else:
            out[k] = v.clone()
    return out


def make_states(n, seed=0):
    torch.manual_seed(seed)
    states = {}
    for c in range(n):
        m = Net()
        with torch.no_grad():
            m.bn.running_mean.normal_(0, 1)
            m.bn.running_var.uniform_(0.5, 3.0)
            m.bn.num_batches_tracked.fill_(7)
        states[c] = {k: v.clone() for k, v in m.state_dict().items()}
    return states


def test_secure_fedavg_matches_plain_within_quantisation():
    n = 5
    states = make_states(n)
    counts = {c: 100 + 37 * c for c in range(n)}
    agg = SecureFedAvg(states[0], counts, select_p=1.0, c_min=3, tau=2)
    out, info = agg.aggregate(1, states, counts)
    ref = plain_fedavg(states, counts)
    assert info["status"] == "ok"
    for k, v in ref.items():
        if v.is_floating_point():
            assert torch.allclose(out[k], v, atol=n * 2 ** -16 / min(counts.values()) * sum(counts.values()))
        else:
            assert torch.equal(out[k], v)


def test_secure_fedavg_weights_only_participants():
    n = 5
    states = make_states(n, seed=1)
    counts = {c: 50 * (c + 1) for c in range(n)}
    agg = SecureFedAvg(states[0], counts, select_p=1.0, c_min=3, tau=2)
    out, info = agg.aggregate(1, states, counts, Behavior(drop_before_submit={4}))
    ref = plain_fedavg({c: s for c, s in states.items() if c != 4}, counts)
    assert info["participants"] == [0, 1, 2, 3]
    assert torch.allclose(out["fc.weight"], ref["fc.weight"], atol=1e-3)


def test_codec_only_is_bit_identical_to_protocol():
    """Masks cancel exactly in Z_2^32, so the codec alone reproduces the protocol output."""
    n = 5
    counts = {c: 1000 * (c + 1) ** 2 for c in range(n)}
    secure = SecureFedAvg(make_states(1)[0], counts, select_p=1.0, c_min=3, tau=2)
    codec_only = FixedPointFedAvg(make_states(1)[0], counts)
    for t in (1, 2):
        states = make_states(n, seed=10 + t)
        a, info = secure.aggregate(t, states, counts)
        b, _ = codec_only.aggregate(t, states, counts)
        assert info["status"] == "ok"
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])
