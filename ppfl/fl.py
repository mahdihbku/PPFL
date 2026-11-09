"""Weighted FedAvg through the secure aggregation protocol.

Every floating-point entry of the model state (parameters and BatchNorm running statistics, which
also depend on client data) is aggregated securely. Client i submits (n_i / S) * w_i followed by
the extra coordinate n_i / S, where S is the total number of samples registered at enrollment, so
the decoded aggregate divided by its last coordinate is the FedAvg of the participating clients
even when some of them drop out. Integer buffers (BatchNorm num_batches_tracked) are copied from
one participant, as in plain FedAvg.

FixedPointFedAvg applies only the fixed-point codec (encode, sum in Z_2^32, decode). Masks cancel
exactly in Z_2^32, so its output is bit-identical to SecureFedAvg's when every client participates;
it is the cheap path for accuracy runs.
"""

import numpy as np
import torch

from . import codec
from .protocol import Behavior, Params, System


class FixedPointFedAvg:
    def __init__(self, template_state: dict, sample_counts: dict, frac_bits: int = 16):
        self.float_keys = [k for k, v in template_state.items() if v.is_floating_point()]
        self.shapes = {k: tuple(template_state[k].shape) for k in self.float_keys}
        self.sizes = {k: int(np.prod(self.shapes[k])) for k in self.float_keys}
        self.length = sum(self.sizes.values()) + 1
        self.total = float(sum(sample_counts.values()))
        self.frac_bits = frac_bits
        clients = sorted(sample_counts)
        if clients != list(range(len(clients))):
            raise ValueError("client ids must be 0..n-1")

    def flatten(self, state: dict) -> np.ndarray:
        return np.concatenate([state[k].detach().float().cpu().numpy().ravel() for k in self.float_keys])

    def unflatten(self, vec: np.ndarray, template: dict) -> dict:
        out, off = {}, 0
        for k, v in template.items():
            if k in self.sizes:
                n = self.sizes[k]
                out[k] = torch.tensor(vec[off:off + n].reshape(self.shapes[k]), dtype=v.dtype)
                off += n
            else:
                out[k] = v.detach().cpu().clone()
        return out

    def client_input(self, state: dict, n_samples: int) -> np.ndarray:
        share = n_samples / self.total
        return np.concatenate([self.flatten(state) * share, [share]])

    def finish(self, agg: np.ndarray, template: dict) -> dict:
        return self.unflatten(agg[:-1] / agg[-1], template)

    def aggregate(self, t: int, local_states: dict, sample_counts: dict):
        participants = sorted(local_states)
        enc_sum = np.zeros(self.length, dtype=np.uint64)
        for cid in participants:
            enc = codec.encode(self.client_input(local_states[cid], sample_counts[cid]), self.frac_bits)
            enc_sum = (enc_sum + enc) % codec.RING_MOD
        agg = codec.decode(enc_sum.astype(np.uint32), self.frac_bits)
        info = {"status": "ok", "attempts": 0, "participants": participants, "committee": [], "recovered": []}
        return self.finish(agg, local_states[participants[0]]), info


class SecureFedAvg(FixedPointFedAvg):
    def __init__(self, template_state: dict, sample_counts: dict, workers: int = 1, **params):
        super().__init__(template_state, sample_counts, params.get("frac_bits", 16))
        params.setdefault("L", 0)  # ignored when select_p is given
        self.system = System(Params(n_users=len(sample_counts), length=self.length, **params), workers=workers)

    def aggregate(self, t: int, local_states: dict, sample_counts: dict, behavior: Behavior | None = None):
        inputs = {cid: self.client_input(state, sample_counts[cid]) for cid, state in local_states.items()}
        res = self.system.run_round(t, inputs, behavior)
        info = {"status": res.status, "attempts": len(res.attempts), "participants": res.participants,
                "committee": res.attempts[-1].committee if res.attempts else [],
                "recovered": res.attempts[-1].recovered if res.attempts else []}
        if res.status != "ok":
            return None, info
        return self.finish(res.aggregate, local_states[res.participants[0]]), info
