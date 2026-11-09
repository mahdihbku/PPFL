"""Fig. 6 runs: FedAvg vs the secure protocol on the SG dataset and NSL-KDD.

Data loading, client splits, model (Classifier1) and local training are imported unchanged from the
original accuracy code (PP-FL/Code/Codes SG Datasets); only aggregation differs. In secure mode
every round runs the full protocol (election, masking, committee sub-sums with tau-recovery, LHH
check, commit-reveal, co-signatures) and the per-round deviation from plain FedAvg of the same
local models is logged. Codec mode applies only the fixed-point codec (encode, sum in Z_2^32,
decode), which is bit-identical to the protocol output (tests/test_fl.py). In every mode the
accuracy of plain FedAvg of the same local models is also logged (accuracy_fedavg), so a single
codec or secure run gives both curves from one training trajectory.

  .venv/bin/python accuracy/run_fl.py --dataset nsl_binary --split iid --mode secure --rounds 200 \
      --out results/<dir>/nsl_binary_iid_secure
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Datasets: <data>/datasets/data_without_ioa.csv (SG) and <data>/nsl-kdd/KDDTrain+.txt, KDDTest+.txt
DATA = Path(os.environ.get("PPFL_DATA", Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_helpers as orig  # noqa: E402  (training helpers copied from the original FedAvg code)
from fl_helpers import Classifier1  # noqa: E402

from ppfl.fl import FixedPointFedAvg, SecureFedAvg  # noqa: E402


def load_sg(seed):
    data = pd.read_csv(DATA / "datasets/data_without_ioa.csv").dropna()
    X = data.drop("label", axis=1)
    y = LabelEncoder().fit_transform(data["label"])
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test, len(np.unique(y))


def load(dataset, seed):
    if dataset == "sg":
        return load_sg(seed)
    mode = "binary" if dataset == "nsl_binary" else "multiclass"
    X_train, X_test, y_train, y_test, names = orig.load_nsl_kdd(
        DATA / "nsl-kdd/KDDTrain+.txt", DATA / "nsl-kdd/KDDTest+.txt", mode, False)
    return X_train, X_test, y_train, y_test, len(names)


def max_dev(a, b):
    return max(float((a[k].float() - b[k].float().cpu()).abs().max()) for k in a if a[k].is_floating_point())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sg", "nsl_binary", "nsl_multiclass"], required=True)
    ap.add_argument("--split", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--mode", choices=["fedavg", "secure", "codec"], required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--no-lhh", action="store_true")
    ap.add_argument("--frac-bits", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    device = orig.get_device()
    orig.set_seed(args.seed)
    X_train, X_test, y_train, y_test, n_classes = load(args.dataset, args.seed)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)
    if args.split == "iid":
        idx = orig.make_iid_splits(y_train, args.num_clients, seed=args.seed)
    else:
        idx = orig.make_dirichlet_splits(y_train, args.num_clients, alpha=args.alpha, seed=args.seed)
    client_data, counts = orig.build_client_tensors(X_train, y_train, idx, device=device)
    global_model = Classifier1(X_train.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    eval_model = Classifier1(X_train.shape[1], n_classes).to(device)
    secure = None
    if args.mode == "secure":
        secure = SecureFedAvg(global_model.state_dict(), counts, select_p=1.0, c_min=args.tau + 1,
                              tau=args.tau, lhh=not args.no_lhh, frac_bits=args.frac_bits)
    elif args.mode == "codec":
        secure = FixedPointFedAvg(global_model.state_dict(), counts, frac_bits=args.frac_bits)
    n_params = sum(p.numel() for p in global_model.parameters())
    print(f"{args.dataset} {args.split} {args.mode}: params={n_params} "
          f"aggregated_len={secure.length if secure else '-'} device={device}", flush=True)

    with open(args.out / "metrics.csv", "w") as mf:
        mf.write("round,accuracy,local_s,agg_s,eval_s,max_abs_dev_vs_fedavg,attempts,committee,accuracy_fedavg\n")
    for r in range(args.rounds):
        base = orig.clone_state_dict(global_model.state_dict())
        t0 = time.perf_counter()
        local = {}
        for cid, tensors in client_data.items():
            m = Classifier1(X_train.shape[1], n_classes).to(device)
            m.load_state_dict(base)
            opt = optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            orig.train_on_client(tensors, m, args.epochs, opt, criterion, device)
            local[cid] = orig.clone_state_dict(m.state_dict())
        orig.synchronize_device(device)
        t1 = time.perf_counter()
        plain = orig.federated_avg(counts, local)
        dev, attempts, committee = 0.0, 0, 0
        if secure is None:
            new_state = plain
        else:
            cpu_local = {c: {k: v.cpu() for k, v in s.items()} for c, s in local.items()}
            new_state, info = secure.aggregate(r + 1, cpu_local, counts)
            if new_state is None:
                raise RuntimeError(f"round {r + 1} failed: {info}")
            dev = max_dev(new_state, plain)
            attempts, committee = info["attempts"], len(info["committee"])
        t2 = time.perf_counter()
        global_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})
        global_model.eval()
        with torch.inference_mode():
            acc = (global_model(X_test_t).argmax(1) == y_test_t).float().mean().item()
        t3 = time.perf_counter()
        if secure is None:
            acc_fedavg = acc
        else:
            eval_model.load_state_dict({k: v.to(device) for k, v in plain.items()})
            eval_model.eval()
            with torch.inference_mode():
                acc_fedavg = (eval_model(X_test_t).argmax(1) == y_test_t).float().mean().item()
        with open(args.out / "metrics.csv", "a") as mf:
            mf.write(f"{r + 1},{acc:.10g},{t1 - t0:.4f},{t2 - t1:.4f},{t3 - t2:.4f},{dev:.3e},{attempts},{committee},"
                     f"{acc_fedavg:.10g}\n")
        print(f"round {r + 1}: acc={acc:.4f} acc_fedavg={acc_fedavg:.4f} agg={t2 - t1:.2f}s dev={dev:.2e}", flush=True)


if __name__ == "__main__":
    main()
