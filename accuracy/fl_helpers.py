"""Training helpers for the accuracy experiments (Fig. 6), copied verbatim from the authors' FedAvg
code: data loading, client splits, local training, plain FedAvg, and the Classifier1 model."""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


class Classifier1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier1, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.layer5 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        # return self.softmax(x)
        return x


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_on_client(client_tensors, model, epochs, optimizer, criterion, device):
    X_client_tensor, y_client_tensor = client_tensors
    assert X_client_tensor.device.type == device.type and y_client_tensor.device.type == device.type

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_client_tensor)
        loss = criterion(outputs, y_client_tensor)
        loss.backward()
        optimizer.step()
    return model


def clone_state_dict(state_dict):
    return {key: value.detach().clone() for key, value in state_dict.items()}


def federated_avg(client_sample_counts, model_state_dicts):
    if not model_state_dicts:
        raise ValueError("Cannot aggregate empty client state dicts.")

    first_client_id = next(iter(model_state_dicts))
    first_state = model_state_dicts[first_client_id]
    total_data_points = sum(client_sample_counts[cid] for cid in model_state_dicts.keys())
    if total_data_points <= 0:
        raise ValueError("Total aggregated samples must be > 0.")

    global_state = {}
    for key, first_value in first_state.items():
        if first_value.is_floating_point():
            weighted_sum = torch.zeros_like(first_value, dtype=torch.float32)
            for client_id, client_state in model_state_dicts.items():
                weight = float(client_sample_counts[client_id]) / float(total_data_points)
                weighted_sum.add_(client_state[key].float() * weight)
            global_state[key] = weighted_sum.to(dtype=first_value.dtype)
        else:
            global_state[key] = first_value.clone()
    return global_state


def make_iid_splits(y_train, num_clients, seed):
    """IID-like split: shuffle indices and split evenly."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    rng.shuffle(indices)

    split_size = len(indices) // num_clients
    client_indices = {}
    for i in range(num_clients):
        start = i * split_size
        end = len(indices) if i == num_clients - 1 else (i + 1) * split_size
        client_indices[i] = indices[start:end]
    return client_indices


def make_dirichlet_splits(y_train, num_clients, alpha, seed):
    """
    Standard label-based Dirichlet non-IID split.
    For each class, distribute its examples among clients using Dirichlet(alpha).
    """
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)

    client_indices = {i: [] for i in range(num_clients)}
    for c in classes:
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet([alpha] * num_clients)
        # Convert proportions -> counts
        counts = (proportions * len(idx_c)).astype(int)

        # Fix rounding so total counts match
        diff = len(idx_c) - counts.sum()
        if diff > 0:
            counts[np.argmax(counts)] += diff
        elif diff < 0:
            # remove extras from the largest bins
            for _ in range(-diff):
                j = np.argmax(counts)
                if counts[j] > 0:
                    counts[j] -= 1

        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(idx_c[start:end].tolist())
            start = end

    # Shuffle within each client for good measure
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
        client_indices[cid] = np.array(client_indices[cid], dtype=int)

    return client_indices


def build_client_tensors(X_train, y_train, client_indices, device):
    client_data = {}
    client_sample_counts = {}
    for cid, idx in client_indices.items():
        X_client_tensor = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
        y_client_tensor = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        client_data[cid] = (X_client_tensor, y_client_tensor)
        client_sample_counts[cid] = len(idx)
    return client_data, client_sample_counts


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_nsl_kdd(train_path, test_path, label_mode, include_difficulty):
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS).dropna()
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS).dropna()

    feature_cols = NSL_KDD_COLUMNS[:-2]
    if include_difficulty:
        feature_cols = feature_cols + ["difficulty"]

    categorical_cols = [col for col in ["protocol_type", "service", "flag"] if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ]
    )

    X_train = np.asarray(preprocessor.fit_transform(train_df[feature_cols]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_df[feature_cols]), dtype=np.float32)

    y_train_raw = train_df["label"].astype(str).str.strip()
    y_test_raw = test_df["label"].astype(str).str.strip()

    if label_mode == "binary":
        y_train = (y_train_raw != "normal").astype(np.int64).to_numpy()
        y_test = (y_test_raw != "normal").astype(np.int64).to_numpy()
        label_names = np.array(["normal", "attack"])
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([y_train_raw, y_test_raw], axis=0, ignore_index=True))
        y_train = label_encoder.transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        label_names = label_encoder.classes_

    return X_train, X_test, y_train, y_test, label_names
