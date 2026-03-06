import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


DATA_DIR = os.path.join("gnn_challenge", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
print("Device:", DEVICE)


def build_node_features(G: nx.Graph):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return None

    xs = np.array([G.nodes[n].get("x", 0.0) for n in nodes], dtype=np.float32)
    ys = np.array([G.nodes[n].get("y", 0.0) for n in nodes], dtype=np.float32)

    xs = xs - xs.mean()
    ys = ys - ys.mean()

    scale = float(np.sqrt(xs.var() + ys.var()) + 1e-6)
    xs = xs / scale
    ys = ys / scale

    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)

    return np.stack([xs, ys, deg], axis=1)


def coalesce_undirected_edges(G: nx.Graph, mapping: dict):
    edges = []
    for u, v in G.edges():
        if u in mapping and v in mapping:
            iu, iv = mapping[u], mapping[v]
            edges.append((iu, iv))
            edges.append((iv, iu))
    return edges


def normalize_adj_sparse(indices, values, n):
    indices = indices.long()
    values = values.float()

    row = indices[0]
    deg = torch.zeros(n, device=values.device).scatter_add_(0, row, values)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

    col = indices[1]
    norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(indices, norm_values, (n, n)).coalesce()


def graph_to_tensors(G, device):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return None, None

    mapping = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    X_np = build_node_features(G)
    if X_np is None:
        return None, None

    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    edges = coalesce_undirected_edges(G, mapping)
    edges += [(i, i) for i in range(n)]

    if len(edges) == 0:
        return None, None

    indices = torch.tensor(edges, dtype=torch.long, device=device).t()
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    A_norm = normalize_adj_sparse(adj.indices(), adj.values(), n)

    return X, A_norm


def load_train_data(train_dir, labels_csv, device):

    labels_df = pd.read_csv(labels_csv)
    label_map = dict(zip(labels_df["filename"], labels_df["target"]))

    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pkl")])

    graphs, y = [], []

    for fn in files:

        if fn not in label_map:
            continue

        with open(os.path.join(train_dir, fn), "rb") as f:
            G = pickle.load(f)

        X, A = graph_to_tensors(G, device)

        if X is None:
            continue

        graphs.append((fn, X, A))
        y.append(int(label_map[fn]))

    return graphs, torch.tensor(y, dtype=torch.long, device=device)


def load_test_data(test_dir, device):

    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pkl")])

    graphs = []

    for fn in files:

        with open(os.path.join(test_dir, fn), "rb") as f:
            G = pickle.load(f)

        X, A = graph_to_tensors(G, device)

        if X is None:
            continue

        graphs.append((fn, X, A))

    return graphs


train_graphs, y_all = load_train_data(TRAIN_DIR, TRAIN_LABELS_CSV, DEVICE)
test_graphs = load_test_data(TEST_DIR, DEVICE)

print("Loaded train graphs:", len(train_graphs))
print("Loaded test graphs:", len(test_graphs))


labels_cpu = y_all.detach().cpu().numpy().tolist()

idx_all = list(range(len(labels_cpu)))

idx_tr, idx_val = train_test_split(
    idx_all,
    test_size=0.20,
    stratify=labels_cpu,
    random_state=SEED
)

print("Train:", len(idx_tr), "Val:", len(idx_val))


counts = np.bincount(np.array(labels_cpu), minlength=3)

weights = (counts.sum() / (counts + 1e-6))
weights = weights / weights.mean()

class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)


class SimpleBetterGCN(nn.Module):

    def __init__(self, in_dim=3, hidden=64, num_classes=3):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.att = nn.Linear(hidden, 1)

        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, x, adj):

        h1 = torch.spmm(adj, self.fc1(x))
        h1 = F.relu(h1)

        h2 = torch.spmm(adj, self.fc2(h1))
        h2 = F.relu(h2)

        h = h1 + h2

        weights = torch.softmax(self.att(h), dim=0)

        g = torch.sum(weights * h, dim=0)

        return self.cls(g)


model = SimpleBetterGCN().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss(weight=class_weights)


def eval_split(indices):

    model.eval()

    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():

        for j in indices:

            _, X, A = train_graphs[j]

            logits = model(X, A).unsqueeze(0)

            target = y_all[j].unsqueeze(0)

            loss = criterion(logits, target)

            total_loss += float(loss.item())

            pred = int(torch.argmax(logits, dim=1).item())

            y_true.append(int(target.item()))
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average="macro")

    return total_loss / max(1, len(indices)), acc, f1


best_f1 = -1.0
best_state = None

patience = 20
bad = 0

max_epochs = 400

print("\nTraining...")

for epoch in range(1, max_epochs + 1):

    model.train()

    total_loss = 0.0

    perm = np.random.permutation(idx_tr)

    for j in perm:

        _, X, A = train_graphs[j]

        target = y_all[j].unsqueeze(0)

        optimizer.zero_grad()

        logits = model(X, A).unsqueeze(0)

        loss = criterion(logits, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total_loss += float(loss.item())

    if epoch % 10 == 0:

        val_loss, val_acc, val_f1 = eval_split(idx_val)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={total_loss/len(idx_tr):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | "
            f"val_f1={val_f1:.3f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            bad = 0

        else:

            bad += 1

            if bad >= patience:
                break


if best_state is not None:
    model.load_state_dict(best_state)


val_loss, val_acc, val_f1 = eval_split(idx_val)

print("\nBest validation performance")
print("Accuracy:", val_acc)
print("Macro F1:", val_f1)


model.eval()

pred_rows = []

with torch.no_grad():

    for fn, X, A in test_graphs:

        logits = model(X, A).unsqueeze(0)

        pred = int(torch.argmax(logits, dim=1).item())

        pred_rows.append({
            "filename": fn,
            "prediction": pred
        })


submission = pd.DataFrame(pred_rows).sort_values("filename")

out_path = os.path.join(DATA_DIR, "submission.csv")

submission.to_csv(out_path, index=False)

print("\nSubmission saved to:", out_path)

print(submission.head())
