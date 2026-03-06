import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --------------------------------------------------
# PATH SETUP (robust)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "gnn_challenge", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

# --------------------------------------------------
# GRAPH LOADER
# --------------------------------------------------

def load_graph(path):

    with open(path, "rb") as f:
        g = pickle.load(f)

    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)

    x = torch.tensor(g["node_feat"], dtype=torch.float)

    # add degree feature (improves performance)
    deg = torch.bincount(edge_index[0], minlength=x.shape[0]).float().unsqueeze(1)

    x = torch.cat([x, deg], dim=1)

    return Data(x=x, edge_index=edge_index)

# --------------------------------------------------
# LOAD TRAIN DATA
# --------------------------------------------------

def load_train_data():

    labels_df = pd.read_csv(TRAIN_LABELS_CSV)

    graphs = []
    labels = []

    for _, row in labels_df.iterrows():

        graph_path = os.path.join(TRAIN_DIR, row["graph_id"] + ".pkl")

        g = load_graph(graph_path)

        graphs.append(g)
        labels.append(row["label"])

    y = torch.tensor(labels)

    return graphs, y

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

def load_test_data():

    graphs = []
    graph_ids = []

    for file in os.listdir(TEST_DIR):

        if file.endswith(".pkl"):

            path = os.path.join(TEST_DIR, file)

            g = load_graph(path)

            graphs.append(g)
            graph_ids.append(file.replace(".pkl",""))

    return graphs, graph_ids

# --------------------------------------------------
# MODEL
# --------------------------------------------------

class SimpleGCN(nn.Module):

    def __init__(self, in_channels, hidden=64):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        self.lin = nn.Linear(hidden, 2)

        self.dropout = 0.3

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return x

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

def train(model, loader, optimizer):

    model.train()

    total_loss = 0

    for batch in loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        loss = F.cross_entropy(out, batch.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading training data...")

    train_graphs, y = load_train_data()

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor([y[i]])

    loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

    in_channels = train_graphs[0].x.shape[1]

    model = SimpleGCN(in_channels).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")

    for epoch in range(30):

        loss = train(model, loader, optimizer)

        print(f"Epoch {epoch+1} | Loss {loss:.4f}")

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    print("Loading test data...")

    test_graphs, graph_ids = load_test_data()

    test_loader = DataLoader(test_graphs, batch_size=32)

    model.eval()

    preds = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(DEVICE)

            out = model(batch.x, batch.edge_index, batch.batch)

            p = out.argmax(dim=1).cpu().numpy()

            preds.extend(p)

    submission = pd.DataFrame({
        "graph_id": graph_ids,
        "label": preds
    })

    submission.to_csv("submission.csv", index=False)

    print("Submission file saved!")
