import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np


class VesselGCN(torch.nn.Module):
    """GCN with skip connections for retinal vessel graph classification."""
    def __init__(self, in_dim, hid=128, out=2, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.conv3 = GCNConv(hid, hid)
        self.bn1 = torch.nn.BatchNorm1d(hid)
        self.bn2 = torch.nn.BatchNorm1d(hid)
        self.bn3 = torch.nn.BatchNorm1d(hid)
        self.fc1 = torch.nn.Linear(hid * 2, hid)
        self.fc2 = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + res  # skip connection

        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_graphs(data_path, label_path=None):
    df = pd.read_csv(data_path)
    labels = pd.read_csv(label_path) if label_path else None
    graphs = []

    for gid in df["graph_id"].unique():
        g = df[df["graph_id"] == gid].reset_index(drop=True)
        node_map = {row["node_id"]: i for i, row in g.iterrows()}

        type_enc = (g["type"] == "endpoint").astype(float).values
        x = torch.tensor(np.column_stack([
            g["x"].values / 600,
            g["y"].values / 600,
            g["width"].values / 20,
            type_enc,
        ]), dtype=torch.float)

        edges = []
        for i, row in g.iterrows():
            if pd.notna(row["edges"]) and row["edges"]:
                for t in str(row["edges"]).split(";"):
                    if t.strip().isdigit() and int(t) in node_map:
                        edges.append([i, node_map[int(t)]])
        ei = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)

        y = None
        if labels is not None:
            y = torch.tensor([labels.loc[labels["graph_id"] == gid, "label"].values[0]])

        data = Data(x=x, edge_index=ei, y=y)
        data.gid = gid
        graphs.append(data)

    return graphs


def train():
    graphs = load_graphs("data/public/train_data.csv", "data/public/train_labels.csv")

    np.random.seed(7)
    idx = np.random.permutation(len(graphs))
    split = int(0.8 * len(idx))
    train_set = [graphs[i] for i in idx[:split]]
    val_set = [graphs[i] for i in idx[split:]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    y_train = [g.y.item() for g in train_set]
    counts = np.bincount(y_train)
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float)
    weights = weights / weights.sum() * 2

    model = VesselGCN(in_dim=4, hid=128, out=2, dropout=0.4)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    best_acc, best_state, wait = 0, None, 0
    for ep in range(200):
        model.train()
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            F.cross_entropy(out, batch.y, weight=weights).backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                correct += (model(batch.x, batch.edge_index, batch.batch).argmax(1) == batch.y).sum().item()
        acc = correct / len(val_set)

        if acc > best_acc:
            best_acc, wait = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}: val_acc={acc*100:.1f}%")
        if wait >= 25:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    print(f"Best val accuracy: {best_acc*100:.1f}%")
    return model


def predict(model, data_path, out_path):
    graphs = load_graphs(data_path)
    model.eval()
    preds = []
    with torch.no_grad():
        for g in graphs:
            pred = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long)).argmax(1).item()
            preds.append({"graph_id": g.gid, "label": pred})
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    model = train()
    predict(model, "data/public/test_data.csv", "submission.csv")
