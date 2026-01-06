import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os

torch.manual_seed(42)
np.random.seed(42)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return h

def load_data():
    nodes_df = pd.read_csv('../data/nodes.csv')
    features = torch.FloatTensor(nodes_df.drop('node_id', axis=1).values)
    
    edges_df = pd.read_csv('../data/edges.csv')
    g = dgl.graph((edges_df['source'].values, edges_df['target'].values))
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    
    labels = torch.full((len(nodes_df),), -1, dtype=torch.long)
    labels[train_df['node_id'].values] = torch.LongTensor(train_df['target'].values)
    
    train_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    train_mask[train_df['node_id'].values] = True
    
    return g, features, labels, train_mask, test_df['node_id'].values

def train(model, g, features, labels, train_mask, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(g, features)[train_mask].argmax(dim=1)
                f1 = f1_score(labels[train_mask].numpy(), pred.numpy(), average='macro')
                print(f'Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train F1: {f1:.4f}')

def main():
    print("Loading data...")
    g, features, labels, train_mask, test_nodes = load_data()
    print(f"Nodes: {g.num_nodes()}, Edges: {g.num_edges()}, Train: {train_mask.sum().item()}")
    
    model = GCN(features.shape[1], 64, 7, dropout=0.5)
    
    print("\nTraining...")
    train(model, g, features, labels, train_mask)
    
    print("\nPredicting...")
    model.eval()
    with torch.no_grad():
        preds = model(g, features)[test_nodes].argmax(dim=1).numpy()
    
    submission = pd.DataFrame({'node_id': test_nodes, 'target': preds})
    submission.to_csv('../submissions/baseline_submission.csv', index=False)
    print(f"Saved to submissions/baseline_submission.csv")

if __name__ == "__main__":
    main()
