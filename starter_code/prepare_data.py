import urllib.request
import tarfile
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

def download_cora():
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    if not os.path.exists("cora.tgz"):
        print("Downloading Cora...")
        urllib.request.urlretrieve(url, "cora.tgz")
    with tarfile.open("cora.tgz", 'r:gz') as tar:
        tar.extractall()

def prepare_data():
    if not os.path.exists('cora'):
        download_cora()
    
    # Load nodes
    node_ids, features, labels = [], [], []
    label_map = {}
    
    with open('cora/cora.content', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_ids.append(parts[0])
            features.append([int(x) for x in parts[1:-1]])
            label = parts[-1]
            if label not in label_map:
                label_map[label] = len(label_map)
            labels.append(label_map[label])
    
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    features = np.array(features)
    labels = np.array(labels)
    
    # Load edges
    edges_src, edges_dst = [], []
    with open('cora/cora.cites', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0] in node_to_idx and parts[1] in node_to_idx:
                edges_src.append(node_to_idx[parts[0]])
                edges_dst.append(node_to_idx[parts[1]])
    
    print(f"Nodes: {len(node_ids)}, Edges: {len(edges_src)}, Features: {features.shape[1]}, Classes: {len(label_map)}")
    
    # 80/20 train/test split (stratified)
    all_data = pd.DataFrame({'node_id': range(len(labels)), 'target': labels})
    train_df, test_df = train_test_split(all_data, test_size=0.2, stratify=all_data['target'], random_state=42)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Save files
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../submissions', exist_ok=True)
    
    # nodes.csv
    nodes_df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
    nodes_df.insert(0, 'node_id', range(len(features)))
    nodes_df.to_csv('../data/nodes.csv', index=False)
    
    # edges.csv
    pd.DataFrame({'source': edges_src, 'target': edges_dst}).to_csv('../data/edges.csv', index=False)
    
    # train.csv (with features)
    train_with_features = nodes_df.merge(train_df, on='node_id')
    train_with_features.to_csv('../data/train.csv', index=False)
    
    # test.csv (with features, no target)
    test_with_features = nodes_df[nodes_df['node_id'].isin(test_df['node_id'])]
    test_with_features.to_csv('../data/test.csv', index=False)
    
    # test_labels.csv (hidden)
    test_df.to_csv('../data/test_labels.csv', index=False)
    
    # sample_submission.csv
    pd.DataFrame({'node_id': test_df['node_id'], 'target': 0}).to_csv('../submissions/sample_submission.csv', index=False)
    
    # label_names.txt
    with open('../data/label_names.txt', 'w') as f:
        for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {name}\n")
    
    print("Done!")

if __name__ == '__main__':
    prepare_data()
