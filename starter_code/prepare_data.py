"""
Data Preparation Script for Cora Challenge
==========================================
Run this once to download and prepare the Cora dataset.
"""

import urllib.request
import tarfile
import os
import numpy as np
import pandas as pd

def download_cora():
    """Download Cora dataset from original source"""
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    filename = "cora.tgz"
    
    if not os.path.exists(filename):
        print("Downloading Cora dataset...")
        urllib.request.urlretrieve(url, filename)
    
    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    return "cora"

def prepare_challenge_data():
    """Prepare challenge data from Cora"""
    
    # Download if needed
    if not os.path.exists('cora'):
        download_cora()
    
    # Load content file (node features and labels)
    print("Loading node data...")
    content_file = 'cora/cora.content'
    
    node_ids = []
    features = []
    labels = []
    
    label_map = {}
    label_idx = 0
    
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = parts[0]
            feat = [int(x) for x in parts[1:-1]]
            label = parts[-1]
            
            if label not in label_map:
                label_map[label] = label_idx
                label_idx += 1
            
            node_ids.append(node_id)
            features.append(feat)
            labels.append(label_map[label])
    
    # Create node id mapping
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Load edges
    print("Loading edges...")
    cites_file = 'cora/cora.cites'
    edges_src = []
    edges_dst = []
    
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                src, dst = parts
                if src in node_to_idx and dst in node_to_idx:
                    edges_src.append(node_to_idx[src])
                    edges_dst.append(node_to_idx[dst])
    
    # Convert to arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Nodes: {len(node_ids)}")
    print(f"Edges: {len(edges_src)}")
    print(f"Features: {features.shape[1]}")
    print(f"Classes: {len(label_map)}")
    print(f"Label mapping: {label_map}")
    
    # Create train/test split (140 train, rest test - standard Cora split)
    np.random.seed(42)
    
    # Stratified sample: 20 per class
    train_indices = []
    for c in range(len(label_map)):
        class_indices = np.where(labels == c)[0]
        selected = np.random.choice(class_indices, min(20, len(class_indices)), replace=False)
        train_indices.extend(selected)
    
    train_indices = np.array(train_indices)
    all_indices = np.arange(len(labels))
    test_indices = np.setdiff1d(all_indices, train_indices)
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create output directory
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../submissions', exist_ok=True)
    
    # Save nodes.csv
    nodes_df = pd.DataFrame(features)
    nodes_df.columns = [f'feat_{i}' for i in range(features.shape[1])]
    nodes_df.insert(0, 'node_id', range(len(features)))
    nodes_df.to_csv('../data/nodes.csv', index=False)
    print("Saved data/nodes.csv")
    
    # Save edges.csv
    edges_df = pd.DataFrame({'source': edges_src, 'target': edges_dst})
    edges_df.to_csv('../data/edges.csv', index=False)
    print("Saved data/edges.csv")
    
    # Save train.csv
    train_df = pd.DataFrame({
        'node_id': train_indices,
        'label': labels[train_indices]
    })
    train_df.to_csv('../data/train.csv', index=False)
    print("Saved data/train.csv")
    
    # Save test.csv (no labels!)
    test_df = pd.DataFrame({'node_id': test_indices})
    test_df.to_csv('../data/test.csv', index=False)
    print("Saved data/test.csv")
    
    # Save hidden test labels (for scoring only!)
    test_labels_df = pd.DataFrame({
        'node_id': test_indices,
        'label': labels[test_indices]
    })
    test_labels_df.to_csv('../data/test_labels.csv', index=False)
    print("Saved data/test_labels.csv (HIDDEN - for scoring only)")
    
    # Save sample submission
    sample_df = pd.DataFrame({
        'node_id': test_indices,
        'label': np.zeros(len(test_indices), dtype=int)
    })
    sample_df.to_csv('../submissions/sample_submission.csv', index=False)
    print("Saved submissions/sample_submission.csv")
    
    # Save label mapping
    with open('../data/label_names.txt', 'w') as f:
        for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {name}\n")
    print("Saved data/label_names.txt")
    
    print("\nDone! Challenge data ready.")

if __name__ == '__main__':
    prepare_challenge_data()
