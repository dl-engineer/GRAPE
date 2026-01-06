import pandas as pd
from sklearn.metrics import f1_score, classification_report
import sys
import os

def score_submission(submission_file):
    if not os.path.exists(submission_file):
        print(f"Error: File '{submission_file}' not found.")
        return None
    
    submission = pd.read_csv(submission_file)
    
    if 'node_id' not in submission.columns or 'target' not in submission.columns:
        print("Error: Need 'node_id' and 'target' columns.")
        return None
    
    truth = pd.read_csv('data/test_labels.csv')
    merged = truth.merge(submission, on='node_id', how='left', suffixes=('_true', '_pred'))
    
    if merged['target_pred'].isna().any():
        print(f"Warning: {merged['target_pred'].isna().sum()} nodes missing.")
        merged = merged.dropna()
    
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    label_names = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                   'Probabilistic_Methods', 'Reinforcement_Learning', 
                   'Rule_Learning', 'Theory']
    
    print("=" * 50)
    print("Cora Challenge - Submission Scoring")
    print("=" * 50)
    print(f"\nScoring: {submission_file}")
    print("\n" + "=" * 50)
    print(f"  MACRO F1 SCORE: {macro_f1:.4f}")
    print("=" * 50)
    print("\nPer-class breakdown:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    
    return macro_f1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py submissions/file.csv")
        sys.exit(1)
    score_submission(sys.argv[1])
