import pandas as pd
from sklearn.metrics import f1_score
import sys
import os
import re
from datetime import datetime

def update_leaderboard(submission_file, team_name=None):
    # Extract team name from filename if not provided
    if team_name is None:
        team_name = os.path.basename(submission_file).replace('.csv', '')
    
    # Score the submission
    submission = pd.read_csv(submission_file)
    truth = pd.read_csv('data/test_labels.csv')
    merged = truth.merge(submission, on='node_id', how='left', suffixes=('_true', '_pred'))
    y_true = merged['target_true'].values
    y_pred = merged['target_pred'].values
    score = f1_score(y_true, y_pred, average='macro')
    
    # Read current leaderboard
    with open('leaderboard.md', 'r') as f:
        content = f.read()
    
    # Parse existing entries
    lines = content.split('\n')
    header_idx = None
    entries = []
    
    for i, line in enumerate(lines):
        if '| Rank |' in line:
            header_idx = i
        elif header_idx and i > header_idx + 1 and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 4 and parts[0].isdigit():
                entries.append({
                    'team': parts[1],
                    'score': float(parts[2]),
                    'method': parts[3],
                    'date': parts[4] if len(parts) > 4 else ''
                })
    
    # Add new entry
    today = datetime.now().strftime('%Y-%m-%d')
    new_entry = {'team': team_name, 'score': round(score, 4), 'method': 'GNN', 'date': today}
    
    # Check if team already exists, update if better score
    updated = False
    for entry in entries:
        if entry['team'] == team_name:
            if score > entry['score']:
                entry['score'] = round(score, 4)
                entry['date'] = today
            updated = True
            break
    
    if not updated:
        entries.append(new_entry)
    
    # Sort by score descending
    entries.sort(key=lambda x: x['score'], reverse=True)
    
    # Generate new leaderboard
    new_content = """# Leaderboard

| Rank | Team | Macro F1 | Method | Date |
|------|------|----------|--------|------|
"""
    for i, entry in enumerate(entries, 1):
        new_content += f"| {i} | {entry['team']} | {entry['score']:.4f} | {entry['method']} | {entry['date']} |\n"
    
    new_content += """
## How to Submit

1. Fork this repository
2. Run your model and generate predictions
3. Save as `submissions/your_team_name.csv`
4. Create a Pull Request

## Submission Format

```csv
node_id,target
0,3
1,5
2,1
```
"""
    
    with open('leaderboard.md', 'w') as f:
        f.write(new_content)
    
    print(f"Leaderboard updated: {team_name} with score {score:.4f}")
    return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_leaderboard.py submissions/file.csv [team_name]")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    team_name = sys.argv[2] if len(sys.argv) > 2 else None
    update_leaderboard(submission_file, team_name)
