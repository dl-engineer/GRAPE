# GRAPE - Graph Retinal Analysis for Prediction and Evaluation

A GNN benchmark for diabetic retinopathy classification from retinal vessel graphs.

**Task:** Binary graph classification (healthy vs DR)  
**Metric:** Macro F1  
**Data:** DRIVE[[1]](https://drive.grand-challenge.org/) + STARE[[2]](https://cecas.clemson.edu/~ahoover/stare/) + HRF[[3]](https://www5.cs.fau.de/research/data/fundus-images/) (70 graphs total)

📊 **[Live Leaderboard](https://muhammad0isah.github.io/GRAPE/leaderboard.html)**

---

## Motivation

- **463 million** people have diabetes globally
- **Diabetic retinopathy (DR)** is the leading cause of blindness in working-age adults
- Retinal blood vessels form **natural graphs** - bifurcations as nodes, vessel segments as edges
- Vessel topology (branching patterns, tortuosity, connectivity) indicates disease progression
- **No unified GNN benchmark** exists for retinal vessel graph analysis

This challenge evaluates GNN methods on clinically-relevant retinal vessel classification.

---

## Data Sources

| Dataset | Images | Healthy | DR |
|---------|--------|---------|-----|
| DRIVE[[1]](https://drive.grand-challenge.org/) | 20 | 17 | 3 |
| STARE[[2]](https://cecas.clemson.edu/~ahoover/stare/) | 20 | 16 | 4 |
| HRF[[3]](https://www5.cs.fau.de/research/data/fundus-images/) | 30 | 15 | 15 |
| **Total** | **70** | **48** | **22** |

**Graph IDs:**
- `D_XX` = DRIVE image XX
- `S_XX` = STARE image XX
- `H_XX` = HRF healthy image XX
- `R_XX` = HRF DR image XX

---

## Structure

```
data/
├── public/                 # Pre-processed (ready to use)
│   ├── train_data.csv      # 55 graphs, node features + edges
│   ├── train_labels.csv    # Training labels
│   ├── test_data.csv       # 15 graphs for prediction
│   └── sample_submission.csv
└── raw/                    # Original images + masks
    ├── drive/              # XX_training.tif + XX_manual1.gif
    ├── stare/              # imXXXX.ppm + imXXXX.ah.ppm
    └── hrf/                # XX_h.jpg/XX_dr.JPG + XX_h.tif/XX_dr.tif
```

**Two options:**
1. Use pre-processed CSVs directly (recommended)
2. Use raw images for custom graph extraction

**File naming:**
- `D_21` → `drive/21_training.tif` + `21_manual1.gif`
- `S_4` → `stare/im0004.ppm` + `im0004.ah.ppm`
- `H_5` → `hrf/05_h.jpg` + `05_h.tif`
- `R_5` → `hrf/05_dr.JPG` + `05_dr.tif`

---

## CSV Data Format

### train_data.csv / test_data.csv
| Column | Description |
|--------|-------------|
| graph_id | Graph ID (D_21, S_44, etc.) |
| node_id | Node ID within graph |
| x, y | Coordinates (pixels) |
| width | Vessel width |
| type | 1=junction, 0=endpoint |
| edges | Adjacent nodes (semicolon-separated) |

```csv
graph_id,node_id,x,y,width,type,edges
D_21,0,165.8,64.8,2.8,1,7;105
D_21,1,324.7,77.5,4.0,1,3;10;101
```

### train_labels.csv
| Column | Description |
|--------|-------------|
| graph_id | Graph ID |
| label | 0=healthy, 1=diabetic retinopathy |

---

## Submission Format

```csv
graph_id,label
S_4,0
S_235,1
```

- `graph_id` must match test_data.csv
- `label` must be 0 or 1

---

## How to Submit

1. Fork this repo
2. Add `submissions/inbox/<team>/predictions.csv`
3. Open PR

---

## License

MIT
