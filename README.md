# DBLP Venue Analysis – README

This project implements **classification**, **clustering**, **anomaly detection**, and **Network-style EDA** on venues from the DBLP citation dataset, using TF-IDF-based text features and citation metadata and a written report.

---

## Contents

- `edagraph.ipynb`  
  - Loads DBLP data and builds citation graphs.  
  - Visualizes:
    - Author-citation network for the top-cited authors.
    - Paper-citation network for the top-cited papers. 

- `classification.ipynb`  
  - Constructs a balanced multi-class dataset from the six most frequent venues (3,000 papers per venue, 18,000 total). 
  - Trains and evaluates several models (e.g., LightGBM, Random Forest, SVM, KNN, Decision Tree) with accuracy/precision/recall/F1 and ROC curves.

- `clustering.ipynb`  
  - Uses venue-level TF-IDF features (titles concatenated per venue), reduced with PCA and z-score scaled.  
  - Runs:
    - K-Means for multiple values of k, with SSE and silhouette analysis.
    - Agglomerative clustering (single / complete / average / ward linkage).
    - DBSCAN with a parameter sweep over eps and min_samples, including a penalized silhouette that accounts for noise points. 
  - Identifies reasonable clusterings and interprets top clusters in terms of research areas (discrete math, software engineering, applied math, etc.).

- `anomaly.ipynb`  
  - Fits a K-Means model on venue features and uses distance to cluster centers as an anomaly score.  
  - Plots:
    - A histogram of anomaly scores with percentile cutoffs (e.g., 95th, 99th).
    - A 2D PCA projection colored by anomaly score, labeling the highest-anomaly venues.
  - Highlights highly unusual venues (e.g., very math-heavy journals) whose vocabulary diverges from mainstream CS venues.

- `COSC 3337 Report.pdf`  
  - Final report that summarizes:
    - Data preprocessing pipeline.
    - Classification setup and model comparison
    - Clustering experiments and cluster interpretation.- Anomaly detection method and analysis.

---

## Dataset

This project uses the DBLP-based dataset provided in class: ~3M papers and ~25M citation links, each with id, title, authors, venue, year, citation count, references, and abstract. 

Typical raw files (inside a dataset/ folder):

- `dblp-ref-0.json`  
- `dblp-ref-1.json`  
- `dblp-ref-2.json`  
- `dblp-ref-3.json`

Key preprocessing steps (implemented across the notebooks and documented in the report):
- Drop rows with missing/empty title or venue and remove generic venues like “arXiv” / “CoRR”.
- Keep only venues with > 1,000 papers.

- For clustering/anomaly:
  - Concatenate all titles within a venue into one document.
  - Apply TF-IDF (e.g., max_features=1000, English stopwords), reduce to ~50 dimensions with PCA, then standardize (z-score).
- For classification:
  - Sample the six most frequent venues (3,000 papers each).
  - Use text features (SVM on TF-IDF) plus numeric metadata (citations, references, authors, year).
---

## Environment

Created with:

- Python 3.10+
- Packages:
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`
  - `scikit-learn`, `scipy`
  - `networkx`, `lightgbm`
  - `jupyter`

Install (one-time):
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy networkx lightgbm jupyter
```

---

## How to Run

1. Place these files together in one folder:
   - `edagraph.ipynb`
   - `classification.ipynb`
   - `clustering.ipynb`
   - `anomaly.ipynb`
   - DBLP data files in a dataset/ folder

COSC 3337 Report.pdf
2. Open the notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run each notebook as needed (recommended order):
    1. edagraph.ipynb - build and visualize citation graphs.
    2. classification.ipynb - build the balanced dataset and train/evaluate models
    3. clustering.ipynb - run K-Means, hierarchical, and DBSCAN clustering.
    4. anomaly.ipynb - compute anomaly scores and visualize outliers.


4. From the Jupyter UI, select **Kernel → Restart & Run All** (or run cells top‑to‑bottom).

---

## Reproducing the Results/Plots

Due to random train/test splits and random initialization in models like K-Means and some classifiers, exact numbers may differ slightly between runs.