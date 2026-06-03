# gensc-bench

Comparative Evaluation of Deep Generative Models for Capturing Topological Features in Brain Structural Connectivity

**Authors:** Chisato Kumada, Tomoyuki Hiroyasu, Satoru Hiwa

## Overview

This repository contains the implementation code for evaluating deep generative models (VAE, GAN, DDPM) on their ability to capture topological features in brain structural connectivity (SC) networks. The study includes experiments on both synthetic graph datasets and continuous-weighted brain connectivity data.

## Repository Structure

```
├── model/              # Model implementations (VAE, GAN, DDPM)
├── train/              # Training scripts and trainers
├── eval/               # Evaluation metrics and utilities
├── dataset/            # Dataset preparation and loading
├── common/             # Utility functions (graph utilities, visualization)
├── data/               # Dataset directory (binary/continuous/mmdref/target/MNIST/sc)
├── output/             # Evaluation results and plots
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gensc-bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Generating Synthetic Graphs

The synthetic graph datasets used in this study can be generated using the functions in `dataset/graph_prep.py`:

- **Barabási-Albert (BA) graphs**: Generated using `save_ba_adj_matrix()`
  - Parameters: n=64 nodes, m=4 attachment edges, 500 samples
  - Uses DFS node ordering for consistency

- **Watts-Strogatz (WS) graphs**: Generated using `save_ws_adj_matrix()`
  - Parameters: n=64 nodes, k=6 neighbors, p=0.05 rewiring probability, 500 samples
  - Uses k-core ordering

- **Stochastic Block Model (SBM) graphs**: Generated using `save_sbm_adj_matrix()`
  - Parameters: n=64 nodes, 2 communities, p_in=0.3, p_out=0.05, 500 samples
  - Uses community-based ordering

- **Weighted Scale-Free (WSF/WBA) graphs**: Generated using `save_wba_adj_matrix()`
  - Parameters: n=64 nodes, m=4 attachment edges, 500 samples
  - Uses DFS node ordering
  - Weights are assigned based on the weighted Barabási-Albert model

- **Planar graphs**: Obtained from the original repository
  - Source: [Graph generation repository](https://github.com/KarolisMart/SPECTREn)
  - 200 planar graphs with 64 nodes
  - Uses DFS node ordering

### Generating Datasets

To generate all synthetic graphs:

```python
from dataset.graph_prep import save_all_adj_matrices

# Generate all graphs and save to data/binary/raw/ and data/continuous/raw/
save_all_adj_matrices(num=500, output_path="data/binary/raw")
save_wba_adj_matrix(num=500, output_path="data/continuous/raw")
```

### Training

To train all models on all datasets:
```bash
python train/run_all_training.py
```

This script trains VAE, GAN, and DDPM models on each dataset and saves checkpoints to `checkpoints/`.

Individual model training:
```bash
python train/train_vae.py
python train/train_gan.py
python train/train_ddpm.py
```

### Evaluation

Evaluation is performed in multiple stages:

1. **Thresholding and MMD computation:**
```bash
python eval/evaluate.py
```
This script:
- Applies threshold-based binarization to continuous outputs
- Computes Maximum Mean Discrepancy (MMD) between original and generated graphs
- Saves results to `output/mmd_results/`

2. **Graph statistics and metrics:**
Results are computed and saved in `output/metrics/` including:
- Hub analysis (Spearman correlation)
- Modularity scores
- Planarity rates
- Sigma values (network efficiency measures)

3. **Visualization:**
- Mean adjacency matrices: `output/mean_matrices/`
- Plots: `output/plots/`

### Dataset Details

**Note on SC (Structural Connectivity) Data:** SC datasets are not included in this repository.

### Node Ordering

For consistent node ordering across datasets:
- **SC graphs:** Nodes ordered consistently across samples
- **Synthetic graphs (BA, WSF):** Depth-First Search (DFS) ordering
- **Planar graphs:** DFS ordering
- **SBM:** Community-based ordering (by community ID, then k-core number, then degree)
- **WS:** k-core based ordering (by k-core number, then degree)

## Key Files

- `path_list.json` - Training dataset to model output mappings
- `processed_path_list.json` - Configuration for post-processed graph paths
- `train/run_all_training.py` - Main training pipeline for all models
- `eval/evaluate.py` - Main evaluation pipeline
- `eval/graph_mmd.py` - MMD and statistical comparison metrics
- `common/graph_util.py` - Graph utility functions

## Configuration

### Path Configuration Files

Three JSON files manage dataset paths and pipeline workflows:

#### 1. `path_list.json` - Training Dataset Configuration
Maps training datasets to their reference and generated outputs:
```json
{
    "ba": {
        "reference": "data/binary/raw/ba_dfs_adj_matrix.npz",
        "reconstructions": [
            "checkpoints/binary/ba/vae/generated_graphs.npz",
            "checkpoints/binary/ba/gan/generated_graphs.npz",
            "checkpoints/binary/ba/ddpm/generated_graphs.npz"
        ]
    }
}
```

#### 2. `processed_path_list.json` - Post-Processing Configuration
Manages paths after threshold-based processing:
```json
{
    "ba": {
        "original": "path/to/your/dataset/ba_original.npz",
        "reconstructions": {
            "vae": "path/to/your/processed/ba_vae_thresholded.npz",
            "gan": "path/to/your/processed/ba_gan_thresholded.npz",
            "ddpm": "path/to/your/processed/ba_ddpm_thresholded.npz"
        }
    }
}
```

### Model and Hyperparameter Configuration

Individual training scripts in `train/` contain hyperparameter settings:
- `train_vae.py` - VAE training hyperparameters (latent_dim, learning_rate, etc.)
- `train_gan.py` - GAN training hyperparameters
- `train_ddpm.py` - DDPM training hyperparameters

## Code Review Notes

This code is provided for review and reproduction purposes. The current state is pre-publication and intended to enable:
- Verification of experimental methodology
- Code inspection and review
- Reproducibility of results using generated datasets

### Notes on Data and Graphs

- **Datasets are not included** in this repository due to size and licensing constraints
- All synthetic graphs (BA, SBM, WS, WSF) can be generated using functions in `dataset/graph_prep.py`
- Planar graphs should be obtained from the [Graph Generation Repository](https://github.com/JiaxuanYou/graph-generation)
- **Brain connectivity (SC) data is not included** - this data requires access to the original source with appropriate permissions
- SC data preparation code is provided in `dataset/graph_dataset.py` for reference, but users must obtain the source data independently

## Acknowledgments

Parts of the code and evaluation framework are derived from:

### ORCA (Graph Motif Counter) and Evaluation Metrics

The evaluation pipeline utilizes the ORCA (Orbit Counting Algorithm) tool and metric computation scripts, which have the following distinct origins:

- **ORCA Tool (`eval/orca/`):** The original ORCA algorithm was developed by Tomaž Hočevar and Janez Demšar (Bioinformatics, 2014). The source code included in this repository was obtained directly from the GraphRNN repository.
  - Original Paper: *A combinatorial approach to graphlet counting* (Hočevar & Demšar, 2014)

- **Evaluation Code (`eval/graph_mmd.py`):** The scripts for computing MMD (Maximum Mean Discrepancy) and evaluating graph motifs are derived from **GraphRNN** (ICML 2018). The code has been partially modified.
  - Reference: *GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model* (You et al., 2018)
  - Repository: [Graph-Generation](https://github.com/JiaxuanYou/graph-generation)