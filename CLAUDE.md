# CLAUDE.md -- Turing-Radar

## Paper Summary

**Title:** The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving
**ArXiv:** 2602.03856 (January 23, 2026)
**Authors:** Edward Gunn, Adam Hosford, Robert Jones, Leo Zeitler, Ian Groves, Victoria Nockles
**Institution:** The Alan Turing Institute / Dstl

The paper introduces the Turing Synthetic Radar Dataset (TSRD), the first publicly
available, comprehensively simulated pulse train dataset for radar pulse deinterleaving.
The task is to separate interleaved radar pulses from multiple unknown emitters in
electronic warfare (EW) scenarios. The dataset contains 6000 pulse trains across two
receiver modes (stare and scan), totalling almost 3 billion pulses with up to 110
emitters per train and significant parameter-space overlap between emitters.

The accompanying Turing Deinterleaving Challenge frames the problem as unsupervised
clustering with unknown emitter count, measured primarily by median V-measure.

## Problem Formulation

- **Input:** Variable-length sequence of 5D Pulse Descriptor Words (PDWs)
- **Output:** Cluster assignment (emitter ID) per pulse
- **Constraint:** Number of emitters is unknown at test time
- **Formulation:** Unsupervised clustering (not classification)
- **Labels:** Pulse-train local only (not globally consistent across files)

## Architecture

This is a **dataset paper**, not a model paper. There is no prescribed model architecture.
The module implements challenge-compatible baselines:

1. **Baseline:** HDBSCAN identity clustering on raw/preprocessed PDW features
2. **Extension point:** Learned embedding model + clustering head (stub provided)

### Baseline Pipeline
```
H5 pulse train -> PDW extraction (5D) -> Preprocessing -> Clustering -> Labels
```

### Preprocessing Steps
1. Sort pulses by Time of Arrival (ToA)
2. Compute delta-ToA (optional)
3. Standardize features (zero-mean, unit-variance)
4. Clip extreme values (default: 8 sigma)

## PDW Feature Contract

Each pulse is a 5D vector:
| Index | Feature            | Unit          | Range (stare)               |
|------:|--------------------|---------------|-----------------------------|
|     0 | time_of_arrival    | microseconds  | 0 -- 44.4M                  |
|     1 | centre_frequency   | MHz           | 500 -- 16090                |
|     2 | pulse_width        | microseconds  | 0.007 -- 229                |
|     3 | angle_of_arrival   | degrees       | -180 -- 180                 |
|     4 | amplitude          | dB            | -213 -- 93                  |

## Dataset

| Receiver | Pulse Trains | Total Pulses | Max Emitters | Split              |
|----------|-------------:|-------------:|-------------:|--------------------|
| Stare    |        3,000 |  2,876M      |           77 | 2500/250/250       |
| Scan     |        3,000 |     90M      |           90 | 2500/250/250       |

- Source: https://huggingface.co/datasets/alan-turing-institute/turing-deinterleaving-challenge
- Local path: `/mnt/forge-data/datasets/turing_deinterleaving_challenge`
- Format: HDF5 files with `data` (Nx5 float), optional `labels` (N int), optional `metadata`

### Dataset Statistics (stare mode)
| Feature        | Mean         | Std            | Min      | Max          |
|----------------|-------------:|---------------:|---------:|-------------:|
| ToA (us)       | 8,475,956    | 9,032,928      | 0.0      | 44,449,840   |
| Frequency (MHz)| 4,736        | 3,338          | 500      | 16,090       |
| Pulse Width(us)| 4.7          | 17.4           | 0.007    | 229          |
| AoA (deg)      | 8.9          | 94.3           | -180     | 180          |
| Amplitude (dB) | -43.8        | 50.9           | -212     | 93           |

## Hyperparameters (Baseline)

The paper is model-agnostic. For challenge-compatible baseline:

```toml
[clustering]
algorithm = "hdbscan"
min_cluster_size = 20
min_samples = 5
eps = 0.0               # identity baseline
default_label = -1

[preprocess]
sort_by_toa = true
delta_toa = true
standardize = true
clip_std = 8.0
```

## Baseline Metrics

| Model                    | Receiver | V-measure | ARI   | AMI   |
|--------------------------|----------|----------:|------:|------:|
| HDBSCAN identity         | Stare    |     0.538 | 0.270 | 0.496 |
| HDBSCAN identity         | Scan     |     0.187 | 0.017 | 0.146 |

Primary metric of record: **median V-measure**

## Model Requirements

- No pretrained weights required (clustering baseline)
- Optional: PyTorch for learned embedding extension (`embedding.py` stub)
- scikit-learn >= 1.4 for HDBSCAN

## Evaluation Metrics

Challenge-standard clustering metrics (all from scikit-learn):
- V-measure (primary)
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity
- Completeness
- Matthews Correlation Coefficient (MCC)
- F1 score

## Module Commands

```bash
# Activate venv
cd /mnt/forge-data/modules/05_wave9/03_Turing-Radar
source .venv/bin/activate

# Run tests
uv run pytest tests/ -v

# Inference on one pulse train
uv run python -m anima_turing_radar.infer \
  --input data/test/test_0.h5 \
  --config configs/paper.toml

# Baseline benchmark
uv run python scripts/benchmark_baseline.py --config configs/paper.toml --max-files 5

# API server
uv run python -m uvicorn anima_turing_radar.api:create_app --factory --host 0.0.0.0 --port 8080
```

## Key Research Directions (from paper Section III)

1. **Varying sequence lengths** -- adaptive windowing, hierarchical processing
2. **Feature extraction** -- learned fixed-length representations for downstream tasks
3. **Computational efficiency** -- sub-quadratic algorithms for real-time edge deployment

## Constraints

- Labels are pulse-train local only (label IDs not globally consistent)
- Test windows have unknown emitter count
- Formulation is clustering, not closed-set classification
- Module must remain challenge-compatible
- No model weights to download (clustering baseline)
