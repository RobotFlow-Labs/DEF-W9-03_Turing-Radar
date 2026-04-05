# PRD.md -- Turing-Radar Master Build Plan

## Module Overview

**Module:** Turing-Radar
**Paper:** The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving (arXiv 2602.03856)
**Domain:** Electronic Warfare / Radar Signal Processing
**Task:** Unsupervised clustering of interleaved radar pulses into emitter groups
**Primary Metric:** Median V-measure

## Scope

This module implements a challenge-compatible ANIMA defense scaffold for radar pulse
deinterleaving on the Turing Synthetic Radar Dataset (TSRD). The baseline uses
HDBSCAN clustering on preprocessed PDW features. The architecture includes extension
points for learned embedding models and CUDA-accelerated inference.

## Build Plan

| PRD | Title | Priority | Depends On | Status | File |
|-----|-------|----------|------------|--------|------|
| PRD-01 | Foundation and Data Contracts | P0 | -- | Done | [prds/PRD-01-foundation.md](prds/PRD-01-foundation.md) |
| PRD-02 | Core Deinterleaver Models | P0 | PRD-01 | Done | [prds/PRD-02-core-model.md](prds/PRD-02-core-model.md) |
| PRD-03 | Inference Pipeline and CLI | P0 | PRD-02 | Done | [prds/PRD-03-inference.md](prds/PRD-03-inference.md) |
| PRD-04 | Evaluation and Benchmarking | P1 | PRD-03 | Done | [prds/PRD-04-evaluation.md](prds/PRD-04-evaluation.md) |
| PRD-05 | API and Container Serving | P1 | PRD-03 | Done | [prds/PRD-05-api-docker.md](prds/PRD-05-api-docker.md) |
| PRD-06 | ROS2 Integration | P1 | PRD-05 | Done | [prds/PRD-06-ros2-integration.md](prds/PRD-06-ros2-integration.md) |
| PRD-07 | Production Hardening and CUDA Handoff | P2 | PRD-04,05,06 | Done | [prds/PRD-07-production.md](prds/PRD-07-production.md) |

## Deliverables Summary

### Code (src/anima_turing_radar/)
- `constants.py` -- PDW feature names, indices, units
- `config.py` -- TOML config loader with typed dataclasses
- `io.py` -- HDF5 pulse train load/save utilities
- `preprocess.py` -- PDW preprocessing pipeline (sort, delta-ToA, standardize)
- `model.py` -- Deinterleaver interface + sklearn wrappers (HDBSCAN, DBSCAN, KMeans)
- `embedding.py` -- Optional torch embedding stub for learned features
- `evaluate.py` -- Challenge-compatible clustering metrics
- `infer.py` -- CLI inference entrypoint
- `api.py` -- FastAPI serving layer (/health, /ready, /predict)
- `ros2_node.py` -- ROS2 integration skeleton

### Configs (configs/)
- `default.toml` -- development defaults
- `paper.toml` -- paper/challenge baseline profile

### Tests (tests/)
- `test_config.py`, `test_io.py`, `test_preprocess.py`
- `test_model.py`, `test_evaluate.py`
- `test_infer.py`, `test_api.py`

### Infrastructure
- `docker/Dockerfile.serve` -- container serving
- `docker/Dockerfile.cuda` -- CUDA variant
- `docker/docker-compose.serve.yml` -- compose profiles
- `scripts/benchmark_baseline.py` -- baseline benchmark harness
- `anima_module.yaml` -- ANIMA module manifest

## Dataset Requirements

- **Name:** Turing Synthetic Radar Dataset (TSRD)
- **Source:** HuggingFace `alan-turing-institute/turing-deinterleaving-challenge`
- **Local:** `/mnt/forge-data/datasets/turing_deinterleaving_challenge`
- **Size:** ~6000 HDF5 files (stare + scan), ~3B total pulses
- **Format:** HDF5 with `data` (Nx5 float64), `labels` (N int), `metadata` (dict)

## Model Requirements

- No pretrained weights (clustering baseline)
- scikit-learn >= 1.4 (HDBSCAN built-in)
- Optional: PyTorch cu128 for learned embedding extension

## References

- Paper PDF: `papers/2602.03856.pdf`
- Challenge repo: `repositories/turing-deinterleaving-challenge`
- ASSETS.md: dataset scale, PDW contract, baseline metrics
