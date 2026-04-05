# ANIMA Turing-Radar

Challenge-compatible ANIMA defense module scaffold for radar pulse deinterleaving on TSRD.

## Scope
- Reads TSRD-style H5 pulse trains (`data`, optional `labels`, optional `metadata`)
- Runs baseline clustering deinterleaving (HDBSCAN/DBSCAN/KMeans/Agglomerative)
- Supports CLI inference, metrics evaluation, and FastAPI serving
- Includes ROS2 integration skeleton and CUDA handoff plan

## Quickstart

### 1. Install
```bash
python3 -m pip install -e .[dev]
```

### 2. Run tests
```bash
python3 -m pytest -v
```

### 3. Inference on one pulse train
```bash
python3 -m anima_turing_radar.infer \
  --input data/test/test_0.h5 \
  --config configs/paper.toml
```

### 4. Run API
```bash
python3 -m uvicorn anima_turing_radar.api:create_app --factory --host 0.0.0.0 --port 8080
```

### 5. Baseline benchmark
```bash
python3 scripts/benchmark_baseline.py --config configs/paper.toml --max-files 5
```

## Data
- Paper: `papers/2602.03856.pdf`
- Challenge repo: `repositories/turing-deinterleaving-challenge`
- Dataset: https://huggingface.co/datasets/alan-turing-institute/turing-deinterleaving-challenge

## Notes
- The source paper is dataset-focused; model training recipe is intentionally not fixed.
- This implementation prioritizes baseline reproducibility and extension points for CUDA migration.
