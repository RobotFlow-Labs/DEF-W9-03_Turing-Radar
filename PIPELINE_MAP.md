# PIPELINE_MAP.md -- Turing-Radar

## Pipeline Steps

### Phase 1: Scaffolding (BUILD ONLY)
- [x] DONE -- PRD-01: Foundation and data contracts (config, IO, preprocess)
- [x] DONE -- PRD-02: Core deinterleaver models (HDBSCAN/DBSCAN/KMeans wrappers)
- [x] DONE -- PRD-03: Inference pipeline and CLI
- [x] DONE -- PRD-04: Evaluation and benchmarking (V-measure, ARI, AMI)
- [x] DONE -- PRD-05: API and container serving (FastAPI + Docker)
- [x] DONE -- PRD-06: ROS2 integration skeleton
- [x] DONE -- PRD-07: Production hardening (benchmark script, NEXT_STEPS, README)
- [x] DONE -- Module manifest (anima_module.yaml, CLAUDE.md, PRD.md)

### Phase 2: Data Provisioning (CUDA SERVER)
- [x] DONE -- Download TSRD dataset to /mnt/train-data/datasets/turing_deinterleaving_challenge
- [x] DONE -- Validate HDF5 file integrity (shape, dtype, label presence)
- [x] DONE -- Run sample-file IO checks (test_0.h5: 29748 pulses, 78 emitters)

### Phase 3: Baseline Validation (CUDA SERVER)
- [x] DONE -- Install module in editable mode with venv
- [x] DONE -- Run full test suite on CUDA server (17/17 pass)
- [x] DONE -- Bug fix: model.py super() TypeError in slots dataclass
- [x] DONE -- Bug fix: io.py labels (N,1) shape handling
- [x] DONE -- Baseline HDBSCAN evaluation (3-file: median V=0.758)
- [ ] TODO -- Full benchmark on all 250 test files (HDBSCAN slow on large files)

### Phase 4: Model Extension (CUDA SERVER)
- [x] DONE -- Training script (train.py) with triplet contrastive loss
- [x] DONE -- Export script (export_model.py) -- pth, safetensors, ONNX, TRT
- [x] DONE -- Training configs (debug.toml, train.toml)
- [x] DONE -- Training tests (5/5 pass)
- [ ] TODO -- Train embedding on train split (GPU required)
- [ ] TODO -- Evaluate embedding + clustering vs HDBSCAN baseline
- [ ] TODO -- Checkpoint best model to /mnt/artifacts-datai/checkpoints/turing-radar/

### Phase 5: Export and Production (CUDA SERVER)
- [ ] TODO -- ONNX export of embedding model
- [ ] TODO -- TensorRT fp16 + fp32 export
- [ ] TODO -- Push checkpoint to HuggingFace
- [ ] TODO -- Docker build and serve verification
- [ ] TODO -- ROS2 node integration test
