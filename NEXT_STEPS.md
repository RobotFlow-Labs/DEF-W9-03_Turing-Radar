# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 65%

## Done
- [x] Paper read and summarized (arXiv 2602.03856)
- [x] PRD suite created (7 PRDs in prds/)
- [x] Granular task breakdown (tasks/)
- [x] Source code scaffolding (src/anima_turing_radar/ -- 12 modules)
- [x] Baseline benchmark script (scripts/benchmark_baseline.py)
- [x] FastAPI serving layer (api.py)
- [x] ROS2 integration skeleton (ros2_node.py)
- [x] Tests written (17 tests, all passing)
- [x] TOML configs (default.toml, paper.toml, debug.toml, train.toml)
- [x] Docker serving files (Dockerfile.serve, Dockerfile.cuda, docker-compose)
- [x] Module manifest (anima_module.yaml)
- [x] Documentation (CLAUDE.md, PRD.md, PIPELINE_MAP.md)
- [x] Bug fix: model.py super() TypeError (ABC -> Protocol)
- [x] Bug fix: io.py labels shape (N,1) -> flatten to (N,)
- [x] Dataset downloaded to /mnt/train-data/datasets/turing_deinterleaving_challenge (2500 train, 250 test)
- [x] Venv created (symlinked from /mnt/artifacts-datai/venvs/turing-radar)
- [x] torch 2.11.0+cu128 verified, 8x L4 GPUs available
- [x] Baseline HDBSCAN validation: V-measure=0.758 median (3-file subset)
- [x] Single-file inference: test_0.h5 V-measure=0.894, 74/78 emitters
- [x] Training script (train.py) with triplet contrastive loss
- [x] Export script (scripts/export_model.py) -- pth, safetensors, ONNX, TRT
- [x] Training tests (5 tests for dataset, loss, scheduler, early stopping)
- [x] Config paths updated to /mnt/train-data/datasets/

## In Progress
- [ ] Embedding model training (waiting for GPU allocation)

## TODO
- [ ] Ask user for GPU allocation, run /gpu-batch-finder
- [ ] Train embedding model on train split (50 epochs)
- [ ] Evaluate embedding+HDBSCAN vs baseline HDBSCAN
- [ ] Export: pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [ ] Push checkpoint to HuggingFace (ilessio-aiflowlab/turing-radar-checkpoint)
- [ ] Docker build + serve verification
- [ ] Full baseline benchmark on all 250 test files

## Blocking
- Need GPU allocation for training (ask user which GPU to use)

## Notes
- Dataset is flat: archive/train/ and archive/test/ (no stare/scan split at file level)
- Labels are (N,1) int8 -- flattened in io.py
- HDBSCAN is O(n*log(n)) so large files (>100K pulses) take minutes
- Embedding model: PulseEmbeddingNet(5 -> 128 -> 32) with triplet contrastive loss
