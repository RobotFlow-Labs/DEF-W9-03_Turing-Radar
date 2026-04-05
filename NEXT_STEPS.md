# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 90%

## Done
- [x] Paper read and summarized (arXiv 2602.03856)
- [x] PRD suite created (7 PRDs in prds/)
- [x] Source code (12 modules + train.py + serve.py)
- [x] Bug fixes: model.py super() TypeError, io.py labels (N,1), EmbeddedCluster super()
- [x] Dataset downloaded (2500 train + 250 test H5 files)
- [x] Venv + torch 2.11.0+cu128 + TensorRT 10.16
- [x] 17/17 tests passing
- [x] Baseline HDBSCAN: V-measure=0.758 (3-file subset)
- [x] Embedding model trained: val_loss=0.003803 (epoch 44/46, 4.0h on L4)
- [x] Embedding+HDBSCAN: median V-measure=0.792 (+3.4% vs baseline)
- [x] Export: pth (86KB) + safetensors (84KB) + ONNX (84KB) + TRT FP16 (156KB) + TRT FP32 (157KB)
- [x] HuggingFace push: ilessio-aiflowlab/turing-radar-checkpoint
- [x] Hero page (Industrial Cyberpunk defense theme, 1280x640)
- [x] ANIMA infra: Dockerfile.serve, Dockerfile.cuda, docker-compose, .env.serve, serve.py, ros2_node.py
- [x] anima_module.yaml manifest
- [x] TRAINING_REPORT.md with full metrics
- [x] Git pushed to origin main

## TODO
- [ ] Full benchmark on all 250 test files (HDBSCAN slow on large files)
- [ ] Docker build verification (needs docker daemon)
- [ ] ROS2 node integration test (needs ROS2 runtime)

## Notes
- Model is tiny (21K params, 86KB) — VRAM usage negligible (634MB)
- HDBSCAN is the bottleneck, not the embedding model
- Embedding helps most on small files with few emitters
- No custom CUDA kernels needed for this module
