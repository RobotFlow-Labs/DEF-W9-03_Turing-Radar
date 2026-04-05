# Training Report — Turing-Radar Pulse Embedding

## Summary

| Field | Value |
|-------|-------|
| Module | turing-radar |
| Paper | arXiv:2602.03856 (Gunn et al., 2026) |
| Model | PulseEmbeddingNet (5→128→128→32) |
| Parameters | 21,408 |
| Task | Triplet contrastive metric learning |
| Best val_loss | **0.003803** (epoch 44) |
| Total epochs | 46 (early stopped at 47, patience=10) |
| Training time | 4.0 hours |
| GPU | NVIDIA L4 (23GB), GPU 5 |
| Precision | bf16 mixed |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| batch_size | 8192 |
| learning_rate | 0.001 |
| optimizer | AdamW |
| weight_decay | 0.01 |
| scheduler | Warmup cosine (5% warmup) |
| min_lr | 1e-6 |
| margin (triplet) | 1.0 |
| gradient_clip | 1.0 |
| seed | 42 |
| max_files | 500 (344 with labels) |
| train pulses | 7,044,704 |
| val pulses | 782,745 |

## Training Curve

| Epoch | train_loss | val_loss | LR |
|------:|----------:|--------:|--------:|
| 1 | 0.1271 | 0.0420 | 4.00e-4 |
| 5 | 0.0124 | 0.0116 | 9.93e-4 |
| 10 | 0.0072 | 0.0073 | 9.40e-4 |
| 15 | 0.0057 | 0.0058 | 8.39e-4 |
| 20 | 0.0050 | 0.0049 | 7.01e-4 |
| 25 | 0.0045 | 0.0045 | 5.41e-4 |
| 30 | 0.0042 | 0.0043 | 3.77e-4 |
| 35 | 0.0039 | 0.0040 | 2.27e-4 |
| 40 | 0.0037 | 0.0039 | 1.05e-4 |
| 44 | **0.0037** | **0.0038** | 3.89e-5 |
| 46 | 0.0038 | 0.0038 | 1.74e-5 |

## Baseline Comparison (HDBSCAN on test files)

| Model | Median V-measure | Notes |
|-------|-----------------|-------|
| HDBSCAN baseline (paper) | 0.538 | Paper Table, stare mode |
| HDBSCAN baseline (ours, 3-file subset) | 0.758 | Small files only |
| Single file (test_0.h5) | 0.894 | 29K pulses, 78 emitters |

## Export Formats

| Format | Size | Path |
|--------|------|------|
| PyTorch (.pth) | 86.2 KB | `/mnt/artifacts-datai/exports/turing-radar/model.pth` |
| SafeTensors | 84.1 KB | `/mnt/artifacts-datai/exports/turing-radar/model.safetensors` |
| ONNX (opset 17) | 84.4 KB | `/mnt/artifacts-datai/exports/turing-radar/model.onnx` |
| TensorRT FP16 | 155.5 KB | `/mnt/artifacts-datai/exports/turing-radar/model_fp16.engine` |
| TensorRT FP32 | 156.5 KB | `/mnt/artifacts-datai/exports/turing-radar/model_fp32.engine` |

## Hardware

- GPU: NVIDIA L4 (23GB VRAM), CUDA 12.x
- torch: 2.11.0+cu128
- TensorRT: 10.16.0.72
- VRAM used: 634 MiB (model is tiny)

## Artifacts

- Checkpoint: `/mnt/artifacts-datai/checkpoints/turing-radar/best.pth`
- History: `/mnt/artifacts-datai/logs/turing-radar/training_history.json`
- Training log: `/mnt/artifacts-datai/logs/turing-radar/train_20260405_0643.log`
- HuggingFace: `ilessio-aiflowlab/turing-radar-checkpoint`
