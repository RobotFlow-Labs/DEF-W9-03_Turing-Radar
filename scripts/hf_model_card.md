---
license: apache-2.0
tags:
  - radar
  - electronic-warfare
  - clustering
  - pulse-deinterleaving
  - anima
datasets:
  - alan-turing-institute/turing-deinterleaving-challenge
metrics:
  - v_measure
  - adjusted_rand_index
pipeline_tag: feature-extraction
---

# ANIMA Turing-Radar: Pulse Embedding for Deinterleaving

## Overview

Learned pulse embedding model for radar pulse deinterleaving on the
[Turing Synthetic Radar Dataset (TSRD)](https://huggingface.co/datasets/alan-turing-institute/turing-deinterleaving-challenge).

The model maps 5D Pulse Descriptor Words (PDW) to a 32-dimensional embedding space
where same-emitter pulses cluster together. Used as a preprocessing step before
HDBSCAN clustering for improved deinterleaving performance.

## Architecture

- **Input:** 5D PDW vector (ToA, Frequency, Pulse Width, AoA, Amplitude)
- **Network:** 3-layer MLP (5 → 128 → 128 → 32) with ReLU activations
- **Output:** 32D embedding vector
- **Training:** Triplet contrastive loss with margin=1.0
- **Parameters:** 21,408

## Usage

```python
import torch
from safetensors.torch import load_file

# Load model
state = load_file("model.safetensors")
model = torch.nn.Sequential(
    torch.nn.Linear(5, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 32),
)
model.load_state_dict(state, strict=False)
model.eval()

# Embed pulses
pulses = torch.randn(1000, 5)  # preprocessed PDW features
with torch.no_grad():
    embeddings = model(pulses)  # (1000, 32)

# Cluster with HDBSCAN
from sklearn.cluster import HDBSCAN
labels = HDBSCAN(min_cluster_size=20).fit_predict(embeddings.numpy())
```

## Available Formats

| Format | File | Use Case |
|--------|------|----------|
| PyTorch | `model.pth` | Training, fine-tuning |
| SafeTensors | `model.safetensors` | Fast loading, production |
| ONNX | `model.onnx` | Cross-platform inference |
| TensorRT FP16 | `model_fp16.engine` | NVIDIA GPU inference (fastest) |
| TensorRT FP32 | `model_fp32.engine` | NVIDIA GPU inference (precision) |

## Training Details

- **Dataset:** TSRD (alan-turing-institute/turing-deinterleaving-challenge)
- **Split:** 500 training files, 90/10 train/val
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler:** Warmup cosine (5% warmup)
- **Loss:** Triplet contrastive (margin=1.0)
- **Precision:** bf16 mixed precision
- **Hardware:** NVIDIA L4 (23GB)

## Paper

> Gunn et al., "The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving"
> arXiv:2602.03856

## License

Apache 2.0

## Citation

```bibtex
@article{gunn2026turing,
  title={The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving},
  author={Gunn, Edward and Hosford, Adam and Jones, Robert and Zeitler, Leo and Groves, Ian and Nockles, Victoria},
  journal={arXiv preprint arXiv:2602.03856},
  year={2026}
}
```

Built with ANIMA by [Robot Flow Labs](https://github.com/RobotFlow-Labs)
