"""Optional embedding module for extension to learned deinterleaving."""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def torch_available() -> bool:
    return TORCH_AVAILABLE


if TORCH_AVAILABLE:

    class PulseEmbeddingNet(nn.Module):
        def __init__(self, in_dim: int = 5, hidden_dim: int = 64, out_dim: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    def as_numpy_embedder(model: nn.Module, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
        model = model.to(device)
        model.eval()

        def _embed(data: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.as_tensor(data, dtype=torch.float32, device=device)
                z = model(x)
            return z.cpu().numpy().astype(np.float32)

        return _embed

else:

    class PulseEmbeddingNet:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Torch is not installed; embedding network unavailable")


    def as_numpy_embedder(model, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:  # pragma: no cover
        raise RuntimeError("Torch is not installed; embedding adaptor unavailable")
