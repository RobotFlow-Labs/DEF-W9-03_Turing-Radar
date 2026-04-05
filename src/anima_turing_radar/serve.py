"""ANIMA serve integration for Turing-Radar module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_config
from .model import SklearnClusterDeinterleaver
from .preprocess import PDWPreprocessor

try:
    from anima_serve.node import AnimaNode

    ANIMA_SERVE_AVAILABLE = True
except ImportError:
    ANIMA_SERVE_AVAILABLE = False

    class AnimaNode:  # type: ignore
        """Stub when anima_serve is not installed."""

        def __init__(self, *args, **kwargs):
            pass


class TuringRadarNode(AnimaNode):
    """ANIMA serving node for radar pulse deinterleaving."""

    def setup_inference(self) -> None:
        cfg = load_config()
        self.preprocessor = PDWPreprocessor.from_config(cfg.preprocess)
        self.model = SklearnClusterDeinterleaver(
            algorithm=cfg.clustering.algorithm,
            min_cluster_size=cfg.clustering.min_cluster_size,
            min_samples=cfg.clustering.min_samples,
        )
        # Optionally load embedding model if weights exist
        self.embedder = None
        weights_dir = Path("/data/weights")
        best_pth = weights_dir / "best.pth"
        if best_pth.exists():
            try:
                import torch

                from .embedding import PulseEmbeddingNet, as_numpy_embedder

                net = PulseEmbeddingNet(in_dim=5, hidden_dim=128, out_dim=32)
                ckpt = torch.load(best_pth, map_location="cpu", weights_only=False)
                net.load_state_dict(ckpt["model"])
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedder = as_numpy_embedder(net, device=device)
            except Exception:
                pass

    def process(self, input_data: dict) -> dict:
        pulses = np.asarray(input_data.get("pulses", []), dtype=np.float32)
        if pulses.ndim != 2 or pulses.shape[1] != 5:
            return {"error": f"Expected (N,5) pulses, got {pulses.shape}"}

        x = self.preprocessor.fit_transform(pulses)

        if self.embedder is not None:
            x = self.embedder(x)

        labels = self.model.predict(x)
        n_clusters = len(set(labels.tolist()) - {-1})

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_pulses": len(labels),
        }

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "has_embedding": self.embedder is not None,
        }
