"""Core deinterleaver interfaces and baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from sklearn import cluster

from .config import ClusteringConfig


class Deinterleaver(Protocol):
    default_label: int | None

    def predict(self, data: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class SklearnClusterDeinterleaver:
    algorithm: str = "hdbscan"
    default_label: int | None = -1
    min_cluster_size: int = 20
    min_samples: int | None = 5
    eps: float = 0.0
    kmeans_k: int = 8

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.lower()

    def _build_clusterer(self, n_samples: int):
        algo = self.algorithm

        if algo == "hdbscan":
            hdbscan_cls = getattr(cluster, "HDBSCAN", None)
            if hdbscan_cls is not None:
                kwargs = {
                    "min_cluster_size": self.min_cluster_size,
                    "cluster_selection_epsilon": self.eps,
                }
                if self.min_samples is not None:
                    kwargs["min_samples"] = self.min_samples
                return hdbscan_cls(**kwargs)
            # Fallback when sklearn does not ship HDBSCAN in current version.
            return cluster.DBSCAN(eps=max(self.eps, 0.5), min_samples=max(self.min_samples or 5, 2))

        if algo == "dbscan":
            return cluster.DBSCAN(eps=max(self.eps, 0.5), min_samples=max(self.min_samples or 5, 2))

        if algo == "kmeans":
            k = min(max(2, self.kmeans_k), max(2, n_samples))
            return cluster.KMeans(n_clusters=k, n_init=10, random_state=7)

        if algo == "agglomerative":
            k = min(max(2, self.kmeans_k), max(2, n_samples))
            return cluster.AgglomerativeClustering(n_clusters=k)

        if algo == "optics":
            return cluster.OPTICS(min_samples=max(self.min_samples or 5, 2))

        raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        if x.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        clusterer = self._build_clusterer(x.shape[0])
        if hasattr(clusterer, "fit_predict"):
            labels = clusterer.fit_predict(x)
        else:  # pragma: no cover
            clusterer.fit(x)
            labels = getattr(clusterer, "labels_")
        return np.asarray(labels, dtype=np.int64)


@dataclass(slots=True)
class EmbeddedClusterDeinterleaver(SklearnClusterDeinterleaver):
    embedder: Callable[[np.ndarray], np.ndarray] | None = None

    def predict(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=np.float32)
        z = self.embedder(x) if self.embedder is not None else x
        return SklearnClusterDeinterleaver.predict(self, z)


def build_deinterleaver(
    cfg: ClusteringConfig,
    *,
    embedder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Deinterleaver:
    cls = EmbeddedClusterDeinterleaver if embedder is not None else SklearnClusterDeinterleaver
    return cls(
        algorithm=cfg.algorithm,
        default_label=cfg.default_label,
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        eps=cfg.eps,
        kmeans_k=cfg.kmeans_k,
        **({"embedder": embedder} if embedder is not None else {}),
    )
