from __future__ import annotations

import numpy as np
import pytest

from anima_turing_radar.model import SklearnClusterDeinterleaver


def make_clusters(n: int = 80) -> np.ndarray:
    a = np.random.normal(loc=-2.0, scale=0.3, size=(n, 5))
    b = np.random.normal(loc=2.0, scale=0.3, size=(n, 5))
    return np.concatenate([a, b], axis=0).astype(np.float32)


def test_kmeans_predict_shape() -> None:
    x = make_clusters()
    model = SklearnClusterDeinterleaver(algorithm="kmeans", kmeans_k=2)
    y = model.predict(x)
    assert y.shape == (x.shape[0],)


def test_invalid_algorithm_raises() -> None:
    x = make_clusters(10)
    model = SklearnClusterDeinterleaver(algorithm="unknown")
    with pytest.raises(ValueError):
        _ = model.predict(x)
