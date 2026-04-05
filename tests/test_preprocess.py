from __future__ import annotations

import numpy as np

from anima_turing_radar.preprocess import PDWPreprocessor


def test_preprocess_shape_and_sort() -> None:
    x = np.array(
        [
            [3.0, 10.0, 1.0, 0.0, -30.0],
            [1.0, 11.0, 1.1, 0.1, -29.0],
            [2.0, 10.5, 1.2, -0.1, -31.0],
        ],
        dtype=np.float32,
    )
    pre = PDWPreprocessor(sort_by_toa=True, delta_toa=True, standardize=False)
    z = pre.fit_transform(x)
    assert z.shape == x.shape
    assert z[0, 0] == 0.0


def test_standardize() -> None:
    x = np.random.randn(100, 5).astype(np.float32)
    pre = PDWPreprocessor(sort_by_toa=False, delta_toa=False, standardize=True)
    z = pre.fit_transform(x)
    assert z.shape == x.shape
    assert np.all(np.isfinite(z))
