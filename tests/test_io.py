from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from anima_turing_radar.io import load_pulse_train_h5, save_predictions_json


def test_load_h5_with_labels(tmp_path: Path) -> None:
    p = tmp_path / "sample.h5"
    with h5py.File(p, "w") as h5:
        h5.create_dataset("data", data=np.random.randn(10, 5).astype(np.float32))
        h5.create_dataset("labels", data=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        md = h5.create_group("metadata")
        md.attrs["receiver"] = "stare"

    rec = load_pulse_train_h5(p)
    assert rec.data.shape == (10, 5)
    assert rec.labels is not None
    assert rec.metadata["receiver"] == "stare"


def test_save_predictions_json(tmp_path: Path) -> None:
    out = save_predictions_json(
        tmp_path / "pred.json",
        np.array([0, 1, 1, -1]),
        source_file="x.h5",
        algorithm="hdbscan",
    )
    assert out.exists()
