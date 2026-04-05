from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from anima_turing_radar.infer import main


def test_infer_cli_smoke(tmp_path: Path) -> None:
    in_path = tmp_path / "in.h5"
    out_path = tmp_path / "out.json"

    with h5py.File(in_path, "w") as h5:
        h5.create_dataset("data", data=np.random.randn(32, 5).astype(np.float32))

    rc = main(["--input", str(in_path), "--output", str(out_path), "--algorithm", "kmeans"])
    assert rc == 0
    assert out_path.exists()
