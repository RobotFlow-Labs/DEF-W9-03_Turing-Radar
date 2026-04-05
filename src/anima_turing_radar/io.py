"""IO and dataset iteration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass(slots=True)
class PulseTrainRecord:
    data: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_h5_group(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in group.attrs.items()}
    for key in group.keys():
        node = group[key]
        if isinstance(node, h5py.Group):
            out[key] = _read_h5_group(node)
        else:
            out[key] = node[()].tolist() if hasattr(node[()], "tolist") else node[()]
    return out


def load_pulse_train_h5(path: str | Path) -> PulseTrainRecord:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pulse train file not found: {file_path}")

    with h5py.File(file_path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"Invalid pulse train file, missing 'data': {file_path}")

        data = np.asarray(handle["data"], dtype=np.float32)
        labels = np.asarray(handle["labels"]).flatten() if "labels" in handle else None
        metadata = _read_h5_group(handle["metadata"]) if "metadata" in handle else {}

    if data.ndim != 2:
        raise ValueError(f"Expected 2D PDW data, got shape {data.shape}")
    return PulseTrainRecord(data=data, labels=labels, metadata=metadata)


def save_predictions_json(
    path: str | Path,
    labels: np.ndarray,
    *,
    source_file: str,
    algorithm: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "source_file": source_file,
        "algorithm": algorithm,
        "n_pulses": int(labels.shape[0]),
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "labels": labels.astype(int).tolist(),
    }
    if extra:
        payload["extra"] = extra

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _sort_key(path: Path) -> tuple[int, str]:
    parts = path.stem.split("_")
    if parts and parts[-1].isdigit():
        return (int(parts[-1]), path.name)
    return (10**9, path.name)


def iter_pulse_train_files(
    data_root: str | Path,
    *,
    subset: str | None = None,
    file_pattern: str = "*.h5",
) -> Iterator[Path]:
    root = Path(data_root)
    target = root / subset if subset else root
    if not target.exists():
        return iter(())

    for file_path in sorted(target.glob(file_pattern), key=_sort_key):
        yield file_path


def iter_windowed_pdws(
    record: PulseTrainRecord,
    *,
    window_length: int | None,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    if window_length is None:
        yield record.data, record.labels
        return

    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")

    n = record.data.shape[0]
    for start in range(0, n, window_length):
        end = min(start + window_length, n)
        labels = None if record.labels is None else record.labels[start:end]
        yield record.data[start:end], labels
