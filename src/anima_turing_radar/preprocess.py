"""Preprocessing for PDW features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PreprocessConfig


@dataclass(slots=True)
class PDWPreprocessor:
    sort_by_toa: bool = True
    delta_toa: bool = True
    standardize: bool = True
    clip_std: float | None = 8.0

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: PreprocessConfig) -> "PDWPreprocessor":
        return cls(
            sort_by_toa=cfg.sort_by_toa,
            delta_toa=cfg.delta_toa,
            standardize=cfg.standardize,
            clip_std=cfg.clip_std,
        )

    def _prepare(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=np.float32).copy()
        if x.ndim != 2:
            raise ValueError(f"Expected 2D PDW array, got shape {x.shape}")

        if self.sort_by_toa:
            order = np.argsort(x[:, 0], kind="stable")
            x = x[order]

        if self.delta_toa:
            toa = x[:, 0].copy()
            x[:, 0] = np.diff(toa, prepend=toa[:1])

        return x

    def fit(self, data: np.ndarray) -> "PDWPreprocessor":
        x = self._prepare(data)
        if self.standardize:
            self.mean_ = x.mean(axis=0)
            std = x.std(axis=0)
            self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        x = self._prepare(data)
        if self.standardize:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Preprocessor must be fitted before transform")
            x = (x - self.mean_) / self.std_

        if self.clip_std is not None:
            x = np.clip(x, -self.clip_std, self.clip_std)

        return x.astype(np.float32, copy=False)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
