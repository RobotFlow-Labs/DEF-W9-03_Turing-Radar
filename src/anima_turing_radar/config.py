"""Configuration models and loader utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(slots=True)
class DataConfig:
    data_root: str = "data"
    subset: str = "test"
    window_length: int | None = 50000
    min_emitters: int | None = 2
    max_emitters: int | None = 110
    file_pattern: str = "*.h5"


@dataclass(slots=True)
class PreprocessConfig:
    sort_by_toa: bool = True
    delta_toa: bool = True
    standardize: bool = True
    clip_std: float = 8.0


@dataclass(slots=True)
class ClusteringConfig:
    algorithm: str = "hdbscan"
    default_label: int | None = -1
    min_cluster_size: int = 20
    min_samples: int | None = 5
    eps: float = 0.0
    kmeans_k: int = 8


@dataclass(slots=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_app_config(mapping: dict[str, Any]) -> AppConfig:
    data = DataConfig(**mapping.get("data", {}))
    preprocess = PreprocessConfig(**mapping.get("preprocess", {}))
    clustering = ClusteringConfig(**mapping.get("clustering", {}))
    service = ServiceConfig(**mapping.get("service", {}))
    return AppConfig(data=data, preprocess=preprocess, clustering=clustering, service=service)


def default_config_dict() -> dict[str, Any]:
    return {
        "data": asdict(DataConfig()),
        "preprocess": asdict(PreprocessConfig()),
        "clustering": asdict(ClusteringConfig()),
        "service": asdict(ServiceConfig()),
    }


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from TOML path and merge with defaults."""
    if path is None:
        return AppConfig()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("rb") as handle:
        loaded = tomllib.load(handle)

    merged = _merge_dict(default_config_dict(), loaded)
    return _as_app_config(merged)
