from __future__ import annotations

from pathlib import Path

from anima_turing_radar.config import AppConfig, load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.clustering.algorithm == "hdbscan"


def test_config_override(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("""
[clustering]
algorithm = "kmeans"
kmeans_k = 5
""", encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.clustering.algorithm == "kmeans"
    assert cfg.clustering.kmeans_k == 5
