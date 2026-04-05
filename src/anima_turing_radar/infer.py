"""CLI inference entrypoint for Turing-Radar deinterleaving."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ClusteringConfig, load_config
from .io import load_pulse_train_h5, save_predictions_json
from .model import build_deinterleaver
from .preprocess import PDWPreprocessor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deinterleaving inference on a pulse train H5 file")
    parser.add_argument("--input", required=True, help="Path to input pulse train .h5 file")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--algorithm", default=None, help="Override clustering algorithm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    record = load_pulse_train_h5(args.input)
    pre = PDWPreprocessor.from_config(cfg.preprocess)
    x = pre.fit_transform(record.data)

    clustering_cfg = cfg.clustering
    if args.algorithm:
        clustering_cfg = ClusteringConfig(
            algorithm=args.algorithm,
            default_label=cfg.clustering.default_label,
            min_cluster_size=cfg.clustering.min_cluster_size,
            min_samples=cfg.clustering.min_samples,
            eps=cfg.clustering.eps,
            kmeans_k=cfg.clustering.kmeans_k,
        )

    model = build_deinterleaver(clustering_cfg)
    labels = model.predict(x)

    output_path = Path(args.output) if args.output else Path(args.input).with_suffix(".predictions.json")
    save_predictions_json(
        output_path,
        labels,
        source_file=str(Path(args.input).resolve()),
        algorithm=clustering_cfg.algorithm,
    )

    print(f"Wrote predictions: {output_path}")
    print(f"Pulses: {labels.shape[0]} | Clusters: {len(set(labels.tolist()) - {-1})}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
