#!/usr/bin/env python3
"""Run baseline clustering benchmark over local H5 pulse train files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anima_turing_radar.config import load_config
from anima_turing_radar.evaluate import aggregate_scores, evaluate_dataset
from anima_turing_radar.io import iter_pulse_train_files, load_pulse_train_h5
from anima_turing_radar.model import build_deinterleaver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark baseline deinterleaver on local dataset")
    p.add_argument("--config", default=str(ROOT / "configs" / "paper.toml"), help="Config file path")
    p.add_argument("--data-root", default=None, help="Override data root")
    p.add_argument("--subset", default=None, help="Override subset")
    p.add_argument("--max-files", type=int, default=10, help="Max files to evaluate")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    data_root = args.data_root or cfg.data.data_root
    subset = args.subset or cfg.data.subset

    files = list(iter_pulse_train_files(data_root, subset=subset, file_pattern=cfg.data.file_pattern))
    files = files[: args.max_files]
    if not files:
        print(f"No files found under {Path(data_root) / subset}")
        return 1

    records = [load_pulse_train_h5(path) for path in files]
    records = [r for r in records if r.labels is not None]
    if not records:
        print("No labeled records found; cannot run benchmark")
        return 1

    model = build_deinterleaver(cfg.clustering)
    scores = evaluate_dataset(model, records)
    summary = aggregate_scores(scores)

    print("Benchmark Summary")
    print(f"files={len(records)} subset={subset} algorithm={cfg.clustering.algorithm}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
