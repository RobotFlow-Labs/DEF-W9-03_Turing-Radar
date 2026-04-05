#!/usr/bin/env python3
"""Evaluate embedding+HDBSCAN vs baseline HDBSCAN on test files."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from anima_turing_radar.config import load_config
from anima_turing_radar.embedding import PulseEmbeddingNet, as_numpy_embedder
from anima_turing_radar.evaluate import evaluate_labels
from anima_turing_radar.io import iter_pulse_train_files, load_pulse_train_h5
from anima_turing_radar.model import SklearnClusterDeinterleaver, build_deinterleaver
from anima_turing_radar.preprocess import PDWPreprocessor

ARTIFACTS = "/mnt/artifacts-datai"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare embedding vs baseline HDBSCAN")
    p.add_argument(
        "--checkpoint",
        default=f"{ARTIFACTS}/checkpoints/turing-radar/best.pth",
        help="Embedding model checkpoint",
    )
    p.add_argument("--config", default=str(ROOT / "configs" / "paper.toml"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--subset", default=None)
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--max-pulses", type=int, default=50000, help="Skip files larger than this")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    data_root = args.data_root or cfg.data.data_root
    subset = args.subset or cfg.data.subset

    # Load embedding model
    model = PulseEmbeddingNet(in_dim=5, hidden_dim=args.hidden_dim, out_dim=args.embed_dim)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    embedder = as_numpy_embedder(model, device=args.device)
    print(f"[MODEL] Loaded embedding from {args.checkpoint}")

    # Build models
    baseline = SklearnClusterDeinterleaver(
        algorithm="hdbscan", min_cluster_size=20, min_samples=5
    )
    embedded = build_deinterleaver(cfg.clustering, embedder=embedder)

    pre = PDWPreprocessor.from_config(cfg.preprocess)

    files = list(iter_pulse_train_files(data_root, subset=subset, file_pattern="*.h5"))
    files = files[: args.max_files]

    baseline_scores = []
    embed_scores = []
    n_eval = 0

    for fpath in files:
        record = load_pulse_train_h5(fpath)
        if record.labels is None or record.data.shape[0] > args.max_pulses:
            continue

        x = pre.fit_transform(record.data)
        mask = np.isfinite(x).all(axis=1)
        x = x[mask]
        labels = record.labels[mask]

        if len(labels) == 0:
            continue

        n_emitters = len(np.unique(labels))

        # Baseline
        t0 = time.time()
        pred_base = baseline.predict(x)
        dt_base = time.time() - t0
        s_base = evaluate_labels(pred_base, labels)

        # Embedding
        t0 = time.time()
        pred_embed = embedded.predict(x)
        dt_embed = time.time() - t0
        s_embed = evaluate_labels(pred_embed, labels)

        baseline_scores.append(s_base)
        embed_scores.append(s_embed)
        n_eval += 1

        print(
            f"{fpath.name}: "
            f"base_V={s_base['V-measure']:.3f} embed_V={s_embed['V-measure']:.3f} "
            f"pulses={record.data.shape[0]} emitters={n_emitters} "
            f"dt_base={dt_base:.1f}s dt_embed={dt_embed:.1f}s"
        )

    if not baseline_scores:
        print("No files evaluated")
        return 1

    # Summary
    base_v = [s["V-measure"] for s in baseline_scores]
    embed_v = [s["V-measure"] for s in embed_scores]
    base_ari = [s["ARI"] for s in baseline_scores]
    embed_ari = [s["ARI"] for s in embed_scores]

    print(f"\n{'='*60}")
    print(f"RESULTS ({n_eval} files)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Embedding':>12} {'Delta':>10}")
    print(f"{'-'*60}")
    print(
        f"{'Median V-measure':<25} {median(base_v):>12.4f} {median(embed_v):>12.4f} "
        f"{median(embed_v)-median(base_v):>+10.4f}"
    )
    print(
        f"{'Mean V-measure':<25} {mean(base_v):>12.4f} {mean(embed_v):>12.4f} "
        f"{mean(embed_v)-mean(base_v):>+10.4f}"
    )
    print(
        f"{'Median ARI':<25} {median(base_ari):>12.4f} {median(embed_ari):>12.4f} "
        f"{median(embed_ari)-median(base_ari):>+10.4f}"
    )
    print(
        f"{'Mean ARI':<25} {mean(base_ari):>12.4f} {mean(embed_ari):>12.4f} "
        f"{mean(embed_ari)-mean(base_ari):>+10.4f}"
    )
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
