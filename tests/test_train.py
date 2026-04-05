"""Tests for training components."""

from __future__ import annotations

import numpy as np
import torch

from anima_turing_radar.train import (
    ContrastiveTripletLoss,
    EarlyStopping,
    PulsePairDataset,
    WarmupCosineScheduler,
)


def test_pulse_pair_dataset_returns_triplet() -> None:
    features = np.random.randn(100, 5).astype(np.float32)
    labels = np.repeat(np.arange(5), 20)
    ds = PulsePairDataset(features, labels)
    assert len(ds) == 100
    anchor, pos, neg = ds[0]
    assert anchor.shape == (5,)
    assert pos.shape == (5,)
    assert neg.shape == (5,)


def test_contrastive_triplet_loss_perfect() -> None:
    loss_fn = ContrastiveTripletLoss(margin=1.0)
    anchor = torch.zeros(8, 16)
    positive = torch.zeros(8, 16)
    negative = torch.ones(8, 16) * 5.0
    loss = loss_fn(anchor, positive, negative)
    assert loss.item() == 0.0  # d_pos=0, d_neg >> margin


def test_contrastive_triplet_loss_bad() -> None:
    loss_fn = ContrastiveTripletLoss(margin=1.0)
    anchor = torch.zeros(8, 16)
    positive = torch.ones(8, 16) * 5.0
    negative = torch.zeros(8, 16)
    loss = loss_fn(anchor, positive, negative)
    assert loss.item() > 0.0  # d_pos >> d_neg


def test_warmup_cosine_scheduler() -> None:
    model = torch.nn.Linear(5, 5)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Warmup: LRs should increase
    assert lrs[0] < lrs[9]
    # Cosine decay: LRs should decrease after warmup
    assert lrs[50] < lrs[10]
    # State dict roundtrip
    state = scheduler.state_dict()
    scheduler.load_state_dict(state)
    assert scheduler.current_step == 100


def test_early_stopping() -> None:
    es = EarlyStopping(patience=3, min_delta=0.01)
    assert not es.step(1.0)  # new best
    assert not es.step(0.95)  # improved
    assert not es.step(0.95)  # no improve, counter=1
    assert not es.step(0.95)  # counter=2
    assert es.step(0.95)  # counter=3 -> stop
