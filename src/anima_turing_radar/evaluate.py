"""Evaluation helpers for deinterleaving clustering."""

from __future__ import annotations

from statistics import mean, median
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    matthews_corrcoef,
    v_measure_score,
)

from .io import PulseTrainRecord
from .model import Deinterleaver


def _cluster_wise_score(target: np.ndarray, pred: np.ndarray, score: str = "mcc") -> float:
    score_fn = matthews_corrcoef if score == "mcc" else f1_score
    per_cluster = []
    for t in np.unique(target):
        t_mask = (target == t).astype(int)
        best = -1.0
        for c in np.unique(pred):
            c_mask = (pred == c).astype(int)
            val = score_fn(c_mask, t_mask)
            if val > best:
                best = val
        per_cluster.append(best)
    return float(min(per_cluster)) if per_cluster else 0.0


def evaluate_labels(labels_pred: np.ndarray, labels_true: np.ndarray, predict_ratio: float = 1.0) -> dict[str, float]:
    y_pred = np.asarray(labels_pred).reshape(-1)
    y_true = np.asarray(labels_true).reshape(-1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {y_pred.shape} vs {y_true.shape}")

    penalty = float(predict_ratio)
    return {
        "Homogeneity": penalty * homogeneity_score(y_true, y_pred),
        "Completeness": penalty * completeness_score(y_true, y_pred),
        "V-measure": penalty * v_measure_score(y_true, y_pred),
        "ARI": penalty * adjusted_rand_score(y_true, y_pred),
        "AMI": penalty * adjusted_mutual_info_score(y_true, y_pred),
        "MCC": penalty * _cluster_wise_score(y_true, y_pred, score="mcc"),
        "F1": penalty * _cluster_wise_score(y_true, y_pred, score="f1"),
        "discount": penalty,
    }


def evaluate_record(model: Deinterleaver, record: PulseTrainRecord) -> dict[str, float]:
    if record.labels is None:
        raise ValueError("Record has no labels; cannot evaluate")

    pred = model.predict(record.data)
    labels = record.labels.reshape(-1)

    if model.default_label is None:
        return evaluate_labels(pred, labels)

    mask = labels != model.default_label
    ratio = float(mask.sum()) / float(mask.shape[0]) if mask.shape[0] else 0.0
    return evaluate_labels(pred[mask], labels[mask], predict_ratio=ratio)


def evaluate_dataset(model: Deinterleaver, records: Iterable[PulseTrainRecord]) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for record in records:
        if record.labels is None:
            continue
        scores.append(evaluate_record(model, record))
    return scores


def aggregate_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}

    keys = list(scores[0].keys())
    summary: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in scores]
        summary[f"mean_{key}"] = float(mean(vals))
        summary[f"median_{key}"] = float(median(vals))
    return summary
