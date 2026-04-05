from __future__ import annotations

import numpy as np

from anima_turing_radar.evaluate import aggregate_scores, evaluate_labels


def test_evaluate_labels_perfect() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    scores = evaluate_labels(y, y)
    assert scores["V-measure"] > 0.99
    assert scores["ARI"] > 0.99


def test_aggregate_scores() -> None:
    rows = [
        {"V-measure": 0.5, "ARI": 0.4, "AMI": 0.45, "Homogeneity": 0.5, "Completeness": 0.5, "MCC": 0.1, "F1": 0.1, "discount": 1.0},
        {"V-measure": 0.7, "ARI": 0.6, "AMI": 0.65, "Homogeneity": 0.7, "Completeness": 0.7, "MCC": 0.2, "F1": 0.2, "discount": 1.0},
    ]
    summary = aggregate_scores(rows)
    assert "mean_V-measure" in summary
    assert "median_ARI" in summary
