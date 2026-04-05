# PRD-04: Evaluation and Benchmarking

> Module: Turing-Radar | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement challenge-compatible clustering metrics and dataset-level aggregation for reproducible benchmarking.

## Context (from paper)
Challenge ranking emphasizes V-measure while complementary clustering quality metrics are required for robust analysis.

Paper references:
- Section III.A (Turing Deinterleaving Challenge)
- Abstract (V-measure emphasis)

## Acceptance Criteria
- [ ] `evaluate_labels` computes Homogeneity, Completeness, V-measure, ARI, AMI, MCC, F1.
- [ ] Batch evaluator supports multiple pulse trains.
- [ ] Optional penalty/discount for `default_label` filtering is implemented.
- [ ] Evaluation tests include perfect and imperfect clustering cases.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_turing_radar/evaluate.py` | Metrics and evaluators | §III.A | ~200 |
| `tests/test_evaluate.py` | Metric correctness tests | — | ~90 |

## Test Plan
```bash
python3 -m pytest tests/test_evaluate.py -v
```

## References
- Paper: §III.A
- Reference repo: `src/turing_deinterleaving_challenge/models/evaluate.py`
- Feeds into: PRD-07
