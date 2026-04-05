# PRD-02: Core Deinterleaver Models

> Module: Turing-Radar | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement essential deinterleaving model interfaces with a robust challenge baseline and extension-ready embedding path.

## Context (from paper)
The challenge is clustering-centric: unknown emitter cardinality and heavy feature overlap require unsupervised grouping quality measured by V-measure.

Paper references:
- Section II.B (challenge framing)
- Section III.A (benchmark objective)

## Acceptance Criteria
- [ ] Abstract `Deinterleaver` interface exists.
- [ ] HDBSCAN/DBSCAN/KMeans wrappers run on PDW arrays.
- [ ] Default baseline reproduces challenge-style behavior (`default_label` support).
- [ ] Optional embedding+clustering wrapper exists for future CUDA model insertion.
- [ ] Core model tests pass on synthetic clustered data.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_turing_radar/model.py` | Deinterleaver abstractions + sklearn implementations | §III.A | ~220 |
| `src/anima_turing_radar/embedding.py` | Optional torch embedding module | §III (feature extraction outlook) | ~140 |
| `tests/test_model.py` | Synthetic clustering correctness tests | — | ~100 |

## Test Plan
```bash
python3 -m pytest tests/test_model.py -v
```

## References
- Paper: §II.B, §III.A
- Reference repo: `src/turing_deinterleaving_challenge/models/model.py`
- Feeds into: PRD-03, PRD-04
