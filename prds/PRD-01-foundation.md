# PRD-01: Foundation and Data Contracts

> Module: Turing-Radar | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Create a reproducible ANIMA-ready Python package with explicit PDW contracts, config loading, HDF5 IO, and basic tests.

## Context (from paper)
The paper frames deinterleaving as clustering over variable-length pulse trains of 5D PDWs and unknown emitter counts.

Paper references:
- Section I (problem framing)
- Section II.A (dataset properties)
- Table I/II/III (scale and PDW statistics)

## Acceptance Criteria
- [ ] Project package initializes and imports cleanly.
- [ ] Config profiles load from TOML (`default` and `paper`).
- [ ] HDF5 pulse train IO supports `data` + optional `labels`.
- [ ] Preprocessing pipeline supports ToA sorting, delta-ToA option, and standardization.
- [ ] Tests for config and data contracts pass.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `pyproject.toml` | Packaging and dependencies | §II | ~80 |
| `src/anima_turing_radar/constants.py` | PDW feature constants | §II.A | ~40 |
| `src/anima_turing_radar/config.py` | Typed config loader | §II | ~180 |
| `src/anima_turing_radar/io.py` | H5 load/save utilities | §II.A | ~140 |
| `src/anima_turing_radar/preprocess.py` | Input preprocessing | §II.A | ~140 |
| `configs/default.toml` | Local defaults | §II | ~40 |
| `configs/paper.toml` | Paper/challenge baseline profile | §II, §III | ~40 |
| `tests/test_config.py` | Config tests | — | ~50 |
| `tests/test_io.py` | H5 IO tests | — | ~70 |
| `tests/test_preprocess.py` | Preprocess tests | — | ~70 |

## Test Plan
```bash
python3 -m pytest tests/test_config.py tests/test_io.py tests/test_preprocess.py -v
```

## References
- Paper: §I, §II.A, Table I-III
- Reference repo: `src/turing_deinterleaving_challenge/data/*`
- Feeds into: PRD-02
