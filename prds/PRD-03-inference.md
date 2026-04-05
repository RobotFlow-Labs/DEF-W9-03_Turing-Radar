# PRD-03: Inference Pipeline and CLI

> Module: Turing-Radar | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Provide a runnable local inference path that loads pulse trains, preprocesses PDWs, predicts cluster IDs, and writes outputs.

## Context (from paper)
Pulse trains are variable-length and emitted labels are file-local; inference must preserve per-file context and avoid cross-file label assumptions.

Paper references:
- Section II.A (pulse train composition)
- Section III.A (unknown emitters at test time)

## Acceptance Criteria
- [ ] `python3 -m anima_turing_radar.infer` supports H5 input and JSON output.
- [ ] Inference outputs include labels, cluster count, and metadata.
- [ ] CLI allows config path override and algorithm selection.
- [ ] Inference smoke tests pass on synthetic pulse train.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_turing_radar/infer.py` | CLI entrypoint for deinterleaving inference | §II.A, §III.A | ~180 |
| `tests/test_infer.py` | End-to-end CLI smoke test | — | ~80 |

## Test Plan
```bash
python3 -m pytest tests/test_infer.py -v
python3 -m anima_turing_radar.infer --help
```

## References
- Paper: §II.A, §III.A
- Reference repo: `examples/identity_model.ipynb`
- Feeds into: PRD-04, PRD-05
