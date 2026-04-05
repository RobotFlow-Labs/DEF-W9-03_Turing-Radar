# PRD-07: Production Hardening and CUDA Handoff

> Module: Turing-Radar | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
Prepare this local baseline for CUDA-server migration with explicit optimization hooks, validation gates, and handoff docs.

## Context (from paper)
Computational efficiency is a highlighted research direction; practical deployment requires deterministic behavior, profiling, and clear migration steps.

Paper references:
- Section III (Computational efficiency outlook)

## Acceptance Criteria
- [ ] Baseline benchmark script for scan/stare subsets exists.
- [ ] Export hooks and TODO points for TensorRT/ONNX migration are documented.
- [ ] `NEXT_STEPS.md` captures CUDA-server tasks and acceptance gates.
- [ ] End-to-end sanity run command documented.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `scripts/benchmark_baseline.py` | Local benchmark harness | §III | ~140 |
| `NEXT_STEPS.md` | CUDA handoff and backlog | §III | ~120 |
| `README.md` | Module runbook | — | ~180 |

## Test Plan
```bash
python3 scripts/benchmark_baseline.py --help
```

## References
- Paper: §III
- Feeds into: CUDA optimization sprint
