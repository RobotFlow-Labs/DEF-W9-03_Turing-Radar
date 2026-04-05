# PRD-05: API and Container Serving

> Module: Turing-Radar | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose inference through a minimal FastAPI service and provide docker assets for local/remote serving.

## Context (from paper)
Operational deinterleaving needs repeatable interfaces for downstream EW pipelines; API service offers that bridge without changing core clustering logic.

Paper references:
- Section I (operational relevance)
- Section III (community benchmark infrastructure)

## Acceptance Criteria
- [ ] FastAPI app provides `/health`, `/ready`, `/predict`.
- [ ] `/predict` accepts raw PDW arrays and returns emitter labels.
- [ ] `docker/Dockerfile.serve` and `docker/docker-compose.serve.yml` are present.
- [ ] API tests pass.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_turing_radar/api.py` | FastAPI serving layer | §I, §III | ~180 |
| `docker/Dockerfile.serve` | Containerized service | — | ~40 |
| `docker/docker-compose.serve.yml` | Local service profile | — | ~30 |
| `tests/test_api.py` | Endpoint tests | — | ~70 |

## Test Plan
```bash
python3 -m pytest tests/test_api.py -v
```

## References
- Paper: §I, §III
- Feeds into: PRD-06, PRD-07
