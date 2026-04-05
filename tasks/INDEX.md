# Turing-Radar Task Index — 17 Tasks

## Build Order

| Task | Title | Depends | Status |
|---|---|---|---|
| PRD-0101 | Package and config scaffold | None | ⬜ |
| PRD-0102 | H5 pulse train IO contracts | PRD-0101 | ⬜ |
| PRD-0103 | PDW preprocessing pipeline | PRD-0102 | ⬜ |
| PRD-0104 | Dataset windowing and iterators | PRD-0103 | ⬜ |
| PRD-0201 | Sklearn deinterleaver interfaces | PRD-0104 | ⬜ |
| PRD-0202 | Optional embedding module hooks | PRD-0201 | ⬜ |
| PRD-0203 | Core model tests | PRD-0201, PRD-0202 | ⬜ |
| PRD-0301 | Inference CLI | PRD-0203 | ⬜ |
| PRD-0302 | Prediction serialization and metadata | PRD-0301 | ⬜ |
| PRD-0401 | Label-level metric library | PRD-0302 | ⬜ |
| PRD-0402 | Dataset/batch evaluator | PRD-0401 | ⬜ |
| PRD-0501 | FastAPI app + request model | PRD-0302 | ⬜ |
| PRD-0502 | Docker serving assets | PRD-0501 | ⬜ |
| PRD-0601 | ROS2 node skeleton | PRD-0501 | ⬜ |
| PRD-0602 | ROS2 contract docs | PRD-0601 | ⬜ |
| PRD-0701 | Baseline benchmark script | PRD-0402 | ⬜ |
| PRD-0702 | CUDA handoff docs and runbook | PRD-0701 | ⬜ |
