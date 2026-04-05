# PRD-06: ROS2 Integration

> Module: Turing-Radar | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Provide a ROS2 integration skeleton that can consume PDW batches and publish deinterleaving results in ANIMA environments.

## Context (from paper)
The paper is dataset-centric, but deployment scenarios in EW/sensing stacks require message-based runtime integration.

Paper references:
- Section I (EW downstream applications)

## Acceptance Criteria
- [ ] ROS2 node skeleton supports `process_pdws` path.
- [ ] Node can run in environments where `rclpy` is available.
- [ ] Graceful fallback message is provided when ROS2 is unavailable.
- [ ] Interface contract documented.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---:|
| `src/anima_turing_radar/ros2_node.py` | ROS2 bridge skeleton | §I | ~120 |
| `docs/ros2_contract.md` | Topic/message contract draft | — | ~80 |

## Test Plan
```bash
python3 -m anima_turing_radar.ros2_node --help
```

## References
- Paper: §I
- Feeds into: PRD-07
