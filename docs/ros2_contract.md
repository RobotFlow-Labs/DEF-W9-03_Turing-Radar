# ROS2 Contract (Draft)

## Node
- Name: `turing_radar_node`
- Package: `anima_turing_radar`

## Input Topic (planned)
- Topic: `/radar/pdws`
- Message: `std_msgs/Float32MultiArray`
- Shape convention: flattened `[N,5]` PDW features

## Output Topic (planned)
- Topic: `/radar/deinterleaved_labels`
- Message: `std_msgs/Int32MultiArray`
- Payload: `[N]` emitter cluster IDs for the same pulse order

## Service / Action Extensions (planned)
- `GetDeinterleavingStats` for summary metrics and runtime diagnostics

## Runtime Notes
- If `rclpy` is unavailable, module runs in fallback/no-ROS mode.
- CUDA optimization and ROS2 performance tuning are deferred to server migration phase.
