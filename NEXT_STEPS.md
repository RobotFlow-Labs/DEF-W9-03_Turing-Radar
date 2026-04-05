# Turing-Radar — CUDA Handoff Next Steps

## Completed in this workspace
- Generated full PRD suite (`prds/`) and granular tasks (`tasks/`)
- Implemented essential module code (`src/anima_turing_radar/`)
- Added baseline benchmark script, API, tests, and docs

## Immediate CUDA Server Tasks
1. Provision runtime on GPU host
- Install module in editable mode
- Install optional torch CUDA build and verify GPU visibility

2. Data pull and validation
- Download TSRD subsets to `/mnt/forge-data/datasets/turing_deinterleaving_challenge`
- Run sample-file integrity checks (shape, dtype, labels)

3. Throughput profiling
- Benchmark preprocessing + clustering latency by window size
- Compare CPU baseline vs GPU-accelerated embedding path (if enabled)

4. Model extension sprint
- Replace optional embedding stub with learned metric model
- Add checkpoint loading and reproducibility controls

5. Productionization
- Add ONNX export route for embedding model
- Add structured logs and telemetry for `/predict`
- Finalize ROS2 message bindings

## Acceptance Gates
- Median V-measure on validation subset meets or exceeds baseline
- End-to-end inference latency target is documented and reproducible
- API, CLI, benchmark script, and tests run green on server
