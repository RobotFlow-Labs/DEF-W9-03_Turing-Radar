#!/usr/bin/env python3
"""Export trained embedding model to multiple formats: safetensors, ONNX, TensorRT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from anima_turing_radar.embedding import PulseEmbeddingNet

PROJECT = "turing-radar"
ARTIFACTS = "/mnt/artifacts-datai"
EXPORT_DIR = Path(f"{ARTIFACTS}/exports/{PROJECT}")


def export_safetensors(model: PulseEmbeddingNet, output_dir: Path) -> Path:
    from safetensors.torch import save_file

    path = output_dir / "model.safetensors"
    save_file(model.state_dict(), str(path))
    print(f"[EXPORT] safetensors: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def export_onnx(model: PulseEmbeddingNet, output_dir: Path, device: str = "cpu") -> Path:
    path = output_dir / "model.onnx"
    dummy = torch.randn(1, 5, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["pulses"],
        output_names=["embeddings"],
        dynamic_axes={"pulses": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
    )
    print(f"[EXPORT] ONNX: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def export_tensorrt(onnx_path: Path, output_dir: Path, precision: str = "fp16") -> Path:
    """Export ONNX to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        # Fallback: use shared TRT toolkit
        trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
        if trt_script.exists():
            import subprocess

            trt_path = output_dir / f"model_{precision}.engine"
            cmd = [
                sys.executable,
                str(trt_script),
                "--onnx",
                str(onnx_path),
                "--output",
                str(trt_path),
                "--precision",
                precision,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[EXPORT] TensorRT {precision}: {trt_path}")
                return trt_path
            else:
                print(f"[WARN] TRT export failed: {result.stderr[:200]}")
                return _trt_trtexec_fallback(onnx_path, output_dir, precision)
        return _trt_trtexec_fallback(onnx_path, output_dir, precision)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT ERROR] {parser.get_error(i)}")
            raise RuntimeError("TRT ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("pulses", (1, 5), (1024, 5), (65536, 5))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TRT engine build failed")

    trt_path = output_dir / f"model_{precision}.engine"
    with open(trt_path, "wb") as f:
        f.write(engine)
    print(f"[EXPORT] TensorRT {precision}: {trt_path} ({trt_path.stat().st_size / 1024:.1f} KB)")
    return trt_path


def _trt_trtexec_fallback(onnx_path: Path, output_dir: Path, precision: str) -> Path:
    """Fallback: use trtexec CLI."""
    import shutil
    import subprocess

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print(f"[SKIP] TensorRT {precision}: trtexec not found")
        return Path()

    trt_path = output_dir / f"model_{precision}.engine"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--minShapes=pulses:1x5",
        "--optShapes=pulses:1024x5",
        "--maxShapes=pulses:65536x5",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print(f"[EXPORT] TensorRT {precision} (trtexec): {trt_path}")
        return trt_path
    else:
        print(f"[SKIP] TensorRT {precision}: trtexec failed")
        print(f"  stderr: {result.stderr[:300]}")
        return Path()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export embedding model to all formats")
    parser.add_argument(
        "--checkpoint",
        default=f"{ARTIFACTS}/checkpoints/{PROJECT}/best.pth",
        help="Path to best.pth checkpoint",
    )
    parser.add_argument("--output-dir", default=str(EXPORT_DIR), help="Export output directory")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT exports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return 1

    model = PulseEmbeddingNet(in_dim=5, hidden_dim=args.hidden_dim, out_dim=args.embed_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[LOAD] Model from {ckpt_path}")

    # Save raw pth
    pth_path = output_dir / "model.pth"
    torch.save(ckpt["model"], pth_path)
    print(f"[EXPORT] pth: {pth_path} ({pth_path.stat().st_size / 1024:.1f} KB)")

    # safetensors
    try:
        export_safetensors(model, output_dir)
    except ImportError:
        print("[SKIP] safetensors: package not installed")

    # ONNX
    onnx_path = export_onnx(model, output_dir)

    # TensorRT FP16 + FP32
    if not args.skip_trt:
        export_tensorrt(onnx_path, output_dir, "fp16")
        export_tensorrt(onnx_path, output_dir, "fp32")
    else:
        print("[SKIP] TensorRT exports (--skip-trt)")

    print(f"\n[DONE] All exports in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
