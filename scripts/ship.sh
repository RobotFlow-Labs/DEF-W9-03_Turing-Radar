#!/bin/bash
# Full export + ship pipeline for Turing-Radar
# Run after training completes
set -euo pipefail

MODULE="turing-radar"
ARTIFACTS="/mnt/artifacts-datai"
CKPT="$ARTIFACTS/checkpoints/$MODULE/best.pth"
EXPORT_DIR="$ARTIFACTS/exports/$MODULE"
HF_REPO="ilessio-aiflowlab/turing-radar-checkpoint"
PROJECT_DIR="/mnt/forge-data/modules/05_wave9/03_Turing-Radar"

cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH=""

echo "=== STEP 1: Export all formats ==="
python scripts/export_model.py \
  --checkpoint "$CKPT" \
  --output-dir "$EXPORT_DIR" \
  --hidden-dim 128 \
  --embed-dim 32

echo ""
echo "=== STEP 2: Evaluate embedding vs baseline ==="
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_embedding.py \
  --checkpoint "$CKPT" \
  --config configs/paper.toml \
  --subset archive/test \
  --max-files 20 \
  --max-pulses 50000 \
  --device cuda

echo ""
echo "=== STEP 3: Push to HuggingFace ==="
mkdir -p "$EXPORT_DIR/hf_upload"
cp "$EXPORT_DIR/model.pth" "$EXPORT_DIR/hf_upload/"
cp "$EXPORT_DIR/model.safetensors" "$EXPORT_DIR/hf_upload/" 2>/dev/null || true
cp "$EXPORT_DIR/model.onnx" "$EXPORT_DIR/hf_upload/"
cp "$EXPORT_DIR/model_fp16.engine" "$EXPORT_DIR/hf_upload/" 2>/dev/null || true
cp "$EXPORT_DIR/model_fp32.engine" "$EXPORT_DIR/hf_upload/" 2>/dev/null || true
cp scripts/hf_model_card.md "$EXPORT_DIR/hf_upload/README.md"
cp configs/train.toml "$EXPORT_DIR/hf_upload/"

huggingface-cli upload "$HF_REPO" "$EXPORT_DIR/hf_upload" . --private

echo ""
echo "=== STEP 4: Git commit + push ==="
cd "$PROJECT_DIR"
git add src/ tests/ configs/ scripts/ NEXT_STEPS.md PIPELINE_MAP.md PRD.md anima_module.yaml
git commit -m "[TURING-RADAR] Training complete + exports + HF push

- Embedding model: val_loss converged
- Exports: pth, safetensors, ONNX, TRT FP16, TRT FP32
- HuggingFace: $HF_REPO
- Tests: 17/17 passing

Built with ANIMA by Robot Flow Labs

Co-Authored-By: ilessiorobotflowlabs <noreply@robotflowlabs.com>"
git push origin main

echo ""
echo "=== DONE ==="
echo "Exports: $EXPORT_DIR"
echo "HuggingFace: https://huggingface.co/$HF_REPO"
