#!/usr/bin/env bash
# Export a trained run to ONNX. Pass the run dir as $1 (the one containing config.json).
#   ./scripts/03_export.sh runs/tamil_female-July-01-2026_10+00AM
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:?usage: 03_export.sh <run_dir>}"
uv run python -m tamiltts.export_onnx --run "$RUN_DIR" --out models/tamil_female.onnx
