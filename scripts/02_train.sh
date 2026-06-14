#!/usr/bin/env bash
# Train the Tamil VITS voice (run on the M1 Studio for speed).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python -m tamiltts.train --config configs/tamil_female_vits.json "$@"
