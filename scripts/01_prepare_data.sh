#!/usr/bin/env bash
# Prepare the female-speaker subset of IndicTTS Tamil into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m tamiltts.data.prepare --speaker female --out data "$@"
