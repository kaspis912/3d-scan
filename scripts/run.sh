#!/usr/bin/env bash
# Start the scan3d web service at http://localhost:${PORT:-8000}
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --extra masking --extra silhouette uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
