#!/usr/bin/env bash
# Packs the project into dist/scan3d.zip for copying to another computer (e.g. a Windows PC).
# Leaves out the Python environment, scan data and caches: the installer recreates those.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/scan3d.zip
zip -rq dist/scan3d.zip . \
  -x '.venv/*' 'data/*' 'dist/*' '.claude/*' '.git/*' '*/__pycache__/*' '*.pyc' '.DS_Store' '*/.DS_Store' '*.log'
echo "Wrote dist/scan3d.zip ($(du -h dist/scan3d.zip | cut -f1))"
