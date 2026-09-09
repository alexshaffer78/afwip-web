#!/usr/bin/env bash
# Bundle the Python engine (python/afwip) into public/afwip.zip so Pyodide can
# unpack it into its virtual filesystem at runtime. Run automatically before
# every build (package.json "prebuild").
set -euo pipefail
cd "$(dirname "$0")/.."

# Drop stale caches so they never end up in the zip.
find python/afwip -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

rm -f public/afwip.zip
( cd python && zip -q -r ../public/afwip.zip afwip -x '*.pyc' -x '*__pycache__*' )
echo "[pack-python] wrote public/afwip.zip ($(du -h public/afwip.zip | cut -f1))"
