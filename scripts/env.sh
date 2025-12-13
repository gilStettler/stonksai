#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo "[env] .env not found in repo root"
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

echo "[env] loaded .env (exported vars)"
