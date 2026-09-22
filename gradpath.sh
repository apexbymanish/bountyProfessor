#!/usr/bin/env bash
# gradpath.sh — run gradpath from a checkout without installing
set -euo pipefail
exec python -m gradpath.cli "$@"
