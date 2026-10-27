#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VINA_RTMSCORE_PYTHON:-}" ]]; then
  SEARCH_PYTHON="$VINA_RTMSCORE_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  SEARCH_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  SEARCH_PYTHON=python3
fi
exec "$SEARCH_PYTHON" "$PROJECT_ROOT/scripts/dock.py" --root "$PROJECT_ROOT" "$@"
