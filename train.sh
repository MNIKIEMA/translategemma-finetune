#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

uv run translategemma-finetune \
  --use_chat_template true \
  --chat_template_path translategemma_template.jinja \
  "$@"
