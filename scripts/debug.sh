#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${USE_UV:-1}" == "1" ]]; then
  RUN=(uv run translategemma train)
else
  RUN=(translategemma train)
fi

"${RUN[@]}" \
  --fp16 true \
  --output-dir outputs/debug-translategemma-4b-it-fr-mos-lora \
  --max-steps 20 \
  --learning-rate 1e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --train-sampling-strategy group_by_length \
  --load-in-4bit false \
  --logging-steps 1 \
  --no-save-adapter \
  --test-prompt "Bonjour, comment allez-vous ?" \
  "$@"
