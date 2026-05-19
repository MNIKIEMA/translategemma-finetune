#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

uv run translategemma-finetune \
  --use_chat_template true \
  --chat_template_path translategemma_template.jinja \
  --fp16 true \
  --output_dir outputs/debug-translategemma-4b-it-fr-mos-lora \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --no_save_adapter \
  --test_prompt "Bonjour, comment allez-vous ?" \
  "$@"
