#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

uv run translategemma-finetune \
  --use_chat_template true \
  --chat_template_path chat_template.jinja \
  --fp16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_steps -1 \
  --max_grad_norm 1.0 \
  --learning_rate 1e-5 \
  --optim adamw_torch \
  --save_total_limit 3 \
  --test_prompt "Bonjour, comment allez-vous ?" \
  "$@"
