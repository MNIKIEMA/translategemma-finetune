#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${USE_UV:-1}" == "1" ]]; then
  RUN=(uv run translategemma-finetune)
else
  RUN=(translategemma-finetune)
fi

"${RUN[@]}" \
  --use_chat_template true \
  --fp16 true \
  --output_dir outputs/debug-translategemma-4b-it-fr-mos-lora \
  --max_steps 20 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --train_sampling_strategy group_by_length \
  --load_in_4bit false \
  --logging_steps 1 \
  --no_save_adapter \
  --test_prompt "Bonjour, comment allez-vous ?" \
  "$@"
