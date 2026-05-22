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
  --dataset_name madoss/fr-mos-final-data-backtranslated-merged \
  --use_chat_template true \
  --chat_template_path chat_template.jinja \
  --fp16 false \
  --bf16 true \
  --load_in_4bit true \
  --max_seq_length 512 \
  --output_dir outputs/translategemma-4b-it-fr-mos-backtranslated-merged-lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --max_steps 200 \
  --num_train_epochs 1 \
  --max_grad_norm 1.0 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --optim adamw_8bit \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --test_prompt "Bonjour, comment allez-vous ?" \
  "$@"
