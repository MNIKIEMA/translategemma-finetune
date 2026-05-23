#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${USE_UV:-0}" == "1" ]]; then
  RUN=(uv run translategemma-finetune)
else
  RUN=(translategemma-finetune)
fi

"${RUN[@]}" \
  --use_chat_template true \
  --chat_template_path chat_template.jinja \
  --max_seq_length 512 \
  --fp16 false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --max_steps -1 \
  --num_train_epochs 1 \
  --max_grad_norm 1.0 \
  --learning_rate 6e-6 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --optim adamw_8bit \
  --eval_strategy steps \
  --eval_steps 200 \
  --early_stopping_patience 3 \
  --early_stopping_threshold 0.0 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --load_in_4bit true \
  --train_sampling_strategy group_by_length \
  --run_name translategemma-4b-it-fr-mos-lora \
  --project moore-web-translategemma \
  --test_prompt "Bonjour, comment allez-vous ?" \
  "$@"
