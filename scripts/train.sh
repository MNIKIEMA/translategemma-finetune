#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${USE_UV:-0}" == "1" ]]; then
  RUN=(uv run translategemma train)
else
  RUN=(translategemma train)
fi

"${RUN[@]}" \
  --test true \
  --lora-r 128 \
  --lora-alpha 256 \
  --max-seq-length 512 \
  --fp16 false \
  --bf16 true \
  --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing true \
  --max-steps -1 \
  --num-train-epochs 1 \
  --max-grad-norm 1.0 \
  --learning-rate 6e-6 \
  --lr-scheduler-type cosine_with_min_lr \
  --lr-scheduler-kwargs '{"min_lr": 1e-6}' \
  --optim adamw_8bit \
  --eval-strategy steps \
  --eval-steps 200 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.0 \
  --load-best-model-at-end true \
  --metric-for-best-model eval_loss \
  --greater-is-better false \
  --save-strategy steps \
  --save-steps 200 \
  --save-total-limit 5 \
  --load-in-4bit true \
  --train-sampling-strategy group_by_length \
  --report-to wandb \
  --run-name translategemma-4b-it-fr-mos-lora \
  --project moore-web-translategemma \
  --test-prompt "Bonjour, comment allez-vous ?" \
  "$@"
