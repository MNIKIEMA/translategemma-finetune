# 2026-10-29: Plan — fair TranslateGemma vs Hy-MT comparison (steps 1–2 done)

## Goal

Find out which base model is better for French ↔ Mooré fine-tuning:
`google/translategemma-4b-it` or `tencent/HY-MT1.5-1.8B`. The current numbers
([2026-10-29-two-epoch-run-fra-mos-mix.md](2026-10-29-two-epoch-run-fra-mos-mix.md))
do not answer it: Hy-MT leads into French (BOUQuET test mos→fra 27.45 vs
23.40 chrF++ at penalty 1.05) and is close into Mooré, but the two runs differ
in much more than the base model.

## What differs today

| | Hy-MT 1.5 1.8B (`Hy-MT2-Finetune`) | TranslateGemma 4B (this repo) |
|---|---|---|
| Data | 200k rows, 10 directions (mos/dyu/fra/eng); fra↔mos is the same 50k slice | `madoss/translategemma-fra-mos-mix` only (50k fra↔mos) |
| Other Mooré data | eng↔mos, dyu↔mos (~74k rows) | none |
| Loss | assistant reply only | whole rendered text, prompt included (TRL `text` dataset) |
| Base precision | bf16 | 4-bit (`--load-in-4bit true`) |
| LoRA r / alpha / dropout | 64 / 128 / 0.05 | 128 / 256 / 0 |
| LR, effective batch, warmup | 2e-4, 64, 100 steps | 1e-4, 32, ratio 0.08 |
| Best checkpoint | step 2,400 = epoch 0.77 (~38k fra↔mos rows seen) | step 2,400 = epoch 1.5 (~77k rows seen) |
| Eval max new tokens | 256 | 128 |

## Plan

1. **This repo: translation-only loss.** Train on the assistant reply only
   (e.g. prompt/completion rows for SFTTrainer, or `assistant_only_loss`), as
   Hy-MT does. Check with `--preview-sample` and the label mask that the
   prompt tokens are `-100`.
2. **This repo: bf16 option.** Train with `--load-in-4bit false` (4B in bf16
   fits on the A100-80GB) so both models use the same base precision.
3. **Retrain TranslateGemma** with 1 and 2 on the same data revision
   (`ca1409b`).
4. **Hy-MT2-Finetune: train Hy-MT on `madoss/translategemma-fra-mos-mix`**
   (fra↔mos only), converted to its ShareGPT JSONL format with its v1
   prompt. Cheaper than training Gemma on the 200k mix (~1 h for 1.8B vs
   ~4–5 h per epoch for 4B), and the Gemma template would also need Dyula.
5. **Learning rate.** Do not force the same LR or LoRA rank on a 1.8B and a
   4B model: give each a small sweep (e.g. 1e-4 and 2e-4) at the same
   training budget (same epochs over the same rows) and compare each
   model's best.
6. **Same evaluation for both.** BOUQuET sentence-level test and moore-web
   test, penalty 1.0 and 1.05, the same max new tokens (256), greedy, vLLM.

## Notes

- Pick checkpoints the same way for both: on eval loss (decided 2026-11-24;
  generating on a dev set at every eval is too costly). With translation-only
  loss, eval loss measures the translation alone; in the full-text runs it rose
  in the second epoch while into-Mooré scores improved, so check whether that
  still happens.
- BOUQuET test is the leaderboard set: report on it, never select on it.

## 2026-11-24: Steps 1 and 2 done (code only, no training run yet)

- **Translation-only loss.** `format_for_training` now returns prompt/completion
  rows. The prompt is the inference prompt (`add_generation_prompt=True`); the
  completion is the stripped translation + `<end_of_turn>\n`, and TRL appends
  `<eos>`. TRL 1.4 masks the prompt for prompt/completion data
  (`completion_only_loss`, new `--completion-only-loss`, default true; false
  reproduces the old full-text loss). Checked with the real `SFTTrainer` and the
  TranslateGemma processor on a tiny random Gemma 3: in the batch it builds,
  only `<translation><end_of_turn>\n<eos>` has labels ≠ -100, e.g. 4 of 89
  tokens for a short row. `--preview-sample true` now prints the tokens
  trained on.
- **Template whitespace.** The TranslateGemma chat template renders the reply
  as `<start_of_turn>model\n` + `\n` + 8 spaces + reply: its Jinja tags do not
  strip the newline between them. Earlier runs trained on that whitespace. The
  completion is now built directly, without it, so the target is what
  inference prompts are followed by.
- **bf16.** `scripts/train.sh` now passes `--load-in-4bit false`: LoRA on the
  bf16 base. PEFT keeps the LoRA weights in fp32. Checked on the tiny model:
  bf16 base, fp32 adapters, gradient checkpointing, one step with non-zero
  grad norm. Not yet run on the 4B model; the 4B bf16 weights (~8.6 GB) fit on
  the A100-80GB.
- **Eval loss changes meaning.** It is now the loss on translation tokens only,
  so it is not comparable with eval loss from earlier runs, and is higher in
  absolute terms (the prompt tokens were easy to predict).

## 2026-11-24: Modal smoke run (30 steps, bf16 + translation-only loss)

`MODAL_HF_SECRET=huggingface-secret TRAINER=train-modal bash scripts/train.sh
--data madoss/translategemma-fra-mos-mix --output-dir smoke-bf16-completion-only
--max-steps 30 --eval-steps 15 --save-steps 15 --report-to none
--wandb-eval-samples 0`, A100-80GB.

- Runs end to end: evals, checkpoints, adapter saved to
  `/models/smoke-bf16-completion-only`.
- Train loss 6.4 (step 1) → 1.42 (step 30); eval loss 1.83 (step 15) → 1.68
  (step 30).
- ~2 s/step at effective batch 32, so ~1 h per epoch of the 50k rows plus evals
  (vs the ~4–5 h estimated for 4-bit).
- "Bonjour, comment allez-vous ?" → "A Mise, sẽn yaa n yaa?": weak after 30
  steps, but generation stops cleanly at end of turn.
