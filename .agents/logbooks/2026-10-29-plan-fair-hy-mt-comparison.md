# 2026-10-29: Plan — fair TranslateGemma vs Hy-MT comparison (steps 1–4 and 6 done; step 5 open)

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

## 2026-11-24: First full bf16 run crashed (CUDA OOM at step 250)

W&B `h5esoe5v`, Modal `ap-YHko2LA1KFso5pxf1Aw3At`. Eval loss 2.17 at step 200
(warmup, LR near its 1e-4 peak).

- Cause: `--max-seq-length` only set the tokenizer's `model_max_length`, so
  TRL truncated at its own default `max_length=1024`. 69 of the 50k training
  rows exceed 512 tokens (6 exceed 1024, the longest 1241; median 146), and
  `group_by_length` puts them in the same batch. At 16 × 1024 tokens, TRL's
  token-accuracy metric copies the logits in fp32 (16 GiB), which overflowed
  80 GB with the bf16 base. Earlier 4-bit runs also trained at up to 1024
  tokens; they had ~6 GB more headroom.
- Fix: `--max-seq-length` now sets `SFTConfig.max_length` (default 512, the
  value `scripts/train.sh` passes). The 69 longer rows are truncated, losing
  the end of the translation.

## 2026-11-24: Step 3 trained — TranslateGemma bf16, translation-only loss

W&B `0739f02e`, Modal `ap-9kIl4awFFMEElk18JCMkec`. `MODAL_HF_SECRET=huggingface-secret
TRAINER=train-modal bash scripts/train.sh --data madoss/translategemma-fra-mos-mix
--output-dir translategemma-bf16-completion-only --num-train-epochs 2` (data revision
`ca1409b`, max length 512). 1 h 29 min of training on an A100-80GB, ~1.8 s/step.

| Step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | **2400** | 2600 | 2800 | 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eval loss | 2.311 | 1.551 | 1.312 | 1.223 | 1.175 | 1.124 | 1.114 | 1.120 | 1.115 | 1.098 | 1.102 | **1.089** | 1.095 | 1.090 | 1.097 |

- Early stopping (patience 3) ended the run at step 3000 of 3126. The saved
  adapter is the best checkpoint, step 2400 (epoch 1.54, eval loss 1.0895,
  token accuracy 0.778), in `/models/translategemma-bf16-completion-only`.
  Checkpoints 2200–3000 are kept next to it.
- Eval loss flattens at ~1.09–1.12 from step 1400; it does not rise in epoch 2
  as the full-text eval loss did. Train loss drops from ~0.63 to ~0.51 at the
  epoch boundary (repeated rows), so the train/eval gap grows.
- Test prompt "Bonjour, comment allez-vous ?" → "Yiibã, yãmb maanda wãn n tɩgimd
  y menga ?" (penalty 1.0).
- Next: step 6 evaluation (BOUQuET test, moore-web test; penalty 1.0 and 1.05;
  256 max new tokens).

## 2026-11-24: Step 6 evaluation of the bf16 translation-only adapter

`translategemma eval {bouquet,moore-web} --split test --direction both --adapter
/models/translategemma-bf16-completion-only --max-new-tokens 256 --temperature 0
--repetition-penalty {1.0,1.05}` (vLLM, A100-40GB). Predictions in
`outputs/eval/bf16-completion-only/` (local, not committed). chrF++ (BLEU);
looping = outputs of 8+ words with under 40% unique words, fra→mos.

"Old 2 ep" is the 4-bit full-text-loss run of
[2026-10-29-two-epoch-run-fra-mos-mix.md](2026-10-29-two-epoch-run-fra-mos-mix.md),
scored with 128 max new tokens; Hy-MT used 256.

**BOUQuET test, sentence level** (854 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
|---|---:|---:|---:|---:|
| Old 2 ep | 1.0 | 24.41 (8.18) | 23.54 (6.46) | 27 |
| **bf16, translation-only** | 1.0 | 24.92 (7.76) | 24.34 (6.82) | 15 |
| Old 2 ep | 1.05 | 24.80 (8.06) | 23.40 (6.07) | 10 |
| **bf16, translation-only** | 1.05 | 24.93 (7.70) | 24.83 (6.59) | 3 |
| Hy-MT 1.8B | 1.05 | 25.57 (7.08) | 27.45 (8.34) | 12 |

**moore-web test** (2,573 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
|---|---:|---:|---:|---:|
| Old 2 ep | 1.0 | 29.00 (8.95) | 30.19 (14.82) | 87 |
| **bf16, translation-only** | 1.0 | 30.63 (8.94) | 31.34 (15.78) | 74 |
| Old 2 ep | 1.05 | 29.49 (8.95) | 30.32 (14.62) | 18 |
| **bf16, translation-only** | 1.05 | 31.02 (9.27) | 31.75 (15.63) | 15 |
| Hy-MT 1.8B | 1.05 | 30.32 (8.93) | 34.67 (17.41) | — |

- chrF++ improves in every direction and set; most into French (+1.4 BOUQuET,
  +1.4 moore-web at penalty 1.05). BLEU drops slightly on BOUQuET fra→mos.
- vs Hy-MT: ahead on moore-web fra→mos (31.02 vs 30.32), behind on BOUQuET
  fra→mos (24.93 vs 25.57) and into French (gap 2.6 BOUQuET, 2.9 moore-web;
  was 4.05 and 4.35).
- Not yet a fair comparison: Hy-MT still trained on the 200k 10-direction mix
  (step 4 pending). The change bundles translation-only loss, bf16, the
  template-whitespace fix and 512-token truncation, and the old run was scored
  with 128 max new tokens, so the gain is not attributable to one of them.
- vLLM logs "Chat template warmup failed" at startup: its warm-up request is
  not in TranslateGemma's content format. Harmless: prompts are rendered
  locally; no empty predictions on BOUQuET, 1 on moore-web mos→fra.

## 2026-11-24: Step 4 — Hy-MT on the same 50k rows; same-data comparison

Hy-MT trained on `madoss/translategemma-fra-mos-mix` (`ca1409b`) converted to
its v1 rows, validated on the same 1,000 moore-web rows, lr 2e-4, 2 epochs,
best checkpoint by eval loss (step 800, epoch 1.02). Details, command and
eval-loss curve: `Hy-MT2-Finetune/.agents/experiments/2026-11-24-hy-mt15-1_8b-fra-mos-mix.md`
(branch `fra-mos-mix-comparison`). Both models rescored with the same
sacrebleu code; references identical. chrF++ (BLEU), penalty 1.05; looping
is fra→mos.

| Test set | Model | fra→mos | mos→fra | fra→mos looping |
|---|---|---|---|---:|
| BOUQuET (854) | TranslateGemma 4B | 24.93 (7.70) | 24.83 (6.59) | 3 |
| BOUQuET (854) | Hy-MT 1.8B | 24.94 (6.89) | **27.44** (9.00) | 9 |
| moore-web (2,573) | TranslateGemma 4B | **31.02** (9.27) | 31.75 (15.63) | 15 |
| moore-web (2,573) | Hy-MT 1.8B | 30.24 (8.25) | **35.11** (17.45) | 40 |

At penalty 1.0: BOUQuET fra→mos 24.92 vs 23.76, mos→fra 24.34 vs 26.97;
moore-web fra→mos 30.63 vs 29.10, mos→fra 31.34 vs 34.54 (TranslateGemma vs
Hy-MT).

- **Into French, Hy-MT is better** on the same data: +2.6 (BOUQuET) and +3.4
  (moore-web) chrF++, ~2 BLEU. Larger than a learning-rate sweep is likely
  to close.
- **Into Mooré they are close**: tied on BOUQuET at 1.05 (TranslateGemma
  +0.8 BLEU), TranslateGemma +0.8 on moore-web, and it loops less and does
  not need the penalty as much (+1.2 / +1.5 at 1.0).
- Hy-MT's extra eng/dyu data did not drive its lead: on these 50k rows it
  scores within ±0.6 chrF++ of its 200k-mix run.
- Step 5 (learning rates: Hy-MT 1e-4, TranslateGemma 2e-4) is only needed to
  settle into-Mooré.
