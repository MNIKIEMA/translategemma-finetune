# 2026-10-29: Plan — fair TranslateGemma vs Hy-MT comparison (not started)

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

- Pick checkpoints the same way for both, preferably on chrF++ on a dev set
  (BOUQuET dev) rather than eval loss: eval loss rose in the second epoch
  while into-Mooré scores improved.
- BOUQuET test is the leaderboard set: report on it, never select on it.
