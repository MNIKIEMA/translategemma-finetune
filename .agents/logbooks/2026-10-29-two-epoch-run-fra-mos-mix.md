# 2026-10-29: Two-epoch LoRA run on the French ↔ Mooré mix

## Goal

See whether a second epoch helps, compared with the one-epoch run
([2026-10-29-first-modal-run-fra-mos-mix.md](2026-10-29-first-modal-run-fra-mos-mix.md))
and with Hy-MT 1.5 1.8B (`Hy-MT2-Finetune`, checkpoint-2400, same mos-fra
slice inside a 200k mos/dyu/fra/eng mix).

## Setup

Same as the one-epoch run except `--num-train-epochs 2` (cosine schedule over
3,126 steps) and `--data madoss/translategemma-fra-mos-mix` (revision
`ca1409b`, loaded from the Hub in the container):

```bash
MODAL_HF_SECRET=huggingface-secret TRAINER=train-modal bash scripts/train.sh \
  --data madoss/translategemma-fra-mos-mix --num-train-epochs 2 \
  --output-dir fr-mos-lora-2ep --run-name translategemma-4b-it-fr-mos-lora-mix-lr1e-4-2ep --detach
```

Adapter: `/models/fr-mos-lora-2ep` (workspace `nikiemamahamadi01`), the
step-2,400 checkpoint. Eval: vLLM, greedy, 128 new tokens.

## Results

Early stopping ended the run at step 3,000 (100 min). Validation loss:

| Step | 200 | 800 | 1400 | 1600 | 2000 | 2400 | 2600 | 2800 | 3000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 epochs | 2.290 | 1.222 | 1.116 | 1.133 | 1.105 | **1.095** | 1.100 | 1.098 | 1.100 |
| 1 epoch | 2.009 | 1.134 | 1.056 | (end, 1.056) | | | | | |

The loss rose when the second epoch started (step 1,563) and never reached
the one-epoch run's 1.056.

chrF++ (BLEU); looping = outputs of 8+ words with under 40% unique words.

**BOUQuET test, sentence level** (854 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
|---|---:|---:|---:|---:|
| 1 epoch | 1.0 | 23.04 (6.58) | 23.33 (7.52) | 63 |
| 2 epochs | 1.0 | 24.41 (8.18) | 23.54 (6.46) | 27 |
| 1 epoch | 1.05 | 24.19 (7.07) | **23.87 (7.41)** | 14 |
| 2 epochs | 1.05 | **24.80 (8.06)** | 23.40 (6.07) | 10 |
| Hy-MT 1.8B | 1.05 | 25.57 (7.08) | 27.45 (8.34) | 12 |

**moore-web test** (2,573 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
|---|---:|---:|---:|---:|
| 1 epoch | 1.0 | 27.05 (7.60) | 29.37 (13.66) | 183 |
| 2 epochs | 1.0 | 29.00 (8.95) | 30.19 (14.82) | 87 |
| 1 epoch | 1.05 | 28.19 (8.07) | 29.64 (13.47) | 40 |
| 2 epochs | 1.05 | **29.49 (8.95)** | **30.32 (14.62)** | 18 |
| Hy-MT 1.8B | 1.05 | 30.32 (8.93) | 34.67 (17.41) | — |

## Leaderboard submission

`outputs/submissions/translategemma-4b-it-fra-mos-lora-2ep-rp1.05.csv` (local,
not committed) replaces the one-epoch CSV: the 2-epoch adapter at penalty
1.05 for both directions, 1,708 rows, same ids. One model for both
directions; picking each direction's adapter from these BOUQuET test scores
would select on the leaderboard's own test set.

## Notes / decisions

- Validation loss was misleading: it got worse, but the second epoch improved
  every into-Mooré score and halved looping (BOUQuET 63 → 27 without
  penalty). Choose checkpoints on chrF++, not eval loss (Hy-MT2 reached the
  same conclusion).
- Into French it helps only in domain: moore-web mos→fra +0.7 to +0.8 chrF++,
  BOUQuET mos→fra −0.3 to −0.5 chrF++ and −1 BLEU. Consistent with
  memorizing the repeated moore-web rows (reviewed rows seen ~5.6×).
- The penalty matters less after two epochs (BOUQuET fra→mos +0.4 chrF++,
  BLEU slightly lower), because the model loops less.
- Against Hy-MT 1.8B at penalty 1.05: into Mooré TranslateGemma is close
  (BOUQuET 24.80 vs 25.57 chrF++, higher BLEU 8.06 vs 7.08; moore-web 29.49 vs
  30.32); into French Hy-MT stays well ahead (BOUQuET 27.45 vs 23.40, moore-web
  34.67 vs 30.32).
- Next levers for mos→fra: fewer repeats of moore-web rows (more unique
  sentence data, lower reviewed share), a higher into-French ratio than 40%,
  or a higher learning rate for one epoch.
