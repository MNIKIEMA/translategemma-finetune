# 2026-10-29: First LoRA run on the French ↔ Mooré mix (Modal)

## Goal

First training run on `madoss/translategemma-fra-mos-mix` (revision `ca1409b`,
see [2026-10-28-fra-mos-data-mix.md](2026-10-28-fra-mos-data-mix.md)), scored
against the base model on BOUQuET dev and moore-web test in both directions.

## Setup

- `TRAINER=train-modal bash scripts/train.sh --output-dir fr-mos-lora
  --run-name translategemma-4b-it-fr-mos-lora-mix-lr1e-4 --detach`, Modal
  workspace `nikiemamahamadi01` (`MODAL_HF_SECRET=huggingface-secret`),
  A100-80GB.
- `google/translategemma-4b-it`, 4-bit, LoRA r128 / alpha 256, chat template,
  LR 1e-4 (cosine to 1e-6), effective batch 32, max length 512, 1 epoch
  (1,563 steps), eval every 200 steps on 1,000 validation rows.
- Adapter: `/models/fr-mos-lora` in the `translategemma-models` volume of that
  workspace (final checkpoint = best validation loss).
- Eval: `translategemma eval {bouquet,moore-web}`, vLLM 0.30.0 on Modal,
  greedy, max 128 new tokens, no repetition penalty.

## Results

Training took 70 min (2.7 s/step). Validation loss improved at every eval and
flattened at the end:

| Step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1563 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eval loss | 2.009 | 1.449 | 1.233 | 1.134 | 1.087 | 1.071 | 1.056 | 1.056 |
| token accuracy | 65.8% | 70.8% | 74.2% | 75.8% | 76.9% | 77.2% | 77.5% | 77.5% |

Final train loss (last 50 steps) 0.62.

| Set | Direction | Base BLEU | Base chrF++ | LoRA BLEU | LoRA chrF++ |
|---|---|---:|---:|---:|---:|
| BOUQuET dev (504 per direction) | fr → mos | 2.50 | 8.53 | 5.93 | 21.75 |
| | mos → fr | 3.17 | 18.43 | 7.38 | 22.77 |
| moore-web test (2,573 per direction) | fr → mos | 1.38 | 10.76 | 7.60 | 27.05 |
| | mos → fr | 2.94 | 19.94 | 13.66 | 29.37 |

Base BOUQuET fr→mos matches the earlier baseline (2.49 / 8.53).

Output checks on the LoRA predictions:

| Set | Direction | Empty | Length ratio (hyp/ref words) | Looping outputs |
|---|---|---:|---:|---:|
| BOUQuET | fr → mos | 0 | 1.23 | 45 (8.9%) |
| BOUQuET | mos → fr | 0 | 1.01 | 4 |
| moore-web | fr → mos | 0 | 1.17 | 183 (7.1%) |
| moore-web | mos → fr | 1 | 0.88 | 29 |

"Looping" = more than 8 words with under 40% unique words, e.g.
"yɩɩg yʋʋm a tãab pʋgẽ, yɩɩg yʋʋm a tãab pʋgẽ, ..." until the 128-token limit.

### Repetition penalty (BOUQuET dev, same adapter)

| Penalty | fr→mos BLEU / chrF++ | mos→fr BLEU / chrF++ | fr→mos looping | fr→mos length ratio |
|---:|---:|---:|---:|---:|
| 1.0 | 5.93 / 21.75 | 7.38 / 22.77 | 45 | 1.23 |
| **1.05** | **7.11 / 23.62** | **7.38 / 23.66** | 6 | 1.06 |
| 1.1 | 6.39 / 23.09 | 6.52 / 23.50 | 1 | 1.07 |
| 1.2 | 3.97 / 20.49 | 4.37 / 23.25 | 1 | 1.08 |

1.05 removes most loops (45 → 6) and is best on every score. Higher
penalties remove the last loops but hurt BLEU: vLLM's penalty also covers
prompt tokens and words the translation legitimately repeats (mos→fr outputs
get 24% longer at 1.2).

### BOUQuET test and leaderboard submission

BOUQuET sentence-level test (854 per direction), repetition penalty 1.05:
fr→mos 7.07 / 24.19, mos→fr 7.41 / 23.87 (BLEU / chrF++); 14 looping fr→mos
outputs. Submission CSV for the Burkimbia MT leaderboard (local, not
committed): `outputs/submissions/translategemma-4b-it-fra-mos-lora-rp1.05.csv`,
1,708 rows, no empty translations.

Adapter published (private) as
[`madoss/translategemma-4b-it-fra-mos-lora`](https://huggingface.co/madoss/translategemma-4b-it-fra-mos-lora)
(final adapter only, no checkpoints). It was pushed from a Modal function
reading the volume: the upload from this machine stalled at a few KB/s.

## Notes / decisions

- The model now writes Mooré (the base wrote Swahili-like text); fr→mos
  chrF++ on BOUQuET went from 8.5 to 21.8.
- Into-Mooré outputs loop on 7–9% of sentences, which also inflates the length
  ratio; mos→fr rarely loops. A repetition penalty of 1.05 fixes most of it at
  decoding time (see above); the looping is still a model weakness to train
  away. Defaults stay at 1.0 so scores measure the model; report each run at
  1.0 plus a 1.05 row (1.05 was picked on BOUQuET dev, so its dev score is
  optimistic).
- Sentences that do not loop are often fluent but loosely faithful (e.g. "les
  deux parties … la loi" → "les gens … cette élection").
- moore-web test is in-domain for most of the mix; BOUQuET is the
  out-of-domain yardstick.
- Predictions were kept in the session scratchpad only (not committed).
