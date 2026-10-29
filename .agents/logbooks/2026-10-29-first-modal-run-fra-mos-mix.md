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

## Notes / decisions

- The model now writes Mooré (the base wrote Swahili-like text); fr→mos
  chrF++ on BOUQuET went from 8.5 to 21.8.
- Into-Mooré outputs loop on 7–9% of sentences, which also inflates the length
  ratio; mos→fr rarely loops. Next: re-score with `--repetition-penalty`
  (e.g. 1.1) to see how much of the gap is decoding.
- Sentences that do not loop are often fluent but loosely faithful (e.g. "les
  deux parties … la loi" → "les gens … cette élection").
- moore-web test is in-domain for most of the mix; BOUQuET is the
  out-of-domain yardstick.
- Predictions were kept in the session scratchpad only (not committed).
