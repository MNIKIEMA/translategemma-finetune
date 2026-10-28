# 2026-10-28: French ↔ Mooré training mix (from the Hy-MT2 recipe)

## Goal

Replace `madoss/fr-mos-final-data` (French → Mooré only) with the mos-fra part of
the Hy-MT2 data mix (`Hy-MT2-Finetune/scripts/mix_translation_data.py`), trained
in both directions.

## Why

`madoss/fr-mos-final-data` (33,835 train rows):

- ~59% dictionary entries (`lexicon_entries` 16,569 + `lexicon` 3,400); 38% of
  rows have ≤ 2 French words, median 5. Too many single-word pairs teach short
  outputs, and BOUQuET is full sentences.
- Includes `news`, whose Mooré side is a summary rewrite (Hy-MT2 finding).
- Quality columns (`laser_score` down to −0.08, `len_ratio` down to 0.06) are
  never used for filtering.
- The `-backtranslated-merged` variant only adds French paraphrases
  (`*_fr_en_fr`, French → English → French) with the same Mooré targets.

## Setup

`uv run translategemma mix` (`src/translategemma_finetune/mix_data.py`),
seed 42, `--total-examples 50000` (the size of the mos-fra slice of the 200k
Hy-MT2 mix), `--into-moore-ratio 0.6`. Same sources, shares, filters, dedupe and
decontamination as Hy-MT2, plus BOUQuET sentence-level dev/test in the held-out
set. Language codes: `fr` and `mos_Latn`.

## Results

Held-out sentences: 16,843 (moore-web validation/test, FLORES+ dev/devtest in
mos/fra, BOUQuET dev/test).

| Source | Usable | Held-out dropped | Sampled | Repeat |
|---|---:|---:|---:|---:|
| moore_web_reviewed | 5,374 | 71 | 15,000 | 2.79× |
| moore_web_aligned | 10,897 | 11 | 17,500 | 1.61× |
| nllb_fra_mos_pivot | 11,153 | 140 | 12,500 | 1.12× |
| moore_web_lexicon | 16,567 | 102 | 5,000 | 0.30× |

Pool sizes match the Hy-MT2 logbook (lexicon 2 rows fewer, removed as BOUQuET
sentences).

Files in `data/fra-mos-mix/` (gitignored):

- `train.jsonl`: 50,000 rows (fr→mos 29,940, mos→fr 20,060).
- `validation.jsonl`: 500 moore-web validation pairs per direction (1,000 rows).
- `test.jsonl`: full moore-web test in both directions (2,573 each, 5,146 rows).

## Notes / decisions

- Training reads each row's direction from `source_lang_code`/`target_lang_code`
  (`translategemma train --data data/fra-mos-mix`, the default), and test scores
  are reported per direction plus `all`. `translategemma eval` does the same
  for mixed-direction files.
- The reviewed rows are repeated 2.8× in one epoch; watch them for memorization
  before adding epochs (same caveat as Hy-MT2).
- Thresholds are the Hy-MT2 starting points, not validated by reading rows near
  the cutoffs.
- Scores on the new test file are not comparable with the old
  `fr-mos-final-data` test (different dataset, both directions). BOUQuET dev
  stays the common yardstick; the base-model baseline is in
  [2026-10-28-bouquet-base-baseline.md](2026-10-28-bouquet-base-baseline.md).
- Next: a base-model baseline on BOUQuET mos→fr, then the first training run on
  the mix.
