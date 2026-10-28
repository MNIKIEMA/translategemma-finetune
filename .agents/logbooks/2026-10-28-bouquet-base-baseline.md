# 2026-10-28: BOUQuET baseline for base TranslateGemma (fr → mos)

## Goal

Get a baseline for `google/translategemma-4b-it` on French → Mooré before scoring
fine-tunes, and check which prompt format to use: the official chat template, or
the raw `user\n{json}\nmodel\n` string recommended by a Google developer.

## Setup

- Model: `google/translategemma-4b-it`, no adapter.
- Data: `facebook/bouquet`, `benchmark_data/sentence_level/dev/fra_Latn-mos_Latn.parquet`
  (504 sentences, columns `src_text` → `tgt_text`).
- Generation: offline vLLM 0.30.0 on Modal (A100-40GB), greedy, `max_new_tokens=128`,
  stop on `<eos>` / `<end_of_turn>`, `repetition_penalty=1.0`.
- Metrics: sacrebleu corpus BLEU and chrF++.
- Language code `mos_Latn`, registered as `mos_Latn=Mossi` with `--extra_languages`.

```bash
uv run translategemma-evaluate \
  --dataset_name facebook/bouquet \
  --data_files benchmark_data/sentence_level/dev/fra_Latn-mos_Latn.parquet \
  --dataset_split train --source_field src_text --target_field tgt_text \
  [--use_chat_template false]
```

The raw + `<bos>` run used a scratch wrapper that prepends `<bos>` to the raw prompt
(the repo's raw path does not add one; see notes).

## Results

| Prompt format | BLEU | chrF++ | Empty outputs | Runaway (> 3× ref length) |
|---|---:|---:|---:|---:|
| Chat template (default) | **2.49** | 8.53 | 0 | 2 |
| Raw JSON, no `<bos>` (repo's raw path) | 0.03 | 1.07 | 430 | 36 |
| Raw JSON + `<bos>` (Google dev's recipe) | 0.14 | 9.44 | 35 | 39 |

Example, source "En Égypte, le mahshi est généralement farci de riz et de légumes.",
reference "Ezipt tẽnga pʋgẽ, a nãagda ne mui ne zẽ-biis wʋsgo.":

- Chat template: `N Egipto, mahshi ni, kwa riji na mboga.`
- Raw, no `<bos>`: empty.
- Raw + `<bos>`: ```` ```json [ { "type": "translation", ..., "text": "In Egypt, the mahshi is usually stuffed ```` …

Model load ≈ 190–220 s per run; generating 504 sentences takes 3–6 s.

## Notes / decisions

- The base model does not know Mooré. With the chat template it translates, but
  into Swahili-like text (`kwa riji na mboga`), so scores are near zero. This is the
  floor a fine-tune must beat.
- Raw JSON is not a format the base model understands. Without `<bos>` it mostly
  outputs nothing; with `<bos>` it behaves like a chat model: it echoes the JSON,
  translates into English, or answers the request ("Okay, I understand…"). Its
  higher chrF++ (9.44) most likely comes from long outputs that share characters,
  names and numbers with the reference, not from Mooré; BLEU is 0.14.
- Keep the chat template as the default for training and evaluation.
- Bug: the repo's raw path never has `<bos>`. Training sets `add_bos_token=False`
  and evaluation tokenizes with `add_special_tokens=False`, and the raw string does
  not contain `<bos>`. This also affected the earlier "no template" training run
  (BLEU 0.20 in `Memory.md`). **Fixed** the same day: the raw format now starts with
  the tokenizer's `<bos>` (`raw_prompt_prefix` in `templates.py`). Verified one
  `<bos>` per example in both formats, for training text as TRL tokenizes it and for
  evaluation prompts. The "raw + `<bos>`" row above is what the raw path now gives.
- Found while testing: `pillow` was used but not declared, so `uv sync` removed it
  and `AutoProcessor` failed with "Unrecognized image processor". Added to
  `pyproject.toml`; `torchvision` is not needed.
- Next: evaluate a fine-tuned adapter on the same file. The `madoss/translategemma-*`
  adapters on the Hub were trained outside this repo with Unsloth, probably on the
  raw format, so score them with raw + `<bos>`.
