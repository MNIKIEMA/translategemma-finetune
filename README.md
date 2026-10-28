# translategemma-finetune

Minimal training utilities for fine-tuning [translategemma](https://huggingface.co/collections/google/translategemma) with
LoRA adapters using TRL and PEFT.

## Setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
```

Everything runs through one command:

```text
translategemma mix     Build the French <-> Mooré training mix.
translategemma train   Fine-tune a LoRA adapter.
translategemma eval    Score a model with BLEU and chrF++ on Modal.
translategemma merge   Merge a LoRA adapter into the model.
```

`translategemma <command> --help` lists each command's options.

## Quick start

```bash
uv run translategemma mix              # writes data/fra-mos-mix/{train,validation,test}.jsonl
bash train.sh                          # trains on data/fra-mos-mix
uv run translategemma eval bouquet --adapter /models/fr-mos-lora
```

## Training data

`translategemma mix` builds a weighted French ↔ Mooré mix from the mos-fra
recipe of the Hy-MT2 fine-tune:

| Source | Share | Filter |
| --- | ---: | --- |
| `madoss/moore-web-parallel`, reviewed | 30% | `reviewed` (checked by a person) |
| `madoss/moore-web-parallel`, aligned | 35% | `conseils`, `lexicon`, `digital-defs`; `laser_score >= 0.70`; `comet_qe >= 0.52` when present |
| `burkimbia/nllb-burkina-parallel` `fra-mos` | 25% | NLLB `score >= 1.06`, `length_ratio >= 0.5`, GlotLID fra/mos with probability `>= 0.9` |
| `madoss/moore-web-parallel`, dictionary | 10% | `lexicon_entries`, `digital-terms` |

moore-web `news` is excluded: its Mooré side is a summary rewrite. 60% of rows
are translated into Mooré (`--into-moore-ratio`), the rest into French.
Sentences from moore-web validation/test, FLORES+ dev/devtest and BOUQuET
dev/test are removed from training, and duplicates are dropped. FLORES+ is
gated: accept its terms on the Hub and put `HF_TOKEN` in `.env`.

Validation is 500 moore-web validation rows per direction; test is the full
moore-web test split in both directions. Every row stores its direction in
`source_lang_code`/`target_lang_code`.

## Train

`train.sh` holds the hyperparameters; extra options are passed through:

```bash
bash train.sh --learning-rate 1e-4
```

Useful `translategemma train` options:

```bash
uv run translategemma train --preview-sample true     # print the first formatted example and exit
uv run translategemma train --data madoss/fr-mos-final-data   # a Hub dataset with french/moore columns
uv run translategemma train --test true               # score the test split after training, per direction
uv run translategemma train --use-chat-template false  # raw `user\n{json}\nmodel\n` prompts
```

The adapter and tokenizer are saved to `--output-dir`; pass
`--save-merged-path <path>` to also save a merged 16-bit model.

### Languages in the chat template

TranslateGemma's chat template only knows the languages in its `languages` map.
Missing ones are passed as `CODE=Name` entries with `--extra-languages`
(default `mos_Latn=Mossi`) and added to the template when the tokenizer is
loaded; the updated template is saved with the adapter or merged model. Pass
several entries to add more, e.g. `--extra-languages mos_Latn=Mossi dyu_Latn=Dyula`.
`--chat-template-path <path>` uses another template, and the extra languages are
added to it too. `translategemma eval` takes the same options.

## Evaluate

`translategemma eval` renders and tokenizes the prompts locally, so they match
the training format exactly, then generates offline with vLLM in a Modal
function (`modal_generate.py`) that starts for the run and stops when it is
done. Scores are printed per direction, plus `all` when there are several.

```bash
uv run translategemma eval bouquet                        # BOUQuET dev, both directions
uv run translategemma eval bouquet --split test --direction fr-mos
uv run translategemma eval moore-web                      # moore-web test, both directions
uv run translategemma eval data/fra-mos-mix --split validation
uv run translategemma eval path/to/rows.jsonl --output outputs/predictions.jsonl
```

The model is `google/translategemma-4b-it` unless you pass `--model`. Upload an
adapter to the Modal volume and evaluate it on top of the base model:

```bash
uvx modal volume put translategemma-models outputs/translategemma-4b-it-fr-mos-lora /fr-mos-lora
uv run translategemma eval bouquet --adapter /models/fr-mos-lora
```

`--model` and `--adapter` also accept HF repo ids; for a merged model, pass it as
`--model` without `--adapter`. Set `MODAL_GPU` to change the GPU type (default
`A100-40GB`).

To keep a model running behind an OpenAI-compatible endpoint instead, e.g. for a
demo, use `modal_serve.py`:

```bash
ADAPTER=/models/fr-mos-lora VLLM_API_KEY=<key> uvx modal serve modal_serve.py
```

## Merge

```bash
uv run translategemma merge \
  --adapter-path outputs/translategemma-4b-it-fr-mos-lora \
  --output-dir translategemma-4b-it_16bit
```

## Development

```bash
just format
just lint
just test
```
