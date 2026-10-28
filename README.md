# translategemma-finetune

Minimal training utilities for fine-tuning [translategemma](https://huggingface.co/collections/google/translategemma) with
LoRA adapters using TRL and PEFT.

## Setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
```

## Usage

Preview the formatted first training example without loading the model:

```bash
uv run translategemma-finetune --preview_sample true
```

Run a short fine-tuning job with the defaults:

```bash
uv run translategemma-finetune
```

Evaluate the test split immediately after training and log BLEU, chrF++, and
the generated predictions table to W&B:

```bash
uv run translategemma-finetune \
  --report_to wandb \
  --project moore-web-translategemma \
  --test_split test \
  --test_batch_size 4 \
  --test_max_new_tokens 128
```

Useful options:

```bash
uv run translategemma-finetune \
  --dataset_name madoss/fr-mos-final-data \
  --source_field french \
  --target_field moore \
  --source_lang_code fr \
  --target_lang_code mos_Latn \
  --output_dir outputs/translategemma-4b-it-fr-mos-lora
```

Chat templates are applied by default with the tokenizer's bundled template:

```bash
uv run translategemma-finetune --use_chat_template true
```

TranslateGemma's chat template only knows the languages in its `languages` map.
Languages missing from it are passed as `CODE=Name` entries with
`--extra_languages` and added to the template when the tokenizer is loaded.
Nothing in the repository is modified, and the updated template is saved with
the adapter or merged model. The default registers Mooré:

```bash
uv run translategemma-finetune \
  --target_lang_code mos_Latn \
  --extra_languages mos_Latn=Mossi
```

Pass several entries to add more languages, e.g.
`--extra_languages mos_Latn=Mossi dyu_Latn=Dyula`. `translategemma-evaluate`
loads the tokenizer saved with the adapter, so it reuses the training template;
it accepts `--extra_languages` too when evaluating a base or merged model.

To use a different template altogether, pass `--chat_template_path <path>`; the
extra languages are added to that template too.

By default the script saves the LoRA adapter and tokenizer to `output_dir`.
Pass `--save_merged_path <path>` to also save a merged 16-bit model.

Merge a saved LoRA adapter into the base model after training:

```bash
uv run translategemma-merge-lora \
  --adapter_path outputs/translategemma-4b-it-fr-mos-lora \
  --output_dir translategemma-4b-it_16bit
```

Evaluate a model with BLEU and chrF++. Prompts are rendered and tokenized locally
with the tokenizer from `--model_name`, so they match the training format exactly;
generation runs offline with vLLM in a Modal function (`modal_generate.py`) that
starts for the run and stops when it is done. Upload the adapter to the Modal
volume, then evaluate it on top of the base model:

```bash
uvx modal volume put translategemma-models outputs/translategemma-4b-it-fr-mos-lora /fr-mos-lora
uv run translategemma-evaluate \
  --adapter /models/fr-mos-lora \
  --dataset_split test \
  --source_lang_code fr \
  --target_lang_code mos_Latn \
  --output_predictions_path outputs/predictions.jsonl
```

`--adapter` and `--model` also accept HF repo ids. To evaluate a merged model,
pass it as `--model` without `--adapter`. Set `MODAL_GPU` to change the GPU type
(default `A100-40GB`).

To keep a model running behind an OpenAI-compatible endpoint instead, e.g. for a
demo, use `modal_serve.py`:

```bash
ADAPTER=/models/fr-mos-lora VLLM_API_KEY=<key> uvx modal serve modal_serve.py
```

## Development

```bash
just format
just lint
just test
```
