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

Useful options:

```bash
uv run translategemma-finetune \
  --dataset_name madoss/fr-mos-final-data \
  --source_field french \
  --target_field moore \
  --source_lang_code fr \
  --target_lang_code mos \
  --packing false \
  --output_dir outputs/translategemma-4b-it-fr-mos-lora
```

Pass `--packing true` to let TRL pack multiple short examples into each
training sequence.

Chat templates are applied by default with the tokenizer's bundled template:

```bash
uv run translategemma-finetune --use_chat_template true
```

To add a language that is not already in
`src/translategemma_finetune/languages.json`, generate a local chat template
from the bundled `translategemma_template.jinja`:

```bash
uv run translategemma-update-languages "mos=Mooré"
```

This updates `src/translategemma_finetune/languages.json` and writes
`chat_template.jinja`. Use the generated template during training:

```bash
uv run translategemma-finetune \
  --use_chat_template true \
  --chat_template_path chat_template.jinja
```

If `--chat_template_path` is omitted, the default tokenizer template is used.

By default the script saves the LoRA adapter and tokenizer to `output_dir`.
Pass `--save_merged_path <path>` to also save a merged 16-bit model.

Evaluate a fine-tuned LoRA adapter with BLEU and chrF++:

```bash
uv run translategemma-evaluate \
  --adapter_path outputs/translategemma-4b-it-fr-mos-lora \
  --dataset_split test \
  --source_lang_code fr \
  --target_lang_code mos \
  --chat_template_path chat_template.jinja \
  --output_predictions_path outputs/predictions.jsonl
```

The evaluation script loads the base model from `--model_name`, applies the
adapter from `--adapter_path`, generates translations for the dataset split,
and reports corpus BLEU and chrF++.

To evaluate a merged model instead, pass the merged model directory as
`--model_name` and disable adapter loading with an empty `--adapter_path`:

```bash
uv run translategemma-evaluate \
  --model_name translategemma-4b-it_16bit \
  --adapter_path "" \
  --dataset_split test \
  --source_lang_code fr \
  --target_lang_code mos \
  --chat_template_path chat_template.jinja
```

## Development

```bash
just format
just lint
just test
```
