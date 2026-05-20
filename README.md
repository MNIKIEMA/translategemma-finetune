# translategemma-finetune

Minimal training utilities for fine-tuning `google/translategemma-4b-it` with
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
  --output_dir outputs/translategemma-4b-it-fr-mos-lora
```

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

## Development

```bash
just format
just lint
just test
```
