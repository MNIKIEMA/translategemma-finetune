"""``translategemma <command>``: one entry point for mixing data, training, scoring and merging."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "mix": ("translategemma_finetune.mix_data", "Build the French <-> Mooré training mix."),
    "train": ("translategemma_finetune.finetune", "Fine-tune a LoRA adapter."),
    "train-modal": ("translategemma_finetune.modal_train", "Run `train` on a Modal GPU."),
    "eval": ("translategemma_finetune.evaluate", "Score a model with BLEU and chrF++ on Modal."),
    "merge": ("translategemma_finetune.merge_lora_weight", "Merge a LoRA adapter into the model."),
}

USAGE = (
    "usage: translategemma <command> [options]\n\ncommands:\n"
    + "\n".join(f"  {name:<13}{help_text}" for name, (_, help_text) in COMMANDS.items())
    + "\n\nRun `translategemma <command> --help` for its options."
)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return
    command, *rest = argv
    if command not in COMMANDS:
        print(f"translategemma: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        sys.exit(2)
    # Imported on demand: `train` and `merge` load transformers, the Modal commands do not.
    importlib.import_module(COMMANDS[command][0]).main(rest)


if __name__ == "__main__":
    main()
