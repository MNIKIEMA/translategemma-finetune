"""Score a model with BLEU and chrF++, per direction.

Prompts are rendered and tokenized locally, so they match the training format exactly, and
generation runs offline with vLLM in a Modal function (``modal_generate.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from translategemma_finetune.data import (
    DEFAULT_EVAL_SPLITS,
    DIRECTIONS,
    EVAL_SETS,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    load_eval_set,
)
from translategemma_finetune.finetune import (
    DEFAULT_MODEL,
    DEFAULT_REPETITION_PENALTY,
    directions,
    get_eos_token_ids,
    maybe_apply_chat_template,
    metrics_by_direction,
)
from translategemma_finetune.templates import format_single_for_prediction


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="translategemma eval", description=__doc__.split("\n")[0])
    parser.add_argument(
        "data",
        help=f"{' or '.join(EVAL_SETS)}, a mix directory (e.g. data/fra-mos-mix), a .jsonl/.parquet "
        "file, or a Hub dataset with french/moore columns.",
    )
    parser.add_argument(
        "--split",
        help="Split to score. Defaults: "
        + ", ".join(f"{name} {split}" for name, split in DEFAULT_EVAL_SPLITS.items())
        + ", otherwise test.",
    )
    parser.add_argument("--direction", choices=[*DIRECTIONS, "both"], default="both")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model on Modal: an HF repo id or a /models/... path in the translategemma-models "
        "volume.",
    )
    parser.add_argument(
        "--adapter", help="LoRA adapter on Modal: an HF repo id or /models/... path."
    )
    parser.add_argument(
        "--tokenizer", default=DEFAULT_MODEL, help="Tokenizer used to render the prompts locally."
    )
    parser.add_argument("--max-samples", type=int, help="Score only the first N rows.")
    parser.add_argument("--output", type=Path, help="Write sources, references, predictions here.")

    prompt = parser.add_argument_group("prompt")
    prompt.add_argument(
        "--use-chat-template",
        type=lambda value: value.lower() in {"true", "1", "yes"},
        default=True,
        metavar="{true,false}",
        help="false uses the raw `user\\n{json}\\nmodel\\n` format (default: true).",
    )
    prompt.add_argument("--chat-template-path", help="Use this Jinja template file.")
    prompt.add_argument(
        "--extra-languages",
        nargs="+",
        default=["mos_Latn=Mossi"],
        metavar="CODE=NAME",
        help="Languages added to the chat template.",
    )
    prompt.add_argument("--max-prompt-tokens", type=int, default=256)

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--max-new-tokens", type=int, default=128)
    sampling.add_argument("--temperature", type=float, default=0.0, help="0 decodes greedily.")
    sampling.add_argument("--top-p", type=float, default=0.9)
    sampling.add_argument("--top-k", type=int, default=50)
    sampling.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="1 disables it (default); 1.05 stops most loops in Mooré outputs.",
    )
    return parser.parse_args(argv)


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    return tokenizer


def tokenize_prompts(
    tokenizer: Any, rows: dict[str, list[Any]], args: argparse.Namespace
) -> list[list[int]]:
    prompts = [
        format_single_for_prediction(
            source_text=source,
            source_lang_code=source_code,
            target_lang_code=target_code,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
        )
        for source, source_code, target_code in zip(
            rows[SOURCE_COLUMN], rows["source_lang_code"], rows["target_lang_code"], strict=True
        )
    ]
    # Token ids are passed as-is, so vLLM neither re-renders nor adds a second <bos>.
    return tokenizer(
        prompts,
        truncation=True,
        max_length=args.max_prompt_tokens,
        add_special_tokens=False,
    )["input_ids"]


def build_sampling(args: argparse.Namespace, stop_token_ids: list[int]) -> dict[str, Any]:
    sampling: dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "stop_token_ids": stop_token_ids,
        "repetition_penalty": args.repetition_penalty,
        "temperature": 0.0,  # greedy
    }
    if args.temperature > 0:
        sampling.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})
    return sampling


def generate_predictions(
    prompt_token_ids: list[list[int]], stop_token_ids: list[int], args: argparse.Namespace
) -> list[str]:
    import modal

    from translategemma_finetune.modal_generate import app, generate

    with modal.enable_output(), app.run():
        predictions = generate.remote(
            model=args.model,
            adapter=args.adapter,
            prompt_token_ids=prompt_token_ids,
            sampling=build_sampling(args, stop_token_ids),
            max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        )
    return [prediction.strip() for prediction in predictions]


def write_predictions(
    path: Path,
    rows: dict[str, list[Any]],
    predictions: list[str],
    row_directions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for direction, source, reference, prediction in zip(
            row_directions, rows[SOURCE_COLUMN], rows[TARGET_COLUMN], predictions, strict=True
        ):
            record = {
                "direction": direction,
                "source": source,
                "reference": reference,
                "prediction": prediction,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    dataset = load_eval_set(args.data, args.split, args.direction)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    rows = dataset[:]
    row_directions = directions(rows)

    tokenizer = load_tokenizer(args.tokenizer)
    maybe_apply_chat_template(tokenizer, args)
    prompt_token_ids = tokenize_prompts(tokenizer, rows, args)
    stop_token_ids = get_eos_token_ids(tokenizer)
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    predictions = generate_predictions(prompt_token_ids, stop_token_ids or [], args)

    if args.output is not None:
        write_predictions(args.output, rows, predictions, row_directions)
    scores = metrics_by_direction(row_directions, predictions, rows[TARGET_COLUMN])
    for direction, metrics in scores.items():
        print(f"{direction}: BLEU {metrics['bleu']:.2f}, chrF++ {metrics['chrfpp']:.2f}")
    print(f"examples: {len(predictions)}")


if __name__ == "__main__":
    main()
