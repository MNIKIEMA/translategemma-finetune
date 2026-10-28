from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from sacrebleu.metrics import BLEU, CHRF

from translategemma_finetune.finetune import ModelArguments, get_eos_token_ids
from translategemma_finetune.templates import (
    add_languages_to_chat_template,
    format_single_for_prediction,
    parse_extra_languages,
)


DEFAULT_DATASET = "madoss/fr-mos-final-data"


@dataclass
class EvaluationDataArguments:
    dataset_name: str = field(
        default=DEFAULT_DATASET, metadata={"help": "Hugging Face dataset name."}
    )
    dataset_split: str = field(default="test", metadata={"help": "Dataset split to evaluate on."})
    data_files: str | None = field(
        default=None,
        metadata={
            "help": "Load only this file from --dataset_name, e.g. benchmark_data/sentence_level/"
            "dev/fra_Latn-mos_Latn.parquet for facebook/bouquet. Its rows are in split 'train'."
        },
    )
    source_field: str = field(
        default="french", metadata={"help": "Dataset column containing source text."}
    )
    target_field: str = field(
        default="moore", metadata={"help": "Dataset column containing reference translation."}
    )
    source_lang_code: str = field(
        default="fr", metadata={"help": "TranslateGemma source language code."}
    )
    target_lang_code: str = field(
        default="mos_Latn", metadata={"help": "TranslateGemma target language code."}
    )
    use_chat_template: bool = field(
        default=True, metadata={"help": "Format prompts with tokenizer.apply_chat_template()."}
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional local Jinja chat template path used when --use_chat_template is true."
        },
    )
    extra_languages: list[str] = field(
        default_factory=lambda: ["mos_Latn=Mossi"],
        metadata={
            "help": "Languages missing from the chat template, as CODE=Name entries. They are "
            "added to the template at load time."
        },
    )
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "Limit evaluation to the first N examples."}
    )


@dataclass
class EvaluationArguments:
    model: str | None = field(
        default=None,
        metadata={
            "help": "Model to generate with on Modal: an HF repo id or a /models/... path in the "
            "translategemma-models volume. Defaults to --model_name."
        },
    )
    adapter: str | None = field(
        default=None,
        metadata={"help": "Optional LoRA adapter on Modal: an HF repo id or a /models/... path."},
    )
    max_new_tokens: int = field(default=128, metadata={"help": "Maximum generated tokens."})
    temperature: float = field(
        default=0.0, metadata={"help": "Sampling temperature. 0 uses greedy decoding."}
    )
    top_p: float = field(default=0.9, metadata={"help": "Top-p value when sampling."})
    top_k: int = field(default=50, metadata={"help": "Top-k value when sampling."})
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty for repeated tokens during generation. 1.0 disables it."},
    )
    output_predictions_path: str | None = field(
        default=None, metadata={"help": "Optional JSONL path for sources, references, predictions."}
    )


def parse_args() -> tuple[ModelArguments, EvaluationDataArguments, EvaluationArguments]:
    from transformers import HfArgumentParser

    parser = HfArgumentParser(
        (ModelArguments, EvaluationDataArguments, EvaluationArguments)
    )
    return parser.parse_args_into_dataclasses()  # ty:ignore[invalid-return-type]


def load_dataset_split(data_args: EvaluationDataArguments) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(
        data_args.dataset_name, data_files=data_args.data_files, split=data_args.dataset_split
    )
    if data_args.max_eval_samples is not None:
        dataset = dataset.select(range(min(data_args.max_eval_samples, len(dataset))))
    return dataset


def load_tokenizer(model_args: ModelArguments) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = model_args.max_seq_length
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def maybe_apply_chat_template(tokenizer: Any, data_args: EvaluationDataArguments) -> Any:
    if not data_args.use_chat_template:
        return tokenizer

    chat_template = tokenizer.chat_template
    if data_args.chat_template_path is not None:
        path = Path(data_args.chat_template_path)
        if not path.exists():
            raise FileNotFoundError(f"Chat template file does not exist: {path}")
        chat_template = path.read_text(encoding="utf-8")

    tokenizer.chat_template = add_languages_to_chat_template(
        chat_template, parse_extra_languages(data_args.extra_languages)
    )
    return tokenizer


def tokenize_prompts(
    tokenizer: Any,
    sources: list[str],
    model_args: ModelArguments,
    data_args: EvaluationDataArguments,
) -> list[list[int]]:
    prompts = [
        format_single_for_prediction(
            source_text=source,
            source_lang_code=data_args.source_lang_code,
            target_lang_code=data_args.target_lang_code,
            tokenizer=tokenizer,
            use_chat_template=data_args.use_chat_template,
        )
        for source in sources
    ]
    # Token ids are passed as-is, so vLLM neither re-renders nor adds a second <bos>.
    return tokenizer(
        prompts,
        truncation=True,
        max_length=model_args.max_seq_length,
        add_special_tokens=False,
    )["input_ids"]


def build_sampling(eval_args: EvaluationArguments, stop_token_ids: list[int]) -> dict[str, Any]:
    sampling: dict[str, Any] = {
        "max_tokens": eval_args.max_new_tokens,
        "stop_token_ids": stop_token_ids,
        "repetition_penalty": eval_args.repetition_penalty,
        "temperature": 0.0,  # greedy
    }
    if eval_args.temperature > 0:
        sampling.update(
            {
                "temperature": eval_args.temperature,
                "top_p": eval_args.top_p,
                "top_k": eval_args.top_k,
            }
        )
    return sampling


def generate_predictions(
    prompt_token_ids: list[list[int]],
    stop_token_ids: list[int],
    model_args: ModelArguments,
    eval_args: EvaluationArguments,
) -> list[str]:
    import modal

    from translategemma_finetune.modal_generate import app, generate

    with modal.enable_output(), app.run():
        predictions = generate.remote(
            model=eval_args.model or model_args.model_name,
            adapter=eval_args.adapter,
            prompt_token_ids=prompt_token_ids,
            sampling=build_sampling(eval_args, stop_token_ids),
            max_model_len=model_args.max_seq_length + eval_args.max_new_tokens,
        )
    return [prediction.strip() for prediction in predictions]


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "bleu": BLEU().corpus_score(predictions, [references]).score,
        "chrfpp": CHRF(word_order=2).corpus_score(predictions, [references]).score,
    }


def write_predictions(
    path: str | None,
    sources: list[str],
    references: list[str],
    predictions: list[str],
) -> None:
    if path is None:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for source, reference, prediction in zip(sources, references, predictions, strict=True):
            output_file.write(
                json.dumps(
                    {
                        "source": source,
                        "reference": reference,
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    model_args, data_args, eval_args = parse_args()
    dataset = load_dataset_split(data_args)
    tokenizer = load_tokenizer(model_args)
    maybe_apply_chat_template(tokenizer, data_args)
    sources = dataset[data_args.source_field]
    references = dataset[data_args.target_field]
    prompt_token_ids = tokenize_prompts(tokenizer, sources, model_args, data_args)
    stop_token_ids = get_eos_token_ids(tokenizer)
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    predictions = generate_predictions(
        prompt_token_ids, stop_token_ids or [], model_args, eval_args
    )
    scores = compute_metrics(predictions, references)
    write_predictions(eval_args.output_predictions_path, sources, references, predictions)

    print(f"BLEU: {scores['bleu']:.2f}")
    print(f"chrF++: {scores['chrfpp']:.2f}")
    print(f"examples: {len(predictions)}")


if __name__ == "__main__":
    main()
