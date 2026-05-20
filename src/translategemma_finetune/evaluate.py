from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from translategemma_finetune.finetune import DEFAULT_OUTPUT_DIR, ModelArguments
from translategemma_finetune.templates import format_single_for_prediction


DEFAULT_DATASET = "madoss/fr-mos-final-data"


@dataclass
class EvaluationDataArguments:
    dataset_name: str = field(
        default=DEFAULT_DATASET, metadata={"help": "Hugging Face dataset name."}
    )
    dataset_split: str = field(default="test", metadata={"help": "Dataset split to evaluate on."})
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
        default="mos", metadata={"help": "TranslateGemma target language code."}
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
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "Limit evaluation to the first N examples."}
    )


@dataclass
class EvaluationArguments:
    adapter_path: str | None = field(
        default=DEFAULT_OUTPUT_DIR,
        metadata={
            "help": "LoRA adapter path. Pass an empty string to evaluate --model_name directly."
        },
    )
    batch_size: int = field(default=4, metadata={"help": "Generation batch size."})
    max_new_tokens: int = field(default=128, metadata={"help": "Maximum generated tokens."})
    temperature: float = field(
        default=0.0, metadata={"help": "Sampling temperature. 0 uses greedy decoding."}
    )
    top_p: float = field(default=0.9, metadata={"help": "Top-p value when sampling."})
    top_k: int = field(default=50, metadata={"help": "Top-k value when sampling."})
    output_predictions_path: str | None = field(
        default=None, metadata={"help": "Optional JSONL path for sources, references, predictions."}
    )


def parse_args() -> tuple[ModelArguments, EvaluationDataArguments, EvaluationArguments]:
    from transformers import HfArgumentParser

    parser = HfArgumentParser(  # ty:ignore[invalid-argument-type]
        (ModelArguments, EvaluationDataArguments, EvaluationArguments)
    )
    return parser.parse_args_into_dataclasses()  # ty:ignore[invalid-return-type]


def load_dataset_split(data_args: EvaluationDataArguments) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split)
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
    if not data_args.use_chat_template or data_args.chat_template_path is None:
        return tokenizer

    path = Path(data_args.chat_template_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat template file does not exist: {path}")

    tokenizer.chat_template = path.read_text(encoding="utf-8")
    return tokenizer


def load_model(model_args: ModelArguments, eval_args: EvaluationArguments) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Use only one of --load_in_4bit or --load_in_8bit.")

    quantization_config = None
    if model_args.load_in_4bit or model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype="auto",
    )

    if eval_args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, eval_args.adapter_path)

    model.eval()
    return model


def iter_batches(dataset: Any, batch_size: int) -> Any:
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size]


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    model_args: ModelArguments,
    data_args: EvaluationDataArguments,
    eval_args: EvaluationArguments,
) -> tuple[list[str], list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    sources: list[str] = []

    for batch in iter_batches(dataset, eval_args.batch_size):
        batch_sources = batch[data_args.source_field]
        batch_references = batch[data_args.target_field]
        prompts = [
            format_single_for_prediction(
                source_text=source,
                source_lang_code=data_args.source_lang_code,
                target_lang_code=data_args.target_lang_code,
                tokenizer=tokenizer,
                use_chat_template=data_args.use_chat_template,
            )
            for source in batch_sources
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_args.max_seq_length,
            add_special_tokens=False,
        ).to(model.device)
        inputs["token_type_ids"] = inputs["input_ids"].new_zeros(inputs["input_ids"].shape)

        do_sample = eval_args.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": eval_args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": eval_args.temperature,
                    "top_p": eval_args.top_p,
                    "top_k": eval_args.top_k,
                }
            )

        outputs = model.generate(**inputs, **generation_kwargs)
        generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
        batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        sources.extend(batch_sources)
        references.extend(batch_references)
        predictions.extend(prediction.strip() for prediction in batch_predictions)

    return sources, references, predictions


def compute_chrfpp(predictions: list[str], references: list[str]) -> float:
    try:
        from sacrebleu.metrics import CHRF
    except ImportError as exc:
        raise ImportError(
            "Install sacrebleu to compute chrF++: `uv add sacrebleu` or `pip install sacrebleu`."
        ) from exc

    return CHRF(word_order=2).corpus_score(predictions, [references]).score


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
    model = load_model(model_args, eval_args)
    sources, references, predictions = generate_predictions(
        model,
        tokenizer,
        dataset,
        model_args,
        data_args,
        eval_args,
    )
    score = compute_chrfpp(predictions, references)
    write_predictions(eval_args.output_predictions_path, sources, references, predictions)

    print(f"chrF++: {score:.2f}")
    print(f"examples: {len(predictions)}")


if __name__ == "__main__":
    main()
