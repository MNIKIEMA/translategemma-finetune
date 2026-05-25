from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any

from translategemma_finetune.templates import format_for_training, format_single_for_prediction
from transformers import TrainerCallback, TrainingArguments as HFTrainingArguments

DEFAULT_MODEL = "google/translategemma-4b-it"
DEFAULT_DATASET = "madoss/fr-mos-final-data"
DEFAULT_OUTPUT_DIR = "outputs/translategemma-4b-it-fr-mos-lora"
DEFAULT_MERGED_DIR = "translategemma-4b-it_16bit"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class ModelArguments:
    model_name: str = field(default=DEFAULT_MODEL, metadata={"help": "Base model to fine-tune."})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum sequence length."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    load_in_4bit: bool = field(default=True, metadata={"help": "Load the base model in 4-bit."})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load the base model in 8-bit."})
    full_finetuning: bool = field(
        default=False, metadata={"help": "Run full fine-tuning instead of PEFT."}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=DEFAULT_DATASET, metadata={"help": "Hugging Face dataset name."}
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split to train on."})
    validation_split: str | None = field(
        default="validation",
        metadata={"help": "Dataset split to use for validation. Pass an empty string to disable."},
    )
    test_split: str | None = field(
        default=None,
        metadata={"help": "Dataset split to evaluate after training. Pass e.g. --test_split test."},
    )
    source_field: str = field(
        default="french", metadata={"help": "Dataset column containing source text."}
    )
    target_field: str = field(
        default="moore", metadata={"help": "Dataset column containing target text."}
    )
    source_lang_code: str = field(
        default="fr", metadata={"help": "TranslateGemma source language code."}
    )
    target_lang_code: str = field(
        default="mos", metadata={"help": "TranslateGemma target language code."}
    )
    use_chat_template: bool = field(
        default=True, metadata={"help": "Format examples with processor.apply_chat_template()."}
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional local Jinja chat template path used when --use_chat_template is true."
        },
    )
    preview_sample: bool = field(
        default=False,
        metadata={
            "help": "Format and print the first dataset row, then exit before loading the model."
        },
    )
    test_prompt: str | None = field(
        default=None, metadata={"help": "Generate from this source text after training."}
    )
    wandb_eval_samples: int = field(
        default=4,
        metadata={"help": "Number of validation predictions to log to W&B after evaluation."},
    )
    wandb_eval_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new tokens for W&B validation prediction samples."},
    )
    test_batch_size: int = field(
        default=4, metadata={"help": "Post-training test generation batch size."}
    )
    test_max_new_tokens: int = field(
        default=128, metadata={"help": "Maximum new tokens for post-training test generation."}
    )
    max_test_samples: int | None = field(
        default=None,
        metadata={"help": "Limit post-training test evaluation to the first N examples."},
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    output_dir: str = field(
        default=DEFAULT_OUTPUT_DIR, metadata={"help": "Directory for adapter checkpoints."}
    )
    max_steps: int = field(default=60, metadata={"help": "Maximum training steps."})
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Per-device training batch size."}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Gradient accumulation steps."}
    )
    warmup_ratio: float = field(
        default=0.08, metadata={"help": "Warmup ratio if less than 1 or steps if int > 1."}
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate."})
    logging_steps: float = field(default=1, metadata={"help": "Logging interval in steps."})
    optim: str = field(default="adamw_8bit", metadata={"help": "Optimizer name."})
    weight_decay: float = field(default=0.001, metadata={"help": "Weight decay."})
    lr_scheduler_type: str = field(
        default="linear", metadata={"help": "Learning rate scheduler type."}
    )
    seed: int = field(default=3407, metadata={"help": "Random seed."})
    report_to: str | None = field(
        default="none", metadata={"help": "Experiment trackers to report to."}
    )
    project: str | None = field(
        default=None,
        metadata={"help": "Optional W&B project name. Equivalent to setting WANDB_PROJECT."},
    )
    save_adapter: bool = field(default=True, metadata={"help": "Save LoRA adapter and processor."})
    save_merged_path: str | None = field(
        default=None,
        metadata={"help": f"Save a merged 16-bit model here, e.g. {DEFAULT_MERGED_DIR}."},
    )
    early_stopping_patience: int | None = field(
        default=None,
        metadata={
            "help": "Enable early stopping after this many evaluations without improvement."
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum metric improvement required for early stopping."},
    )


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # ty:ignore[invalid-argument-type]
    return parser.parse_args_into_dataclasses()  # ty:ignore[invalid-return-type]


def load_dataset_split(dataset_name: str, split: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)


def format_training_dataset(dataset: Any, data_args: DataArguments, processor: Any) -> Any:
    formatter = partial(
        format_for_training,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
        source_field=data_args.source_field,
        target_field=data_args.target_field,
        tokenizer=processor,
        use_chat_template=data_args.use_chat_template,
    )
    return dataset.map(formatter, batched=True)


def load_processor(model_args: ModelArguments) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_args.model_name)
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = model_args.max_seq_length
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def maybe_apply_chat_template(processor: Any, data_args: DataArguments) -> Any:
    if not data_args.use_chat_template or data_args.chat_template_path is None:
        return processor

    path = Path(data_args.chat_template_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat template file does not exist: {path}")

    processor.chat_template = path.read_text(encoding="utf-8")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.chat_template = processor.chat_template
    return processor


def get_eos_token_ids(tokenizer: Any) -> list[int] | int | None:
    eos_token_id = tokenizer.eos_token_id
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_token_ids = [
        token_id
        for token_id in (eos_token_id, end_of_turn_id)
        if isinstance(token_id, int) and token_id >= 0
    ]
    if not eos_token_ids:
        return eos_token_id
    return list(dict.fromkeys(eos_token_ids))


def get_torch_dtype(training_args: TrainingArguments) -> Any:
    import torch

    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return torch.float32


def load_model(model_args: ModelArguments, training_args: TrainingArguments) -> Any:
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Use only one of --load_in_4bit or --load_in_8bit.")
    if model_args.full_finetuning and (model_args.load_in_4bit or model_args.load_in_8bit):
        raise ValueError("Full fine-tuning requires --no_load_in_4bit and --load_in_8bit false.")

    quantization_config = None
    torch_dtype = get_torch_dtype(training_args)
    if model_args.load_in_4bit or model_args.load_in_8bit:
        quantization_kwargs = {
            "load_in_4bit": model_args.load_in_4bit,
            "load_in_8bit": model_args.load_in_8bit,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if torch_dtype != "auto":
            quantization_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
        quantization_config = BitsAndBytesConfig(**quantization_kwargs)

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch_dtype,
    )
    model.config.use_cache = False
    return model


def add_lora_adapters(
    model: Any, model_args: ModelArguments, training_args: TrainingArguments
) -> Any:
    if model_args.full_finetuning:
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if model_args.load_in_4bit or model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(DEFAULT_TARGET_MODULES),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_sft_config(training_args: TrainingArguments) -> Any:
    from trl import SFTConfig

    sft_fields = {sft_field.name for sft_field in fields(SFTConfig)}
    sft_kwargs = {
        field_name: getattr(training_args, field_name)
        for field_name in sft_fields
        if hasattr(training_args, field_name)
    }
    sft_kwargs["dataset_text_field"] = "text"
    return SFTConfig(**sft_kwargs)


class WandbPredictionCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        raw_eval_dataset: Any | None,
        data_args: DataArguments,
    ) -> None:
        self.processor = processor
        self.data_args = data_args
        self.samples = []
        sample_count = max(data_args.wandb_eval_samples, 0)
        if raw_eval_dataset is None or sample_count == 0:
            return
        for index in range(min(sample_count, len(raw_eval_dataset))):
            row = raw_eval_dataset[index]
            self.samples.append(
                {
                    "index": index,
                    "source": row[data_args.source_field],
                    "reference": row[data_args.target_field],
                }
            )

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if model is None or not self.samples or not state.is_world_process_zero:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prompts = [
            format_single_for_prediction(
                source_text=sample["source"],
                source_lang_code=self.data_args.source_lang_code,
                target_lang_code=self.data_args.target_lang_code,
                tokenizer=self.processor,
                use_chat_template=self.data_args.use_chat_template,
            )
            for sample in self.samples
        ]
        max_length = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_length, int) or max_length > 100_000:
            max_length = None
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        ).to(model.device)

        was_training = model.training
        model.eval()
        import torch

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.data_args.wandb_eval_max_new_tokens,
                do_sample=False,
                eos_token_id=get_eos_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
        if was_training:
            model.train()

        generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
        decoder = getattr(self.processor, "batch_decode", None) or tokenizer.batch_decode
        predictions = [
            prediction.strip()
            for prediction in decoder(generated_tokens, skip_special_tokens=True)
        ]
        table = wandb.Table(columns=["step", "index", "source", "reference", "prediction"])
        for sample, prediction in zip(self.samples, predictions, strict=True):
            table.add_data(
                state.global_step,
                sample["index"],
                sample["source"],
                sample["reference"],
                prediction,
            )
        wandb.log({"eval/predictions": table}, step=state.global_step)


def build_callbacks(
    training_args: TrainingArguments,
    data_args: DataArguments,
    processor: Any,
    raw_eval_dataset: Any | None,
    eval_dataset: Any | None,
) -> list[Any]:
    callbacks = []
    if data_args.wandb_eval_samples > 0:
        callbacks.append(WandbPredictionCallback(processor, raw_eval_dataset, data_args))
    if training_args.early_stopping_patience is None:
        return callbacks
    if training_args.early_stopping_patience <= 0:
        raise ValueError("--early_stopping_patience must be greater than 0.")
    if eval_dataset is None:
        raise ValueError("--early_stopping_patience requires a validation split.")
    if not training_args.load_best_model_at_end:
        raise ValueError(
            "--early_stopping_patience requires --load_best_model_at_end true. "
            "Set --metric_for_best_model eval_loss unless you want a different validation metric."
        )

    from transformers import EarlyStoppingCallback

    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        )
    )
    return callbacks


def train(
    model: Any,
    processor: Any,
    dataset: Any,
    eval_dataset: Any | None,
    training_args: TrainingArguments,
    data_args: DataArguments,
    raw_eval_dataset: Any | None,
) -> Any:
    from trl import SFTTrainer

    sft_config = build_sft_config(training_args)
    callbacks = build_callbacks(
        training_args,
        data_args,
        processor,
        raw_eval_dataset,
        eval_dataset,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=callbacks,
    )
    return trainer.train()


def iter_batches(dataset: Any, batch_size: int) -> Any:
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size]


def generate_predictions(
    model: Any,
    processor: Any,
    raw_dataset: Any,
    data_args: DataArguments,
) -> tuple[list[str], list[str], list[str]]:
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    decoder = getattr(processor, "batch_decode", None) or tokenizer.batch_decode
    predictions: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_length, int) or max_length > 100_000:
        max_length = None

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in iter_batches(raw_dataset, data_args.test_batch_size):
            batch_sources = batch[data_args.source_field]
            batch_references = batch[data_args.target_field]
            prompts = [
                format_single_for_prediction(
                    source_text=source,
                    source_lang_code=data_args.source_lang_code,
                    target_lang_code=data_args.target_lang_code,
                    tokenizer=processor,
                    use_chat_template=data_args.use_chat_template,
                )
                for source in batch_sources
            ]
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
                add_special_tokens=False,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=data_args.test_max_new_tokens,
                do_sample=False,
                eos_token_id=get_eos_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
            batch_predictions = decoder(generated_tokens, skip_special_tokens=True)
            sources.extend(batch_sources)
            references.extend(batch_references)
            predictions.extend(prediction.strip() for prediction in batch_predictions)

    if was_training:
        model.train()
    return sources, references, predictions


def compute_translation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    from sacrebleu.metrics import BLEU, CHRF

    return {
        "bleu": BLEU().corpus_score(predictions, [references]).score,
        "chrfpp": CHRF(word_order=2).corpus_score(predictions, [references]).score,
    }


def evaluate_test_set(model: Any, processor: Any, data_args: DataArguments) -> None:
    if not data_args.test_split:
        return

    raw_test_dataset = load_dataset_split(data_args.dataset_name, data_args.test_split)
    if data_args.max_test_samples is not None:
        raw_test_dataset = raw_test_dataset.select(
            range(min(data_args.max_test_samples, len(raw_test_dataset)))
        )

    sources, references, predictions = generate_predictions(
        model,
        processor,
        raw_test_dataset,
        data_args,
    )
    metrics = compute_translation_metrics(predictions, references)
    print(f"test BLEU: {metrics['bleu']:.2f}")
    print(f"test chrF++: {metrics['chrfpp']:.2f}")
    print(f"test examples: {len(predictions)}")

    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    table = wandb.Table(columns=["index", "source", "reference", "prediction"])
    for index, (source, reference, prediction) in enumerate(
        zip(sources, references, predictions, strict=True)
    ):
        table.add_data(index, source, reference, prediction)
    wandb.log(
        {
            "test/bleu": metrics["bleu"],
            "test/chrfpp": metrics["chrfpp"],
            "test/examples": len(predictions),
            "test/predictions": table,
        }
    )


def generate_sample(model: Any, processor: Any, data_args: DataArguments) -> None:
    if not data_args.test_prompt:
        return

    prompt = format_single_for_prediction(
        source_text=data_args.test_prompt,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
        tokenizer=processor,
        use_chat_template=data_args.use_chat_template,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    inputs = processor(text=prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    model.eval()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=get_eos_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    decoder = getattr(processor, "batch_decode", None) or processor.tokenizer.batch_decode
    print(decoder(generated_tokens, skip_special_tokens=True)[0].strip())


def save_outputs(model: Any, processor: Any, training_args: TrainingArguments) -> None:
    if training_args.save_adapter:
        model.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

    if training_args.save_merged_path:
        if not hasattr(model, "merge_and_unload"):
            raise ValueError("--save_merged_path requires a PEFT LoRA model.")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(training_args.save_merged_path)
        processor.save_pretrained(training_args.save_merged_path)

    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            raise ValueError("--push_to_hub requires --hub_model_id for model.push_to_hub().")
        model.push_to_hub(training_args.hub_model_id)
        processor.push_to_hub(training_args.hub_model_id)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    model_args, data_args, training_args = parse_args()
    if training_args.project:
        os.environ["WANDB_PROJECT"] = training_args.project
    raw_dataset = load_dataset_split(data_args.dataset_name, data_args.dataset_split)

    if data_args.preview_sample:
        processor = load_processor(model_args)
        maybe_apply_chat_template(processor, data_args)
        dataset = format_training_dataset(raw_dataset, data_args, processor)
        print(dataset[0]["text"])
        return

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    processor = load_processor(model_args)
    maybe_apply_chat_template(processor, data_args)
    model = load_model(model_args, training_args)
    dataset = format_training_dataset(raw_dataset, data_args, processor)
    raw_eval_dataset = (
        load_dataset_split(data_args.dataset_name, data_args.validation_split)
        if data_args.validation_split
        else None
    )
    eval_dataset = (
        format_training_dataset(raw_eval_dataset, data_args, processor)
        if raw_eval_dataset is not None
        else None
    )
    model = add_lora_adapters(model, model_args, training_args)
    trainer_stats = train(
        model,
        processor,
        dataset,
        eval_dataset,
        training_args,
        data_args,
        raw_eval_dataset,
    )
    print(trainer_stats)
    evaluate_test_set(model, processor, data_args)
    generate_sample(model, processor, data_args)
    save_outputs(model, processor, training_args)


if __name__ == "__main__":
    main()
