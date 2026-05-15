from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any

from translategemma_finetune.templates import format_for_training, format_single_for_prediction
from transformers import TrainingArguments as HFTrainingArguments

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
    preview_sample: bool = field(
        default=False,
        metadata={
            "help": "Format and print the first dataset row, then exit before loading the model."
        },
    )
    test_prompt: str | None = field(
        default=None, metadata={"help": "Generate from this source text after training."}
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
    warmup_ratio: int | float = field(
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
    save_adapter: bool = field(default=True, metadata={"help": "Save LoRA adapter and tokenizer."})
    save_merged_path: str | None = field(
        default=None,
        metadata={"help": f"Save a merged 16-bit model here, e.g. {DEFAULT_MERGED_DIR}."},
    )


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # ty:ignore[invalid-argument-type]
    return parser.parse_args_into_dataclasses()  # ty:ignore[invalid-return-type]


def load_training_dataset(data_args: DataArguments) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split)
    formatter = partial(
        format_for_training,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
        source_field=data_args.source_field,
        target_field=data_args.target_field,
    )
    return dataset.map(formatter, batched=True)


def load_model(model_args: ModelArguments) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        full_finetuning=model_args.full_finetuning,
    )


def add_lora_adapters(
    model: Any, model_args: ModelArguments, training_args: TrainingArguments
) -> Any:
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=list(DEFAULT_TARGET_MODULES),
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_mlp_modules=True,
        finetune_attention_modules=True,
        random_state=training_args.seed,
        use_rslora=False,
        loftq_config=None,
    )


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


def train(model: Any, tokenizer: Any, dataset: Any, training_args: TrainingArguments) -> Any:
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=build_sft_config(training_args),
    )
    return trainer.train()


def generate_sample(model: Any, tokenizer: Any, data_args: DataArguments) -> None:
    if not data_args.test_prompt:
        return

    from transformers import TextStreamer
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    prompt = format_single_for_prediction(
        source_text=data_args.test_prompt,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
    )
    print(prompt)
    model.generate(
        **tokenizer(prompt, return_tensors="pt").to("cuda"),
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def save_outputs(model: Any, tokenizer: Any, training_args: TrainingArguments) -> None:
    if training_args.save_adapter:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.save_merged_path:
        model.save_pretrained_merged(
            training_args.save_merged_path,
            tokenizer,
            save_method="merged_16bit",
        )

    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            raise ValueError("--push_to_hub requires --hub_model_id for model.push_to_hub().")
        model.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    model_args, data_args, training_args = parse_args()
    dataset = load_training_dataset(data_args)

    if data_args.preview_sample:
        print(dataset[0]["text"])
        return

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(model_args)
    model = add_lora_adapters(model, model_args, training_args)
    trainer_stats = train(model, tokenizer, dataset, training_args)
    print(trainer_stats)
    generate_sample(model, tokenizer, data_args)
    save_outputs(model, tokenizer, training_args)


if __name__ == "__main__":
    main()
