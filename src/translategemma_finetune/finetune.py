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
    use_chat_template: bool = field(
        default=True, metadata={"help": "Format examples with tokenizer.apply_chat_template()."}
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
    save_adapter: bool = field(default=True, metadata={"help": "Save LoRA adapter and tokenizer."})
    save_merged_path: str | None = field(
        default=None,
        metadata={"help": f"Save a merged 16-bit model here, e.g. {DEFAULT_MERGED_DIR}."},
    )


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # ty:ignore[invalid-argument-type]
    return parser.parse_args_into_dataclasses()  # ty:ignore[invalid-return-type]


def load_dataset_split(dataset_name: str, split: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)


def format_training_dataset(dataset: Any, data_args: DataArguments, tokenizer: Any) -> Any:
    formatter = partial(
        format_for_training,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
        source_field=data_args.source_field,
        target_field=data_args.target_field,
        tokenizer=tokenizer,
        use_chat_template=data_args.use_chat_template,
    )
    return dataset.map(formatter, batched=True)


def load_tokenizer(model_args: ModelArguments) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = model_args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def maybe_apply_chat_template(tokenizer: Any, data_args: DataArguments) -> Any:
    if not data_args.use_chat_template or data_args.chat_template_path is None:
        return tokenizer

    path = Path(data_args.chat_template_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat template file does not exist: {path}")

    tokenizer.chat_template = path.read_text(encoding="utf-8")
    return tokenizer


def get_torch_dtype(training_args: TrainingArguments) -> Any:
    import torch

    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return "auto"


def load_model(model_args: ModelArguments, training_args: TrainingArguments) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
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


class Gemma3TextCollator:
    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.collator(examples)
        batch["token_type_ids"] = batch["input_ids"].new_zeros(batch["input_ids"].shape)
        return batch


def build_data_collator(tokenizer: Any, sft_config: Any) -> Any:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    pad_token = sft_config.pad_token or tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=sft_config.completion_only_loss,
        padding_free=sft_config.padding_free,
        pad_to_multiple_of=sft_config.pad_to_multiple_of,
    )
    return Gemma3TextCollator(collator)


def train(model: Any, tokenizer: Any, dataset: Any, training_args: TrainingArguments) -> Any:
    from trl import SFTTrainer

    sft_config = build_sft_config(training_args)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_config,
        data_collator=build_data_collator(tokenizer, sft_config),
    )
    return trainer.train()


def generate_sample(model: Any, tokenizer: Any, data_args: DataArguments) -> None:
    if not data_args.test_prompt:
        return

    from transformers import TextStreamer

    prompt = format_single_for_prediction(
        source_text=data_args.test_prompt,
        source_lang_code=data_args.source_lang_code,
        target_lang_code=data_args.target_lang_code,
        tokenizer=tokenizer,
        use_chat_template=data_args.use_chat_template,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs["token_type_ids"] = inputs["input_ids"].new_zeros(inputs["input_ids"].shape)
    model.eval()
    model.generate(
        **inputs,
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
        if not hasattr(model, "merge_and_unload"):
            raise ValueError("--save_merged_path requires a PEFT LoRA model.")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(training_args.save_merged_path)
        tokenizer.save_pretrained(training_args.save_merged_path)

    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            raise ValueError("--push_to_hub requires --hub_model_id for model.push_to_hub().")
        model.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    model_args, data_args, training_args = parse_args()
    raw_dataset = load_dataset_split(data_args.dataset_name, data_args.dataset_split)

    if data_args.preview_sample:
        tokenizer = load_tokenizer(model_args)
        maybe_apply_chat_template(tokenizer, data_args)
        dataset = format_training_dataset(raw_dataset, data_args, tokenizer)
        print(dataset[0]["text"])
        return

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(model_args)
    maybe_apply_chat_template(tokenizer, data_args)
    model = load_model(model_args, training_args)
    dataset = format_training_dataset(raw_dataset, data_args, tokenizer)
    model = add_lora_adapters(model, model_args, training_args)
    trainer_stats = train(model, tokenizer, dataset, training_args)
    print(trainer_stats)
    generate_sample(model, tokenizer, data_args)
    save_outputs(model, tokenizer, training_args)


if __name__ == "__main__":
    main()
