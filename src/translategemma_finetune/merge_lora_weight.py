from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from translategemma_finetune.finetune import DEFAULT_MERGED_DIR, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR


DTypeName = Literal["auto", "float16", "bfloat16", "float32"]


@dataclass
class MergeArguments:
    model_name: str = field(
        default=DEFAULT_MODEL,
        metadata={"help": "Base model used to train the LoRA adapter."},
    )
    adapter_path: str = field(
        default=DEFAULT_OUTPUT_DIR,
        metadata={"help": "Path to the trained LoRA adapter directory."},
    )
    output_dir: str = field(
        default=DEFAULT_MERGED_DIR,
        metadata={"help": "Directory where the merged model will be saved."},
    )
    dtype: DTypeName = field(
        default="bfloat16",
        metadata={"help": "Dtype used to load and save the merged model."},
    )
    max_shard_size: str = field(
        default="5GB",
        metadata={"help": "Maximum checkpoint shard size passed to save_pretrained()."},
    )
    safe_serialization: bool = field(
        default=True,
        metadata={"help": "Save weights as safetensors when supported."},
    )


def parse_args() -> MergeArguments:
    from transformers import HfArgumentParser

    parser = HfArgumentParser(MergeArguments)
    return parser.parse_args_into_dataclasses()[0]  # ty:ignore[invalid-return-type]


def get_torch_dtype(dtype: DTypeName) -> Any:
    if dtype == "auto":
        return "auto"

    import torch

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtypes[dtype]


def load_processor(args: MergeArguments) -> Any:
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(args.adapter_path)
    except OSError:
        return AutoProcessor.from_pretrained(args.model_name)


def load_merged_model(args: MergeArguments) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype=get_torch_dtype(args.dtype),
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    return model.merge_and_unload()


def save_merged_model(model: Any, processor: Any, args: MergeArguments) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        output_path,
        max_shard_size=args.max_shard_size,
        safe_serialization=args.safe_serialization,
    )
    processor.save_pretrained(output_path)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()
    processor = load_processor(args)
    merged_model = load_merged_model(args)
    save_merged_model(merged_model, processor, args)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
