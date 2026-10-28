"""Offline vLLM generation on Modal, used by ``translategemma-evaluate``.

The prompts are rendered and tokenized locally and passed as token ids; the function returns
the generated texts in the same order. Models and adapters are HF repo ids or paths in the
``translategemma-models`` volume, mounted at /models:

    uvx modal volume put translategemma-models outputs/translategemma-4b-it-fr-mos-lora /fr-mos-lora

Set ``MODAL_GPU`` to change the GPU type (default: A100-40GB).
"""

from __future__ import annotations

import os
from typing import Any

import modal

VLLM_VERSION = "0.30.0"
GPU = os.environ.get("MODAL_GPU", "A100-40GB")
MODELS = "/models"
HF_HOME = "/root/.cache/huggingface"
MAX_LORA_RANK = 128

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    # FlashInfer's sampler is JIT-compiled with nvcc, which this image does not have.
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("translategemma_finetune")
)

app = modal.App("translategemma-eval")
models = modal.Volume.from_name("translategemma-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("translategemma-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS: models, HF_HOME: hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * 60 * 60,
)
def generate(
    model: str,
    adapter: str | None,
    prompt_token_ids: list[list[int]],
    sampling: dict[str, Any],
    max_model_len: int,
) -> list[str]:
    import time

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    start = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        # Text translation only: skip the memory reserved for image inputs.
        limit_mm_per_prompt={"image": 0},
        enable_lora=adapter is not None,
        max_lora_rank=MAX_LORA_RANK,
    )
    print(f"Loaded {model} in {time.perf_counter() - start:.0f}s")

    lora_request = LoRARequest("adapter", 1, adapter) if adapter else None
    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids],
        SamplingParams(skip_special_tokens=True, **sampling),
        lora_request=lora_request,
    )
    seconds = time.perf_counter() - start
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        f"Generated {len(outputs):,} predictions in {seconds:.0f}s "
        f"({output_tokens / seconds:.0f} tokens/s)"
    )
    return [output.outputs[0].text for output in outputs]
