"""Serve a TranslateGemma model with vLLM's OpenAI-compatible server on Modal.

For a long-running endpoint (demos, apps). Evaluation uses the offline function in
``translategemma_finetune/modal_generate.py`` instead.

    uvx modal serve modal_serve.py    # ephemeral: stops with Ctrl-C
    uvx modal deploy modal_serve.py   # persistent: scales to zero when idle

Configure the deployment with environment variables when running the command:

    MODEL       HF repo id or a path in the ``translategemma-models`` volume, mounted at /models
                (default: google/translategemma-4b-it). Use a merged model for a fine-tune, or
    ADAPTER     a LoRA adapter (HF repo id or /models/... path) served on top of MODEL.
    VLLM_API_KEY  Requests must send this key. Without it the endpoint is public.
    MODAL_GPU   GPU type (default: A100-40GB).

Upload a local model to the volume with:

    uvx modal volume put translategemma-models outputs/translategemma-4b-it-fr-mos-lora /fr-mos-lora
"""

import os
import subprocess

import modal

VLLM_VERSION = "0.30.0"
GPU = os.environ.get("MODAL_GPU", "A100-40GB")
MODEL = os.environ.get("MODEL", "google/translategemma-4b-it")
ADAPTER = os.environ.get("ADAPTER")
API_KEY = os.environ.get("VLLM_API_KEY")
SERVED_MODEL_NAME = "translategemma"
MAX_MODEL_LEN = 1024
MAX_LORA_RANK = 128
MODELS = "/models"
HF_HOME = "/root/.cache/huggingface"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    # FlashInfer's sampler is JIT-compiled with nvcc, which this image does not have.
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("translategemma-vllm")
models = modal.Volume.from_name("translategemma-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("translategemma-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]
if API_KEY:
    secrets.append(modal.Secret.from_dict({"VLLM_API_KEY": API_KEY}))


def build_command() -> list[str]:
    command = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        # Text translation only: skip the memory reserved for image inputs.
        "--limit-mm-per-prompt",
        '{"image": 0}',
    ]
    if ADAPTER:
        # Requests for SERVED_MODEL_NAME use the adapter; "base" is the model without it.
        command += [
            "--served-model-name",
            "base",
            "--enable-lora",
            "--max-lora-rank",
            str(MAX_LORA_RANK),
            "--lora-modules",
            f"{SERVED_MODEL_NAME}={ADAPTER}",
        ]
    else:
        command += ["--served-model-name", SERVED_MODEL_NAME]
    if API_KEY:
        command += ["--api-key", API_KEY]
    return command


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS: models, HF_HOME: hf_cache},
    secrets=secrets,
    scaledown_window=5 * 60,
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve() -> None:
    subprocess.Popen(build_command())
