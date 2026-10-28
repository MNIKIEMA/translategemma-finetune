"""``translategemma train-modal``: run ``translategemma train`` on a Modal GPU.

Takes the same options as ``translategemma train``. A local ``--data`` (the mix directory by
default) is uploaded to the ``translategemma-data`` volume, and ``--output-dir`` is written to
the ``translategemma-models`` volume, the one ``translategemma eval`` reads, so a finished
adapter is scored with ``--adapter /models/<output-dir>`` without downloading it:

    translategemma train-modal --output-dir fr-mos-lora --max-steps 600
    translategemma eval bouquet --adapter /models/fr-mos-lora

Base model downloads are cached in the ``translategemma-hf-cache`` volume. ``WANDB_API_KEY``
is forwarded from the local environment (or ``.env``). Set ``MODAL_GPU`` to change the GPU
type (default: A100-80GB). Pass ``--detach`` to keep training after the terminal closes.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import modal
from dotenv import load_dotenv

from translategemma_finetune.data import DEFAULT_MIX_DIR

# Before the forwarded secret below reads the environment.
load_dotenv()

TORCH_VERSION = "2.10.0"
GPU = os.environ.get("MODAL_GPU", "A100-80GB")
DATA = "/data"
MODELS = "/models"
HF_HOME = "/root/.cache/huggingface"
# Modal caps a function at 24 hours.
TIMEOUT = 24 * 60 * 60
FORWARDED_ENV = ("WANDB_API_KEY", "WANDB_ENTITY")
# Path options of `translategemma train` that are moved onto the models volume.
OUTPUT_OPTIONS = ("--output-dir", "--output_dir", "--save-merged-path", "--save_merged_path")
DEFAULT_OUTPUT_DIR = "translategemma-4b-it-fr-mos-lora"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu128")
    # The versions the local training environment is tested with.
    .uv_pip_install(
        "accelerate==1.13.0",
        "bitsandbytes==0.49.2",
        "datasets==4.8.5",
        "peft==0.19.1",
        "pillow",
        "python-dotenv",
        "sacrebleu==2.6.0",
        "transformers==5.9.0",
        "trl==1.4.0",
        "wandb==0.27.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("translategemma_finetune")
)

app = modal.App("translategemma-train")
data_volume = modal.Volume.from_name("translategemma-data", create_if_missing=True)
models = modal.Volume.from_name("translategemma-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("translategemma-hf-cache", create_if_missing=True)
forwarded = modal.Secret.from_dict(
    {key: os.environ[key] for key in FORWARDED_ENV if key in os.environ}
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA: data_volume, MODELS: models, HF_HOME: hf_cache},
    secrets=[modal.Secret.from_name("huggingface"), forwarded],
    timeout=TIMEOUT,
)
def train(argv: list[str]) -> None:
    from translategemma_finetune.finetune import main

    try:
        main(argv)
    finally:
        # Keeps the checkpoints written so far when training fails or times out.
        models.commit()


def option_value(argv: list[str], names: tuple[str, ...]) -> tuple[int, str] | None:
    """Index and value of the last ``--name value`` or ``--name=value`` in ``argv``."""
    found = None
    for index, arg in enumerate(argv):
        name, equals, value = arg.partition("=")
        if name not in names:
            continue
        if equals:
            found = (index, value)
        elif index + 1 < len(argv):
            found = (index, argv[index + 1])
    return found


def set_option(argv: list[str], names: tuple[str, ...], value: str) -> list[str]:
    """Replace the value of an option, or append it with the first of ``names``."""
    found = option_value(argv, names)
    if found is None:
        return [*argv, names[0], value]
    index, _ = found
    if "=" in argv[index]:
        return [*argv[:index], f"{argv[index].partition('=')[0]}={value}", *argv[index + 1 :]]
    return [*argv[: index + 1], value, *argv[index + 2 :]]


def volume_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a relative path without '..': {value!r}")
    return path


def upload_data(argv: list[str]) -> list[str]:
    """Upload a local ``--data`` to the data volume and point ``--data`` at it."""
    found = option_value(argv, ("--data",))
    data = found[1] if found else DEFAULT_MIX_DIR
    path = Path(data)
    if not path.exists():
        if data == DEFAULT_MIX_DIR:
            raise FileNotFoundError(
                f"{data} does not exist. Build the mix with `translategemma mix`."
            )
        return argv  # A Hub dataset id, loaded in the container.
    remote = f"/{path.resolve().name}"
    with data_volume.batch_upload(force=True) as batch:
        if path.is_dir():
            batch.put_directory(str(path), remote)
        else:
            batch.put_file(str(path), remote)
    print(f"Uploaded {data} to the translategemma-data volume as {remote}")
    return set_option(argv, ("--data",), f"{DATA}{remote}")


def remote_argv(argv: list[str]) -> list[str]:
    argv = upload_data(argv)
    if option_value(argv, OUTPUT_OPTIONS[:2]) is None:
        argv = [*argv, "--output-dir", DEFAULT_OUTPUT_DIR]
    for names in (OUTPUT_OPTIONS[:2], OUTPUT_OPTIONS[2:]):
        found = option_value(argv, names)
        if found is not None:
            argv = set_option(argv, names, f"{MODELS}/{volume_path(found[1], names[0])}")
    return argv


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        print(__doc__)
        print("Training options: `translategemma train --help`.")
        return
    detach = "--detach" in argv
    argv = remote_argv([arg for arg in argv if arg != "--detach"])
    print("translategemma train " + " ".join(argv))
    with modal.enable_output(), app.run(detach=detach):
        if detach:
            call = train.spawn(argv)
            print(f"Training continues on Modal after this exits (call {call.object_id}).")
        else:
            train.remote(argv)
