FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG PROJECT_DIR=/workspace/translategemma-finetune

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="${PROJECT_DIR}/.venv/bin:${PATH}" \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets

WORKDIR ${PROJECT_DIR}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev

RUN mkdir -p ${PROJECT_DIR}/outputs /workspace/cache/huggingface

VOLUME ["/workspace/OpenSource/translategemma-finetune/outputs", "/cache/huggingface"]

CMD ["bash"]
