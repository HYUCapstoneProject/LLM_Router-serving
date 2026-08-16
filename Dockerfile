FROM python:3.12-slim

ARG BGE_MODEL_ID=BAAI/bge-large-en-v1.5
ARG BGE_MODEL_REVISION=d4aa6901d3a41ba39fb536a557fa166f842b0e09

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LLM_ROUTER_EMBED_MODEL_ID=/opt/models/bge-large-en-v1.5 \
    LLM_ROUTER_EMBED_MODEL_REVISION= \
    LLM_ROUTER_EMBED_MODEL_VERSION=bge-large-en-v1.5 \
    LLM_ROUTER_EMBED_LOCAL_FILES_ONLY=true

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${BGE_MODEL_ID}', revision='${BGE_MODEL_REVISION}', local_dir='/opt/models/bge-large-en-v1.5', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'])" \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home app \
    && mkdir -p /app/var \
    && chown -R 10001:10001 /app/var

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "llm_router_serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
