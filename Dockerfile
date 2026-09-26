FROM python:3.12-slim

ARG BGE_MODEL_ID=BAAI/bge-large-en-v1.5
ARG BGE_MODEL_REVISION=d4aa6901d3a41ba39fb536a557fa166f842b0e09

# Cross-encoder reranker weights. Baked in because the runtime sets
# HF_HUB_OFFLINE=1, so nothing can be fetched after the image is built.
# The revision is pinned: a threshold is only valid for the exact weights it was
# measured on, and the offline screening measured this commit.
ARG RERANK_MODEL_ID=cross-encoder/quora-distilroberta-base
ARG RERANK_MODEL_REVISION=f62e7a4b20b97195c2868e53ec59126df5eac743

# The weights ship in the image but the reranker stays off until a deployment
# sets LLM_ROUTER_RERANK_MODEL_ID=/opt/models/quora-distilroberta-base.
# No candidate passed the offline false-hit gate, so enabling it is a
# per-deployment decision rather than an image default.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LLM_ROUTER_EMBED_MODEL_ID=/opt/models/bge-large-en-v1.5 \
    LLM_ROUTER_EMBED_MODEL_REVISION= \
    LLM_ROUTER_EMBED_MODEL_VERSION=bge-large-en-v1.5 \
    LLM_ROUTER_EMBED_LOCAL_FILES_ONLY=true \
    LLM_ROUTER_RERANK_MODEL_VERSION=cross-encoder/quora-distilroberta-base@f62e7a4

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/check-r0-readiness.py ./scripts/check-r0-readiness.py

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${BGE_MODEL_ID}', revision='${BGE_MODEL_REVISION}', local_dir='/opt/models/bge-large-en-v1.5', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'])" \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${RERANK_MODEL_ID}', revision='${RERANK_MODEL_REVISION}', local_dir='/opt/models/quora-distilroberta-base', allow_patterns=['config.json','*.safetensors','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'])" \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home app \
    && mkdir -p /app/var \
    && chown -R 10001:10001 /app/var

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "llm_router_serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
