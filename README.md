# LLM_Router-serving

FastAPI server for ML routing and embedding.

## Implemented API

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/router/embed` | CLS-pooled, L2-normalized 1,024-dimensional BGE embedding |
| `POST` | `/router/classify` | ModernBERT `easy`/`medium`/`hard` classification |
| `GET` | `/health` | BGE and ModernBERT loading/readiness state |
| `POST` | `/escalate` | Validates a GCS feedback batch and submits the Vertex AI retraining pipeline |

The server does not query pgvector or select the actual LLM. Spring owns cache lookup and the
tier-to-model mapping.

## Local setup

Python 3.11 or 3.12 is required. Python 3.12 is the project default.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

Configure at least the approved ModernBERT package path and version in `.env`. BGE defaults to
`BAAI/bge-large-en-v1.5` revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`. Local execution
downloads it into the Hugging Face cache on first startup. The production Docker image downloads
only the safetensors and tokenizer files at build time and runs with Hugging Face offline mode, so a
GCP instance does not depend on Hugging Face availability during startup.

```bash
.venv/bin/uvicorn llm_router_serving.main:app --host 0.0.0.0 --port 8000
```

Model loading runs in the background. `/health` returns `503` until both models are loaded and
returns `200` afterward. Interactive OpenAPI documentation is available at `/docs` in environments
where that route is exposed.

If `LLM_ROUTER_INTERNAL_API_TOKEN` is set, Spring must send it as a bearer token:

```text
Authorization: Bearer <token>
```

For local development only, leaving the variable unset disables application-level authentication.
Production traffic must still be restricted by network policy and must configure the token or a
stronger workload identity/mTLS layer.

## Container deployment on GCP

The OCI image embeds the pinned BGE safetensors during `docker build` and starts as UID/GID 10001.
It therefore needs Hugging Face access only in the image build environment, not on the serving VM
or Kubernetes node.

```bash
docker build -t llm-router-serving:0.1.0 .
```

The persistent `var` directory must be mounted. Do not put GCP service-account keys in the image;
on a GCE-hosted container use the attached VM service account for GCS and Vertex AI access.

```bash
docker run --rm -p 8000:8000 \
  -e LLM_ROUTER_INTERNAL_API_TOKEN=replace-with-secret \
  -e LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI=gs://llm-router/models/router-modernbert/router-modernbert-base-routerbench-20260731-21c2f1b/approved \
  -e LLM_ROUTER_ROUTER_MODEL_VERSION=router-modernbert-base-routerbench-20260731-21c2f1b \
  -v /absolute/path/to/persistent-var:/app/var \
  llm-router-serving:0.1.0
```

Set `LLM_ROUTER_MODEL_DEVICE=cuda` only on a GPU node with a compatible NVIDIA container runtime;
otherwise the default `auto` selects CUDA when available and CPU otherwise. Readiness must probe
`GET /health`; it returns HTTP 503 until both BGE and ModernBERT have loaded.

## Approved model delivery

Model delivery is pull-based and there is no endpoint that swaps a loaded model. The Vertex AI
pipeline publishes a checksummed package to GCS, a human approves it, and the deployment is then
configured with the approved URI and version:

```text
Vertex AI Pipeline -> evaluate -> deploy gate
  -> gs://<bucket>/models/router-modernbert/<version>/candidate/
  -> modernbert-publish approve (LLM_Router repository)
  -> gs://<bucket>/models/router-modernbert/<version>/approved/ and CURRENT.json
  -> deployment sets LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI and LLM_ROUTER_ROUTER_MODEL_VERSION
  -> rolling restart: download, verify, load
  -> GET /health returns 200
```

Set exactly one model source. `LLM_ROUTER_ROUTER_MODEL_PATH` points at a local directory for
development; `LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI` points at the approved GCS prefix for
deployment. Setting both is rejected at startup.

During startup the process downloads the package into `LLM_ROUTER_ROUTER_MODEL_CACHE_DIR`
(default `var/models/<version>`) and verifies it against `model_manifest.json`: schema version,
`easy=0, medium=1, hard=2` label order, `model_version`, and the SHA-256 of every listed file. With
`LLM_ROUTER_ROUTER_MODEL_REQUIRE_APPROVED=true` (the default) a package whose `gate_status` is not
`approved` is refused, so a candidate can never be served by accident. Verification runs before the
directory is published into the cache, so a failed download leaves no usable model behind, and
`/health` stays 503.

Mount the cache directory on the same persistent volume as `var` to avoid re-downloading the
package on every container replacement. A rolling restart with `GET /health` as the readiness probe
is what makes the swap safe: a new instance only receives traffic once its model is verified and
loaded. Rolling back means pointing the two variables at the previous version and restarting;
`modernbert-publish rollback` moves the `CURRENT.json` pointer to match.

## Retraining configuration

`POST /escalate` is enabled only when all Vertex AI and GCS configuration variables in
[`.env.example`](.env.example) are set. `LLM_ROUTER_VERTEX_PIPELINE_TEMPLATE_URI` must be the
`gs://` URI of the compiled `feedback-retraining-pipeline.yaml` produced by the sibling `LLM_Router`
repository; the IR is read from GCS, not from the serving image.
`LLM_ROUTER_MODEL_CODE_COMMIT` must be the Git commit used by the retraining image and is included in
the candidate model version.

Submission uses `google.cloud.aiplatform.PipelineJob` against managed Vertex AI Pipelines. GKE and
self-hosted Kubeflow are not used. The Vertex `job_id` is derived deterministically from `batchId`,
so a duplicate submission is rejected by Vertex itself with `AlreadyExists` and returns the existing
run instead of starting a second training run.

The dataset and config URIs are sent as pipeline parameters and therefore **override** the defaults
compiled into the IR. They must stay on the v2 contract that the ML repository builds
(`minimum-sufficient-tier-v2` labeling, `routerbench-training-preprocess-v2-balanced` preprocessing,
`modernbert_t4_balanced_v2.json` training). Pointing `LLM_ROUTER_TRAINING_CONFIG_URI` or
`LLM_ROUTER_BASE_TRAINING_DATASET_URI` at a v1 artifact retrains the router on the old contract
without any error, so treat these as versioned artifacts and change them together.

### Filling in `.env` for the first time

`Settings.retraining_configuration_errors()` only checks that these variables are set and look like
`gs://` URIs. It does **not** check that the object actually exists. A well-formed URI pointing at
nothing passes local validation and only fails when Vertex tries to import that artifact at pipeline
submission time, so verify existence yourself before submitting:

```bash
gcloud storage objects describe "$LLM_ROUTER_GATE_RULES_URI"
```

What to do with each group of variables the first time you configure a project:

| Variable(s) | Before the first real training run | After `G-8` (real baseline exists) |
| --- | --- | --- |
| `LLM_ROUTER_GATE_RULES_URI` | Point at `config/<version>/deploy_gate_rules_smoke.json` — a loose gate that only checks package integrity and the candidate model version, so a first run can pass the gate and exercise the full publish path | Switch to a real `deploy_gate_rules.json` with thresholds derived from the actual baseline metrics |
| `LLM_ROUTER_BASELINE_METRICS_URI` | Point at `config/<version>/baseline_metrics_empty.json` (`{}`) — there is no approved model to compare against yet | Switch to `evaluations/<baseline-version>/metrics.json` from the approved baseline |
| `LLM_ROUTER_BASE_TRAINING_DATASET_URI` | Only required for the **retraining** pipeline (`/escalate`), not the initial training pipeline. If you have not uploaded the pre-built v2 `training.parquet` to GCS yet, `/escalate` submissions will fail at import time even though the app starts fine | Should already point at the dataset the current approved model was trained on |
| `LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI`, `LLM_ROUTER_ROUTER_MODEL_VERSION` | Safe to leave as a placeholder if no model has been approved yet. The embedding model still loads independently; the router slot reports `ERROR` and `GET /health` stays `503` until a real approved package exists | Point at the actual `approved/` package once one exists |
| `LLM_ROUTER_INTERNAL_API_TOKEN` | Any local secret works for development; leaving it unset disables auth entirely (dev only) | A real secret, provisioned via Secret Manager in deployment, never committed |
| `LLM_ROUTER_MODEL_CODE_COMMIT` | The Git commit actually baked into the training image you pushed (`git rev-parse --short HEAD` at build time), not whatever was last committed to this repo | Re-set on every training image rebuild |

`smoke` config filenames are a deliberate signal, not a suggestion — never point production traffic
at `deploy_gate_rules_smoke.json`. See `docs/ml/gcp-rollout-and-serving-integration-plan.md` in the
sibling `LLM_Router` repository for the commands that create and upload the GCS layout these
variables assume (bucket structure, `config/<version>/` prefix, service accounts, image digests).

The service persists `batchId`, request fingerprint, pipeline run ID, and state in
`LLM_ROUTER_RETRAINING_DB_PATH`. Mount that path on persistent storage in deployment; otherwise a
container replacement would lose the idempotency record.

If the process dies after claiming the lease but before recording the pipeline run ID, the next
`/escalate` reconciles it: a lease with no run ID that is older than
`LLM_ROUTER_RETRAINING_STALE_SUBMISSION_SECONDS` (default 900) is checked against Vertex, then
either adopted (the job did land, so it is not submitted twice) or released. Without this a single
crashed submission would block every future retraining request.

The current SQLite lease is suitable for one FastAPI replica. Before scaling `/escalate` across
multiple replicas, replace it with a shared transactional database or distributed lease so the
one-pipeline-at-a-time rule remains global.

GCS and Vertex AI access use the runtime's attached Google identity (Application Default
Credentials). Do not put service-account JSON keys in the image or repository. The runtime service
account needs `roles/aiplatform.user` to submit pipeline jobs, `roles/storage.objectViewer` to read
feedback batches and download approved packages, and `roles/iam.serviceAccountUser` on the pipeline
service account it passes to Vertex.

## Verification

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/python scripts/smoke_bge.py
```

The tests use fake inference, GCS, and Vertex AI adapters and do not download model weights or submit
cloud jobs.

## Documentation

- [FastAPI internal API contract](docs/api-contract.md)
- [Serving and retraining architecture](docs/serving-and-retraining-architecture.md)
- [FastAPI source upload to GCS guide](docs/gcs-source-upload-guide.md)
