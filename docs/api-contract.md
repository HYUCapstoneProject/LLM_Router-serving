# FastAPI 내부 API 계약

상태: 구현 기준 초안

기준일: 2026-08-13

호출자: Spring Server

## 1. 범위

이 문서는 Spring Server가 `LLM_Router-serving` FastAPI Server에 호출하는 내부 API를
정의한다.

| Method | Path | 호출 시점 | 목적 |
| --- | --- | --- | --- |
| `POST` | `/router/embed` | cache 조회 전 항상 | 질문의 BGE embedding 생성 |
| `POST` | `/router/classify` | cache miss일 때만 | ModernBERT 기반 난이도 분류 |
| `GET` | `/health` | readiness 및 장애 진단 | embedding/router model 적재 상태 확인 |
| `POST` | `/escalate` | 유효 feedback 1,000건 게시 후 | Vertex AI 재학습 pipeline 제출 |

이 API는 사용자에게 직접 공개하지 않는다. Spring Server가 cache와 실제 LLM routing
정책을 소유하고, FastAPI Server는 embedding·분류 inference와 재학습 pipeline 제출을
담당한다.

## 2. 공통 계약

### 2-1. 전송 형식

- 요청과 응답의 `Content-Type`은 `application/json`이다. `/health` 요청에는 body가 없다.
- JSON 필드명은 Spring 계약에 맞춰 camelCase를 사용한다.
- 시간은 UTC ISO 8601 형식으로 전달한다. 예: `2026-08-12T15:30:00Z`.
- 알 수 없는 필드의 허용 여부는 구현 시 한 가지 정책으로 고정한다. 재학습 요청은 잘못된
  metadata를 조기에 찾기 위해 알 수 없는 필드를 거부하는 방식을 권장한다.

### 2-2. Request ID

Spring Server는 `/router/embed`, `/router/classify`, `/escalate` 요청마다
`X-Request-ID` header를 전달한다. FastAPI Server는 이 값을 로그와 metric에 보존하고
오류 응답의 `requestId`와 모든 응답의 `X-Request-ID` header로 반환한다. header가 없으면
FastAPI Server가 UUID를 생성한다.

### 2-3. 내부 인증

모든 endpoint는 외부 인터넷에 직접 노출하지 않는다. `LLM_ROUTER_INTERNAL_API_TOKEN`이
설정된 환경에서는 Spring이 `Authorization: Bearer <token>`을 전달해야 한다. 로컬 개발에서
변수가 없으면 application-level 인증을 생략할 수 있지만 운영에서는 network policy와
token 또는 더 강한 workload identity/mTLS 계층을 반드시 적용한다.

- 인증 정보 없음 또는 유효하지 않음: `401 Unauthorized`
- 인증은 되었으나 해당 작업 권한 없음: `403 Forbidden`
- credential, token과 질문 원문은 일반 로그에 남기지 않는다.

readiness probe가 `/health`를 호출하는 경우에는 애플리케이션 endpoint와 분리된 network
policy 또는 probe 전용 인증 예외를 명시적으로 구성한다.

### 2-4. 공통 오류 body

```json
{
  "error": "요청을 처리할 수 없습니다.",
  "code": "ROUTER_NOT_READY",
  "requestId": "req-123"
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `error` | string | Y | credential이나 내부 stack trace를 포함하지 않는 설명 |
| `code` | string | Y | Spring이 분기와 metric에 사용할 안정적인 오류 코드 |
| `requestId` | string | Y | 요청 상관관계 ID |

HTTP status와 `code`는 함께 사용한다. Spring은 사람이 읽는 `error` 문자열을 분기
조건으로 사용하지 않는다.

## 3. `POST /router/embed`

### 3-1. 목적과 호출 조건

질문의 embedding vector를 생성한다. cache 조회가 embedding을 선행 요구하므로 cache
hit 여부와 관계없이 모든 chat 요청에서 먼저 호출한다.

FastAPI Server는 vector만 생성한다. 응답 vector로 pgvector를 조회하고 cache hit 여부를
판정하는 주체는 Spring Gateway다.

### 3-2. Request

```json
{
  "message": "string"
}
```

| 필드 | 타입 | 필수 | 검증 |
| --- | --- | --- | --- |
| `message` | string | Y | 빈 문자열이 아니어야 한다. 최대 길이는 tokenizer 계약과 함께 확정한다. |

### 3-3. Success response

Status: `200 OK`

```json
{
  "embedding": [0.0123, -0.0456],
  "dim": 1024,
  "modelVersion": "bge-large-en-v1.5"
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `embedding` | number[] | Y | L2 정규화된 1,024차원 유한 실수 vector |
| `dim` | integer | Y | 항상 `1024` |
| `modelVersion` | string | Y | embedding model 및 preprocessing 계약 version |

Spring은 `modelVersion`을 `request_log`에 기록한다. embedding model 또는 preprocessing
계약을 변경하면 기존 cache vector와 새 vector를 같은 version처럼 비교하면 안 된다.

### 3-4. 고정해야 할 embedding 규격

아래 규격은 EXP-003 측정 코드와 완전히 같아야 한다. 한 항목이라도 달라지면 명시적인
오류 없이 cache 정확도만 낮아질 수 있다.

| 항목 | 값 | 이유 |
| --- | --- | --- |
| 차원 | `1024` | pgvector `vector(1024)`와 일치해야 한다. |
| L2 정규화 | 정규화해서 반환 | pgvector `<=>` cosine distance에서 `similarity = 1 - distance`를 동일한 threshold 의미로 사용하기 위해 필요하다. |
| pooling | CLS | 공식 `1_Pooling/config.json`과 Transformers 사용 예제가 CLS pooling을 사용한다. |
| BGE prefix | 사용하지 않음 | semantic cache는 질문과 질문의 대칭 비교다. 저장과 조회 중 한쪽에만 prefix를 붙이면 유사도가 낮아진다. |

model ID, revision, tokenizer, pooling, normalization, prefix와 출력 차원을 하나의
versioned embedding 계약으로 관리한다.

### 3-5. 오류

| Status | Code | 조건 |
| --- | --- | --- |
| `400 Bad Request` | `INVALID_MESSAGE` | `message`가 비어 있거나 허용 길이를 위반 |
| `401 Unauthorized` | `UNAUTHORIZED` | 내부 인증 실패 |
| `403 Forbidden` | `FORBIDDEN` | 호출 권한 없음 |
| `503 Service Unavailable` | `EMBED_MODEL_NOT_READY` | BGE model이 아직 적재되지 않았거나 error 상태 |
| `500 Internal Server Error` | `EMBEDDING_INFERENCE_FAILED` | 적재 완료 후 inference 실패 |

## 4. `POST /router/classify`

### 4-1. 목적과 호출 조건

cache miss 질문을 ModernBERT로 분류해 `easy`, `medium`, `hard` 중 하나를 반환한다.
cache hit이면 호출하지 않는다.

### 4-2. Request

```json
{
  "message": "string"
}
```

| 필드 | 타입 | 필수 | 검증 |
| --- | --- | --- | --- |
| `message` | string | Y | 빈 문자열이 아니어야 한다. 최대 길이와 truncation 관측 방법은 tokenizer 계약을 따른다. |

### 4-3. Success response

Status: `200 OK`

```json
{
  "tier": "easy",
  "model": "haiku",
  "confidence": 0.83,
  "modelVersion": "router-modernbert-base-routerbench-20260731-21c2f1b"
}
```

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `tier` | enum | Y | `easy`, `medium`, `hard` 중 하나 |
| `model` | enum | Y | `haiku`, `sonnet`, `opus` 중 하나인 참고값 |
| `confidence` | number | Y | 선택한 tier의 confidence, `0.0` 이상 `1.0` 이하 |
| `modelVersion` | string | Y | 판정을 생성한 immutable router model version |

ML label 순서는 `easy=0`, `medium=1`, `hard=2`로 고정한다.

`modelVersion`은 `router-{model}-{dataset}-{date}-{shortCommit}` 명명 규칙을 사용하며
Spring이 `request_log`에 저장한다. 이 값은 재학습 전후 model의 판정과 escalation 비율을
분리해 분석하는 데 필수다.

`confidence`가 낮아도 `tier`는 항상 반환한다. Spring은 confidence threshold로 routing을
분기하지 않고 기록과 사후 분석에만 사용한다.

`model`은 참고값이다. 실제 tier-to-model mapping의 소유자는 Spring의 versioned
`application.yml` 설정이다. FastAPI 응답과 Spring 설정이 다르면 Spring 설정이 우선한다.
mapping만 바뀐 경우 ModernBERT를 재학습하거나 FastAPI를 재배포할 필요가 없어야 한다.

### 4-4. 오류

| Status | Code | 조건 |
| --- | --- | --- |
| `400 Bad Request` | `INVALID_MESSAGE` | `message`가 비어 있거나 허용 길이를 위반 |
| `401 Unauthorized` | `UNAUTHORIZED` | 내부 인증 실패 |
| `403 Forbidden` | `FORBIDDEN` | 호출 권한 없음 |
| `503 Service Unavailable` | `ROUTER_MODEL_NOT_READY` | ModernBERT가 아직 적재되지 않았거나 error 상태 |
| `500 Internal Server Error` | `ROUTER_INFERENCE_FAILED` | 적재 완료 후 inference 실패 |

## 5. `GET /health`

### 5-1. 목적

FastAPI process와 BGE·ModernBERT model의 적재 상태를 반환한다. 배포 readiness 판정과
로컬 통합 장애 진단에 사용한다.

BGE-large와 ModernBERT는 process 시작 후 적재 시간이 필요하므로 process가 실행 중인
상태와 실제 요청을 받을 준비가 된 상태를 구분한다.

### 5-2. Ready response

Status: `200 OK`

```json
{
  "status": "ok",
  "embedModel": "loaded",
  "routerModel": "loaded"
}
```

### 5-3. Not-ready response

Status: `503 Service Unavailable`

```json
{
  "status": "not_ready",
  "embedModel": "loading",
  "routerModel": "loaded"
}
```

| 필드 | 타입 | 가능한 값 | 설명 |
| --- | --- | --- | --- |
| `status` | enum | `ok`, `not_ready` | 전체 readiness |
| `embedModel` | enum | `loaded`, `loading`, `error` | BGE 적재 상태 |
| `routerModel` | enum | `loaded`, `loading`, `error` | ModernBERT 적재 상태 |

두 model 중 하나라도 `loaded`가 아니면 `status=not_ready`와 HTTP `503`을 반환한다.
health 응답에 실제 embedding/router model version을 추가할지는 구현 전에 확정한다.

## 6. `POST /escalate`

### 6-1. 목적과 호출 조건

유효한 Escalation Log가 1,000건 누적되면 Spring Server가 immutable feedback batch를
GCS에 게시한 뒤 FastAPI Server에 Vertex AI 재학습 pipeline 제출을 요청한다.

1,000건은 raw DB row 수가 아니라 중복 제거, PII 제거, 사용자 만족 여부 등 합의된 export
조건을 통과한 유효 feedback 수다. Spring은 event 1,000건을 request body에 넣지 않고
GCS 객체 위치와 metadata만 전달한다.

### 6-2. Request

```json
{
  "batchId": "feedback-20260812-0001",
  "schemaVersion": "feedback-export-v0.1",
  "manifestUri": "gs://llm-router/datasets/feedback/feedback-20260812-0001/manifest.json",
  "eventsUri": "gs://llm-router/datasets/feedback/feedback-20260812-0001/events.parquet",
  "eventCount": 1000,
  "requestedAt": "2026-08-12T15:30:00Z"
}
```

| 필드 | 타입 | 필수 | 검증 |
| --- | --- | --- | --- |
| `batchId` | string | Y | immutable batch 식별자이자 멱등성 키 |
| `schemaVersion` | string | Y | FastAPI와 ML pipeline이 지원하는 feedback schema |
| `manifestUri` | string | Y | 접근 가능한 `gs://` manifest 객체 URI |
| `eventsUri` | string | Y | 접근 가능한 `gs://` Parquet 객체 URI |
| `eventCount` | integer | Y | `1000`, request·manifest·Parquet 행 수가 일치해야 함 |
| `requestedAt` | string | Y | UTC ISO 8601 요청 생성 시각 |

manifest는 다음 필드를 모두 포함해야 한다. 이 목록은 ML pipeline의
`REQUIRED_MANIFEST_FIELDS`(`LLM_Router` 저장소의
`ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py`)와 같아야 하며,
한쪽만 바꾸면 `/escalate`가 202로 받아들인 batch가 pipeline 안에서 실패한다.

```text
schema_version
batch_id
created_at
window_start
window_end
event_count
data_format
data_uri
sha256
producer_version
watermark
```

- `created_at`, `window_start`, `window_end`는 UTC ISO 8601이며 `window_start`가
  `window_end`보다 늦으면 안 된다. timezone이 없는 값은 UTC로 해석한다.
- `data_format`은 `parquet`이어야 한다. pipeline은 `jsonl`도 읽지만 FastAPI Server는
  계약상 Parquet만 검증한다.
- `sha256`은 64자리 hex이며 `sha256:` 접두사를 붙여도 된다.

events Parquet은 다음 컬럼을 모두 포함해야 한다. ML pipeline의 `REQUIRED_EVENT_COLUMNS`와
같은 목록이다.

```text
event_id, request_id, question,
initial_label, initial_model, final_label, final_model,
user_satisfied, escalation_count, created_at,
router_model_version, pii_redacted
```

FastAPI Server는 pipeline을 제출하기 전에 다음을 검증한다.

1. 지원하는 `schemaVersion`인지 확인한다.
2. `manifestUri`와 `eventsUri`가 허용된 GCS 위치이며 객체가 존재하는지 확인한다.
3. manifest가 위 필드를 모두 갖추고 timestamp와 window 순서가 유효한지 확인한다.
4. request와 manifest의 batch ID, schema version, event count와 data URI가 일치하는지
   확인한다.
5. `data_format`이 FastAPI Server가 읽을 수 있는 형식인지 확인한다.
6. manifest의 SHA-256과 실제 events 객체 checksum이 일치하는지 확인한다.
7. events Parquet이 필수 컬럼을 모두 갖추었는지 확인한다.
8. Parquet 행 수가 `eventCount`와 일치하는지 확인한다.
9. 같은 `batchId`로 저장된 기존 요청이 있는지 확인한다.
10. 다른 batch의 재학습 pipeline이 실행 중인지 확인한다.

검증 원칙은 **pipeline이 거부할 batch는 FastAPI Server가 먼저 거부한다**이다. GPU 시간을
쓰기 전에, 그리고 Spring이 아직 요청을 손에 들고 있을 때 422로 돌려주는 것이 목적이다.

### 6-3. Accepted response

Status: `202 Accepted`

```json
{
  "status": "accepted",
  "batchId": "feedback-20260812-0001",
  "pipelineRunId": "retrain-feedback-20260812-0001-a1b2c3"
}
```

`accepted`는 Vertex AI PipelineJob 제출이 완료되어 `pipelineRunId`가 발급되었다는 뜻이다.
`pipelineRunId`는 Vertex resource name
(`projects/<num>/locations/us-west1/pipelineJobs/<job-id>`)이다.
학습, 평가, package, deploy gate 또는 serving 반영의 성공을 의미하지 않는다.

FastAPI Server는 ModernBERT 학습을 직접 실행하지 않는다. 제출된 pipeline이 다음 순서를
orchestration한다.

```text
feedback 검증
  -> 기존 training data 병합
  -> Vertex AI GPU ModernBERT 재학습
  -> 평가
  -> package
  -> deploy gate
  -> candidate 또는 rejected
```

### 6-4. 멱등성

`batchId`는 필수 멱등성 키다.

- 동일한 `batchId`와 동일한 요청 내용으로 재시도하면 새 pipeline을 제출하지 않고 기존
  `pipelineRunId`를 담아 `202 Accepted`를 반환한다.
- 동일한 `batchId`이지만 `schemaVersion`, URI 또는 `eventCount`가 다르면 기존 batch를
  덮어쓰지 않고 `409 Conflict`와 `BATCH_ID_CONFLICT`를 반환한다.

canonical request fingerprint는 batch의 정체성을 규정하는 다음 필드만 SHA-256으로 해싱한다.

```text
batchId, schemaVersion, manifestUri, eventsUri, eventCount
```

**`requestedAt`은 fingerprint에서 제외한다.** batch 정체성과 무관한 metadata이고, timeout
후 재시도하는 client는 timestamp를 새로 찍는 것이 자연스럽다. 이를 포함하면 멱등성이
가장 필요한 상황이 오히려 `BATCH_ID_CONFLICT`가 된다. 따라서 Spring은 재시도 시
`requestedAt`을 그대로 보내도 되고 새로 찍어도 되며, 두 경우 모두 같은 `pipelineRunId`를
담은 `202 Accepted`를 받는다.

동시 실행 가능한 재학습 pipeline은 하나다. 다른 `batchId`의 재학습이 이미 진행 중이면
`409 Conflict`와 `RETRAINING_IN_PROGRESS`를 반환한다. Spring은 GCS batch를 변경하거나
삭제하지 않고 보존한 뒤 나중에 같은 요청으로 재시도한다.

### 6-4-1. 중단된 제출의 lease 회수

FastAPI Server가 lease를 잡은 뒤 pipeline run ID를 기록하기 전에 죽으면(OOM, pod
eviction, 배포) 실행 중인 pipeline이 없는데도 lease가 남는다. 이 상태를 방치하면 해당
batch뿐 아니라 모든 batch의 재학습이 영구히 `RETRAINING_IN_PROGRESS`가 된다.

다음 `/escalate` 요청이 이런 lease를 자동으로 정리한다. run ID가 없고 마지막 갱신이
`LLM_ROUTER_RETRAINING_STALE_SUBMISSION_SECONDS`(기본 900초)보다 오래된 lease에 대해:

- Vertex에 해당 batch의 결정적 job ID가 존재하면 그 run을 채택하고 `pipelineRunId`로
  이어서 사용한다. 같은 batch를 두 번 학습하지 않는다.
- 존재하지 않으면 제출이 성립하지 않은 것으로 보고 lease를 해제한다. 이후 같은 요청을
  재시도하면 정상 제출된다.

임계 시간은 제출 중인 요청의 lease를 빼앗지 않기 위한 안전 여유다. 제출은 초 단위로
끝나므로 기본값을 줄일 이유는 없다.

현재 구현은 persistent SQLite lease를 사용하므로 FastAPI 단일 replica를 전제로 한다.
여러 replica로 확장하기 전에는 shared transactional database 또는 distributed lease로
교체해 동시 실행 제한이 전체 replica에 걸쳐 유지되게 해야 한다.

### 6-5. 오류

| Status | Code | 조건 |
| --- | --- | --- |
| `401 Unauthorized` | `UNAUTHORIZED` | 내부 인증 실패 |
| `403 Forbidden` | `FORBIDDEN` | pipeline 제출 또는 GCS 접근 권한 없음 |
| `409 Conflict` | `BATCH_ID_CONFLICT` | 같은 `batchId`에 다른 요청 내용이 존재 |
| `409 Conflict` | `RETRAINING_IN_PROGRESS` | 다른 batch의 pipeline이 이미 실행 중 |
| `422 Unprocessable Entity` | `UNSUPPORTED_SCHEMA_VERSION` | 지원하지 않는 `schemaVersion` |
| `422 Unprocessable Entity` | `GCS_OBJECT_NOT_FOUND` | manifest 또는 events 객체가 없음 |
| `422 Unprocessable Entity` | `MANIFEST_MISMATCH` | manifest 필수 필드 누락·형식 오류, window 순서 오류, 또는 request와 manifest metadata 불일치 |
| `422 Unprocessable Entity` | `UNSUPPORTED_DATA_FORMAT` | `data_format`이 `parquet`이 아님 |
| `422 Unprocessable Entity` | `CHECKSUM_MISMATCH` | events 객체 checksum 불일치 |
| `422 Unprocessable Entity` | `EVENT_SCHEMA_MISMATCH` | events Parquet에 필수 컬럼 누락 |
| `422 Unprocessable Entity` | `EVENT_COUNT_MISMATCH` | request, manifest 또는 Parquet 행 수 불일치 |
| `422 Unprocessable Entity` | `REQUEST_VALIDATION_FAILED` | request body가 schema를 위반 (`eventCount != 1000` 포함) |
| `503 Service Unavailable` | `GCS_BACKEND_UNAVAILABLE` | GCS API에 연결할 수 없음 |
| `502 Bad Gateway` | `PIPELINE_SUBMISSION_FAILED` | Vertex AI가 제출을 거부하거나 유효한 run ID를 반환하지 못함 |
| `503 Service Unavailable` | `PIPELINE_BACKEND_UNAVAILABLE` | Vertex AI backend에 연결할 수 없음 |

오류 예시:

```json
{
  "error": "다른 재학습 작업이 진행 중입니다.",
  "code": "RETRAINING_IN_PROGRESS",
  "requestId": "req-retrain-123"
}
```

### 6-6. 추적과 감사

Spring과 FastAPI는 다음 값을 양쪽 로그에서 연결할 수 있어야 한다.

- `requestId`
- `batchId`
- `pipelineRunId`
- pipeline 제출 시각과 제출 주체
- manifest/events URI와 checksum
- pipeline 최종 상태 및 candidate/rejected model version

GCS batch는 pipeline 완료 전후에 덮어쓰지 않는다. 보존·삭제 기간은 별도 data retention
정책으로 정한다.

## 7. 상태 추적 API 후속 계약

`POST /escalate`가 비동기 제출 API이므로 Spring이 `pipelineRunId`로 실행 상태를 조회할
방법이 필요하다. 상태 조회 endpoint의 path, polling 간격, 상태 enum과 terminal state는
후속 계약에서 확정한다. 확정 전에도 `pipelineRunId`는 `/escalate` 성공 응답과 양쪽 로그에
반드시 남긴다.

후보 endpoint:

```text
GET /pipelines/{pipelineRunId}
```

후보 상태:

```text
queued | validating | training | evaluating | packaging
candidate | rejected | failed | cancelled
```

## 8. Contract test 최소 범위

- `/router/embed`가 정확히 1,024개의 유한 실수를 반환하고 vector norm이 허용 오차 안에서
  `1.0`인지 검사한다.
- EXP-003 고정 질문 fixture의 embedding과 serving 결과가 허용 오차 안에서 같은지
  검사한다.
- `/router/classify`가 항상 유효한 tier, confidence와 `modelVersion`을 반환하는지 검사한다.
- checkpoint의 `easy=0`, `medium=1`, `hard=2` 순서가 API tier와 일치하는지 검사한다.
- model loading 중 `/health`가 `503`, 모두 loaded일 때 `200`인지 검사한다.
- `/escalate`의 정상 제출, 같은 요청 재시도, batch ID 충돌, 동시 실행 충돌을 검사한다.
- `requestedAt`만 다른 재시도가 같은 `pipelineRunId`로 `202`를 받는지 검사한다.
- manifest metadata, checksum, GCS 객체 존재, 필수 컬럼과 행 수 오류가 각각 구분되는
  `422` code로 반환되는지 검사한다.
- manifest 필수 필드 목록과 events 필수 컬럼 목록이 ML pipeline의
  `REQUIRED_MANIFEST_FIELDS`, `REQUIRED_EVENT_COLUMNS`와 일치하는지 검사한다.
- 중단된 제출이 남긴 lease가 Vertex 조회로 채택 또는 해제되는지 검사한다.
- Spring과 FastAPI fixture를 같은 저장본에서 검증해 필드명과 enum drift를 차단한다.

## 9. 확정이 필요한 항목

- 운영 내부 인증을 bearer token보다 강한 workload identity/mTLS로 교체할지 여부
- `/health` probe의 network policy
- embedding과 classify의 최대 입력 길이, timeout, retry와 concurrency
- EXP-003가 동일한 고정 revision과 preprocessing 계약을 사용하는지 여부
- health 응답에 실제 model version을 포함할지 여부
- 멱등성 record 보존 기간 (fingerprint 구성은 §6-4에서 확정)
- pipeline 상태 조회 endpoint와 상태 enum
- `candidate_model_version`에 batch 식별자를 포함해 같은 날 두 번째 재학습이 첫 candidate
  package를 덮어쓰지 않게 할 방법
- GCS 버킷 allowlist 위반과 서버측 IAM 거부를 다른 code로 분리할지 여부
- GCS URI allowlist, IAM principal과 data retention 기간
