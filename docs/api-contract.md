# FastAPI 내부 API 계약

상태: Spring 연동 기준 문서

기준일: 2026-09-16

이 문서는 Spring과 `LLM_Router-serving` 사이의 HTTP 계약을 정의한다. 재학습 입력은
`feedback-inline-v1` JSON 하나로 통일한다. Spring은 Parquet, manifest, checksum 또는 GCS
URI를 만들지 않는다.

RouterBench+xRouteBench 통합 데이터, 모델 비교, threshold·Vertex/GCS·serving 변경의
실행 기준은 sibling 저장소의
[`integrated-routing-data-and-model-comparison-plan.md`](../../LLM_Router/docs/ml/integrated-routing-data-and-model-comparison-plan.md)다.
여러 backbone은 모델별 Colab notebook의 offline 비교에만 사용하고 KFP/Vertex에 올리지
않는다. ModernBERT 초기학습도 Colab에서 수행하며 Vertex GPU 학습은 feedback 재학습부터
사용한다. continual retraining·자동 승격·FastAPI 적재는 ModernBERT만 대상으로 한다.
비교 실험은 이 HTTP v1을 변경하지 않으며, 내부
ModernBERT package metadata 변경도 Spring payload에 전파하지 않는다.

### 이 책임 경계를 선택한 이유

재학습은 FastAPI와 ML pipeline의 내부 구현이다. Spring에 GCS upload까지 맡기면 Spring이
HTTP용 event DTO와 ML용 Parquet/manifest를 동시에 유지해야 하고, GCS SDK·bucket
경로·checksum·artifact schema·IAM 변경에도 함께 대응해야 한다. 이는 같은 데이터를 두
서버에서 중복 가공하게 만들고 Spring을 학습 인프라에 결합시킨다.

따라서 Spring의 학습 관련 책임은 “재학습할 event를 선택하고 PII를 제거해 HTTP로
전송한다”까지로 제한한다. FastAPI가 ML 계약을 소유하면서 검증·정규화·Parquet·manifest·GCS
게시·Vertex 제출을 한 경계에서 처리한다. 결과적으로:

- Spring에는 재학습용 GCS SDK, URI 조립 코드와 GCS IAM이 필요 없다.
- ML artifact schema가 바뀌어도 FastAPI/ML만 함께 변경하면 된다.
- checksum, replay, version lineage와 immutable 저장을 한 곳에서 일관되게 통제한다.
- Spring은 public service와 business trigger에 집중하고 학습 인프라 장애를 직접 다루지 않는다.

## 1. 책임과 endpoint

Spring의 학습 관련 책임은 재학습 event 선택, PII 제거와 HTTP 전송까지다. FastAPI는
HTTP event 검증부터 ML artifact 생성, GCS 저장, Vertex 제출, 상태 변환과 승인 model
반영까지 맡는다.

| Endpoint | 용도 | 호출 주체 |
| --- | --- | --- |
| `POST /router/embed` | 질문 embedding | Spring |
| `POST /router/classify` | 난이도 분류 | Spring |
| `GET /health` | inference readiness | load balancer·운영 도구 |
| `POST /escalate` | inline feedback 접수와 재학습 제출 | Spring |
| `GET /retraining-runs/{batchId}` | 재학습 단계·최종 결과·model 활성화 조회 | Spring |
| `GET /router/model/current` | 현재 승인/serving model 조회 | Spring·운영 도구 |
| `GET/PUT /developer/retraining/version-policy` | version trigger mode 조회·변경 | 개발자 도구만 |

## 2. 공통 HTTP 규칙

- 기본 URL은 환경별 설정값으로 관리한다. VM IP, GCP region 또는 hostname을 코드에
  하드코딩하지 않는다.
- JSON 요청은 `Content-Type: application/json; charset=utf-8`로 보낸다.
- 인증이 설정된 환경에서는 `Authorization: Bearer <internal-token>`을 보낸다.
- 모든 요청에 Spring이 생성한 `X-Request-ID`를 보내는 것을 권장한다. FastAPI는 같은 값을
  응답 header에 반환하며, 없으면 새 UUID를 생성한다.
- 알 수 없는 JSON 필드는 거부한다.
- 모든 시각은 UTC ISO 8601 문자열로 보내며 `Z` 표기를 권장한다.

공통 오류 body는 다음과 같다.

```json
{
  "error": "request body does not match the API contract",
  "code": "REQUEST_VALIDATION_FAILED",
  "requestId": "0f1d3f52-f560-4b0b-93d0-acde58745ae1"
}
```

Spring은 사람이 읽는 `error`가 아니라 안정된 `code`로 분기하고, 문의·로그 연결에는
`requestId`를 사용한다.

## 3. 온라인 추론 API

### 3.1 `POST /router/embed`

Request:

```json
{
  "message": "How does photosynthesis work?"
}
```

Success: `200 OK`

```json
{
  "embedding": [0.0123, -0.0456],
  "dim": 1024,
  "modelVersion": "bge-large-en-v1.5-d4aa690"
}
```

`embedding`은 FastAPI에서 L2 정규화된 1,024차원 vector다. Spring은
`modelVersion`이 다른 embedding을 같은 cache 공간에서 비교하지 않는다.

### 3.2 `POST /router/classify`

Request는 `/router/embed`와 같다.

Success: `200 OK`

```json
{
  "tier": "medium",
  "model": "sonnet",
  "confidence": 0.87,
  "modelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d"
}
```

- `tier`: `easy | medium | hard`
- `model`: `haiku | sonnet | opus`
- `confidence`: `0.0` 이상 `1.0` 이하
- `modelVersion`: 분류에 실제 사용된 immutable router model version

Spring은 `model`을 참고값으로만 사용하고 실제 provider model ID 선택은 자체 versioned
routing policy로 관리한다. `modelVersion`은 이후 escalation event의
`routerModelVersion`으로 보존한다.

`confidence`는 active ModernBERT가 내는 세 label 점수를 validation에서 정한 방식으로
확률화·보정한 0~1 값이어야 한다. offline 비교 backbone의 raw logit이나 생성 확률을
이 API 값으로 보내지 않는다.

### 3.3 `GET /health`

Ready: `200 OK`

```json
{
  "status": "ok",
  "embedModel": "loaded",
  "routerModel": "loaded"
}
```

두 model 중 하나라도 준비되지 않으면 같은 schema로 `503 Service Unavailable`을 반환한다.

```json
{
  "status": "not_ready",
  "embedModel": "loaded",
  "routerModel": "loading"
}
```

`/health`는 load balancer의 readiness 용도다. 현재 model version을 확인하는 API가 아니다.

## 4. `POST /escalate`

### 4.1 책임과 처리 순서

Spring은 재학습 대상으로 결정한 event를 1건 이상 한 요청에 담는다. HTTP 호출 시점과
event 수는 Spring의 정책이다. FastAPI에는 1,000건 조건이나 애플리케이션 수준의 최대 건수
조건이 없다.

```text
Spring DB
  -> PII 제거와 전송 대상 선택
  -> POST /escalate: feedback-inline-v1 JSON
  -> FastAPI schema·의미·중복 검증
  -> eventId 순 정규화
  -> snake_case Parquet 변환
  -> SHA-256 + feedback-export-v0.1 manifest 생성
  -> 현재 3-call cycle + compact replay 구성
  -> FastAPI service account로 GCS에 immutable 저장
  -> Vertex AI PipelineJob 제출
  <- 202 Accepted + batchId + pipelineRunId
```

GCS는 Vertex 학습에 필요한 내구성 있는 내부 artifact 저장소로 계속 사용한다. 다만 GCS
SDK, bucket 이름, object URI, checksum, manifest와 IAM은 모두 FastAPI/ML 책임이다.

### 4.2 Request header

```http
POST /escalate
Content-Type: application/json; charset=utf-8
Authorization: Bearer <internal-token>
X-Request-ID: req-retrain-20260912-0001
```

### 4.3 Request body

```json
{
  "batchId": "feedback-20260912-0001",
  "schemaVersion": "feedback-inline-v1",
  "requestedAt": "2026-09-12T06:30:00Z",
  "events": [
    {
      "eventId": "esc-284391",
      "requestId": "req-019c1234",
      "question": "Explain why the sky appears blue.",
      "initialLabel": "easy",
      "initialModel": "haiku",
      "finalLabel": "medium",
      "finalModel": "sonnet",
      "userSatisfied": true,
      "escalationCount": 1,
      "createdAt": "2026-09-12T06:27:41Z",
      "routerModelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d",
      "piiRedacted": true
    }
  ]
}
```

Top-level 필드:

| 필드 | JSON 타입 | 필수 | 규칙 |
| --- | --- | :---: | --- |
| `batchId` | string | Y | 1~200자, `[A-Za-z0-9._-]+`; immutable 멱등성 키 |
| `schemaVersion` | string | Y | 정확히 `feedback-inline-v1` |
| `requestedAt` | string | Y | UTC ISO 8601 요청 생성 시각 |
| `events` | array | Y | 1개 이상; 요청 안에서 `eventId` 고유 |

`events[]` 필드:

| 필드 | JSON 타입 | 필수 | 규칙 |
| --- | --- | :---: | --- |
| `eventId` | string | Y | 비공백, 최대 200자, 요청 배열 안에서 고유 |
| `requestId` | string | Y | 원 사용자 요청 ID, 비공백, 최대 200자 |
| `question` | string | Y | 실제 PII 제거가 끝난 원 질문, 비공백, 최대 100,000자 |
| `initialLabel` | string | Y | 최초 router tier: `easy | medium | hard` |
| `initialModel` | string | Y | 최초 응답 model 식별자, 비공백, 최대 300자 |
| `finalLabel` | string | Y | 만족 응답 tier; `initialLabel`보다 반드시 높음 |
| `finalModel` | string | Y | 만족 응답 model 식별자, 비공백, 최대 300자 |
| `userSatisfied` | boolean | Y | JSON boolean `true`만 허용 |
| `escalationCount` | integer | Y | 해당 요청 안에서 상향 호출한 횟수, 1 이상 |
| `createdAt` | string | Y | event 생성 시각, UTC ISO 8601 |
| `routerModelVersion` | string | Y | 최초 분류에 사용된 router model version |
| `piiRedacted` | boolean | Y | 실제 PII 제거를 완료한 뒤 `true` |

허용되는 label 이동은 `easy -> medium`, `easy -> hard`, `medium -> hard`다. 같은 label 또는
하향 이동은 재학습 불만 event가 아니므로 거부한다.

다음 필드는 **존재하지 않으며 보내면 422**다.

```text
eventCount, manifestUri, eventsUri
```

FastAPI는 `events.length`를 직접 계산한다. `eventCount`는 중복 정보이므로 완전히 제거했다.

`piiRedacted=true`는 Spring의 완료 선언이다. FastAPI는 이 boolean을 강제하지만 질문
본문에서 주민번호·이메일 등을 자동 탐지하거나 가리는 기능까지 제공하지 않는다.

### 4.4 FastAPI 내부 artifact

Spring이 보낸 camelCase event는 다음 12개 Parquet column으로 변환된다.

```text
event_id, request_id, question,
initial_label, initial_model, final_label, final_model,
user_satisfied, escalation_count, created_at,
router_model_version, pii_redacted
```

- `created_at`: `timestamp[ms, UTC]`
- `escalation_count`: int32
- `user_satisfied`, `pii_redacted`: boolean
- 나머지: UTF-8 string

FastAPI는 event를 `eventId`로 정렬해 같은 내용의 재시도에서 동일한 Parquet bytes와 SHA-256을
만든다. 생성된 artifact schema는 HTTP schema와 다른 `feedback-export-v0.1`이다.

내부 저장 형태는 다음과 같다. 정확한 bucket/prefix는 운영 설정이며 Spring 계약이 아니다.

```text
<derived-root>/<candidate-version>/feedback/events.parquet
<derived-root>/<candidate-version>/feedback/manifest.json
<derived-root>/<candidate-version>/replay/events.parquet
<derived-root>/<candidate-version>/replay/manifest.json
```

FastAPI는 immutable create 조건으로 저장하며, 같은 URI에 다른 bytes가 이미 있으면
`409 GCS_OBJECT_CONFLICT`로 중단한다.

### 4.5 Accepted response

`202 Accepted`

```json
{
  "schemaVersion": "escalation-acceptance-v1",
  "status": "accepted",
  "batchId": "feedback-20260912-0001",
  "pipelineRunId": "projects/123456789/locations/us-west1/pipelineJobs/retrain-feedback-20260912-0001-a1b2c3",
  "logicalModelVersion": "2.1",
  "candidateModelVersion": "router-modernbert-feedback-v2.1-20260912-a1b2c3d4-45dcc6d",
  "trainingMode": "minor_fine_tune",
  "versionPolicyMode": "successfulHttpRequest",
  "statusUrl": "/retraining-runs/feedback-20260912-0001"
}
```

`202`는 GCS artifact 생성과 Vertex PipelineJob 제출이 성립했다는 뜻이다. 학습, 평가,
자동 승격 또는 FastAPI hot-swap 성공을 뜻하지 않는다. `pipelineRunId`는 식별자이므로
Spring이 그 문자열로 HTTP 요청을 보내지 않는다. Spring은 `statusUrl`을 같은 FastAPI
base URL에 resolve해 상태를 조회한다.

### 4.6 멱등성과 재시도

`batchId`는 단순 표시 이름이 아니라 멱등성 키다. fingerprint는 다음 값으로 결정한다.

```text
batchId + schemaVersion + events 전체 내용(eventId 순 정렬)
```

`requestedAt`과 events 배열 순서는 batch 정체성에서 제외한다.

- 같은 `batchId` + 같은 events: 새 pipeline 없이 기존 `pipelineRunId`로 `202`
- 같은 `batchId` + 다른 events: `409 BATCH_ID_CONFLICT`
- 다른 batch가 학습 중: `409 RETRAINING_IN_PROGRESS`
- client timeout 또는 `502/503`: 새 ID를 만들지 말고 같은 batch와 events로 재시도

Spring은 전송한 batch payload를 terminal 상태가 될 때까지 다시 만들 수 있어야 한다. 한
HTTP 호출이 version trigger 1회이므로 network 오류를 이유로 batch를 임의 분할해서 보내면
안 된다.

현재 FastAPI는 한 번에 하나의 retraining pipeline만 허용하며 SQLite에 멱등성과 lease를
보존한다. 이는 단일 FastAPI replica 전제다. 다중 replica 전에 shared transactional
database 또는 distributed lease로 교체한다.

### 4.7 즉시 오류

| HTTP | `code` | Spring 처리 |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | token 설정 확인; 자동 무한 재시도 금지 |
| 403 | `FORBIDDEN` | FastAPI의 Vertex/GCS IAM 장애로 운영 알림 |
| 409 | `BATCH_ID_CONFLICT` | 재시도 금지; 같은 ID에 다른 payload가 저장됐는지 조사 |
| 409 | `RETRAINING_IN_PROGRESS` | 같은 payload를 보존하고 backoff 후 재시도 |
| 409 | `GCS_OBJECT_CONFLICT` | 운영 조사; immutable path에 다른 내용 존재 |
| 422 | `UNSUPPORTED_SCHEMA_VERSION` | client/server 배포 version 확인 |
| 422 | `REQUEST_VALIDATION_FAILED` | DTO, 필드, UTC, label, boolean, 중복 ID 수정 |
| 502 | `PIPELINE_SUBMISSION_FAILED` | 같은 batch/payload로 제한 재시도 후 알림 |
| 503 | `GCS_BACKEND_UNAVAILABLE` | 같은 batch/payload로 backoff 재시도 |
| 503 | `PIPELINE_BACKEND_UNAVAILABLE` | 같은 batch/payload로 backoff 재시도 |

## 5. 재학습 version 정책

기본 연구 mode는 승격까지 성공한 HTTP 요청 수를 기준으로 한다.

```text
n.0 anchor
  -> 첫 성공 호출: n.1, minor_fine_tune
  -> 둘째 성공 호출: n.2, minor_fine_tune
  -> 셋째 성공 호출: (n+1).0, major_consolidation
```

minor 두 번과 major 한 번은 같은 `.0` anchor에서 시작하며, major는 그 cycle의 세 호출
feedback을 합친다. 이전 cycle 전체를 무한 누적하지 않고 label별 최대 32행 compact replay를
유지한다. major consolidation 5회마다 `pristine_rebase`를 실행해 순정 ModernBERT + 기본
RouterBench + compact replay로 재기준화한다.

호출이 접수만 되고 학습이 실패한 경우 성공 횟수에 포함하지 않는다. 같은 batch의 멱등
재시도도 횟수를 늘리지 않는다.

## 6. `GET /retraining-runs/{batchId}`

Spring은 이 endpoint 하나로 학습 결과와 serving 반영 여부를 조회한다. Vertex 또는 GCS를
직접 조회하지 않는다.

```http
GET /retraining-runs/feedback-20260912-0001
Authorization: Bearer <internal-token>
X-Request-ID: req-retrain-status-1
```

핵심 response schema:

```json
{
  "schemaVersion": "retraining-status-v1",
  "batchId": "feedback-20260912-0001",
  "pipelineRunId": "projects/123456789/locations/us-west1/pipelineJobs/retrain-feedback-20260912-0001-a1b2c3",
  "status": "running",
  "terminal": false,
  "stage": "training",
  "version": {
    "logicalModelVersion": "2.1",
    "candidateModelVersion": "router-modernbert-feedback-v2.1-20260912-a1b2c3d4-45dcc6d",
    "trainingMode": "minor_fine_tune",
    "initializationMode": "approved_anchor",
    "versionPolicyMode": "successfulHttpRequest"
  },
  "stageProgress": {
    "current": 612,
    "total": 7038,
    "unit": "optimizerStep",
    "percent": 8.7,
    "epoch": 1,
    "totalEpochs": 3,
    "heartbeatAt": "2026-09-12T07:12:30Z"
  },
  "result": null,
  "activation": {
    "status": "pending",
    "desiredModelVersion": null,
    "servingModelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d",
    "checkedAt": "2026-09-12T07:12:32Z"
  },
  "submittedAt": "2026-09-12T06:30:02Z",
  "startedAt": "2026-09-12T06:30:45Z",
  "endedAt": null,
  "lastUpdatedAt": "2026-09-12T07:12:32Z",
  "stale": false,
  "warnings": []
}
```

`status` enum:

```text
queued | running | cancelling | paused |
succeeded | rejected | failed | cancelled | unknown
```

`terminal=true`인 값은 `succeeded`, `rejected`, `failed`, `cancelled`다. terminal 실패도 조회
자체는 정상 처리이므로 HTTP 200으로 반환한다.

`stage` enum:

```text
initializing | feedbackPreparation | feedbackMerge | training |
evaluation | packaging | gate | publishing | complete
```

terminal `result`의 핵심 필드:

```json
{
  "outcome": "promoted",
  "candidateModelVersion": "router-modernbert-feedback-v2.1-20260912-a1b2c3d4-45dcc6d",
  "logicalModelVersion": "2.1",
  "gatePassed": true,
  "failedGateChecks": [],
  "failureCode": null,
  "failureStage": null,
  "retryable": false,
  "message": null,
  "approvedAt": "2026-09-12T09:42:10Z"
}
```

`outcome`: `promoted | rejected | failed | cancelled`

안정된 `failureCode`:

```text
FEEDBACK_VALIDATION_FAILED | NO_UNIQUE_FEEDBACK |
TRAINING_FAILED | EVALUATION_FAILED | DEPLOY_GATE_REJECTED |
PACKAGE_FAILED | PUBLISH_FAILED | PIPELINE_CANCELLED |
UNKNOWN_PIPELINE_FAILURE
```

`activation.status`:

| 값 | 의미 |
| --- | --- |
| `pending` | 승격 전이거나 watcher가 새 package를 적재 중 |
| `active` | 이 FastAPI process가 candidate version으로 실제 추론 중 |
| `loadFailed` | 새 package 적재 실패; 직전 known-good model을 계속 사용 |
| `superseded` | 이 run 이후 더 최신 model이 이미 serving 중 |
| `notApplicable` | 거부·실패·취소로 반영 대상이 아님 |

조회 시 Vertex/GCS가 일시적으로 실패해도 저장 snapshot이 있으면 HTTP 200,
`stale=true`, `warnings=["PIPELINE_STATUS_STALE"]`를 반환한다. snapshot도 없으면
`503 PIPELINE_STATUS_UNAVAILABLE`, batch가 없으면 `404 RETRAINING_RUN_NOT_FOUND`다.

## 7. `GET /router/model/current`

승인 포인터가 원하는 model과 이 process가 실제 추론에 쓰는 model을 함께 반환한다.

```json
{
  "schemaVersion": "router-model-status-v1",
  "status": "inSync",
  "source": "currentPointer",
  "desired": {
    "modelVersion": "router-modernbert-feedback-v2.1-20260912-a1b2c3d4-45dcc6d",
    "logicalModelVersion": "2.1",
    "trainingMode": "minor_fine_tune",
    "manifestSha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "approvedAt": "2026-09-12T09:42:10Z"
  },
  "serving": {
    "modelVersion": "router-modernbert-feedback-v2.1-20260912-a1b2c3d4-45dcc6d",
    "logicalModelVersion": "2.1",
    "loadedAt": "2026-09-12T09:43:01Z"
  },
  "previousKnownGood": {
    "modelVersion": "router-modernbert-feedback-v2.0-20260901-b1c2d3e4-45dcc6d"
  },
  "lastPointerCheckAt": "2026-09-12T09:43:01Z",
  "lastSuccessfulSwapAt": "2026-09-12T09:43:01Z",
  "lastLoadError": null
}
```

`status`: `inSync | loading | loadFailed | static | unavailable`

`succeeded` pipeline과 `activation=active`는 다르다. pipeline 성공은 package 게시와
`CURRENT.json` 변경까지이고, `active`는 FastAPI가 package checksum/label 계약을 검증하고
메모리 model swap까지 끝냈다는 뜻이다.

## 8. 개발자 version policy API

이 API는 Spring 연동 대상이 아니다. 별도 developer token, feature flag와 network 제한을
사용한다.

```http
GET /developer/retraining/version-policy
PUT /developer/retraining/version-policy
```

허용 mode:

- `successfulHttpRequest`: 승격까지 성공한 고유 HTTP batch 3회 기준. 현재 연구 기본값.
- `validatedUniqueBatch`: canonical 검증을 통과하고 이전과 다른 digest가 생긴 batch 3회 기준.

PUT request 초안:

```json
{
  "mode": "validatedUniqueBatch",
  "reason": "production trigger validation completed"
}
```

mode 변경은 이미 제출된 run을 재해석하거나 version counter를 초기화하지 않는다. 변경 이력은
행위자, 시각, 이전/신규 mode, 사유와 함께 감사 로그에 남긴다.

## 9. Spring polling 규칙

Spring의 기본 polling 규칙은 다음과 같다.

1. `202`를 받으면 10초 뒤 첫 조회를 한다.
2. `terminal=false`면 15초, 30초, 이후 최대 60초 간격으로 backoff한다.
3. `stale=true` 또는 `status=unknown`은 완료로 확정하지 않는다.
4. `terminal=true`면 polling을 끝낸다.
5. `status=succeeded`여도 `activation.status=active`가 아니면 model 반영 대기 상태로 둔다.
6. `/retraining-runs/{batchId}`의 `statusUrl`만 HTTP로 호출한다. `pipelineRunId`와 GCS URI는
   호출 URL이 아니다.

## 10. Contract test 조건

최소한 다음을 Spring과 FastAPI 양쪽 contract test로 고정한다.

- 1개 event가 포함된 `feedback-inline-v1` 요청이 202가 된다.
- 빈 `events`와 `eventCount`, `manifestUri`, `eventsUri`가 포함된 요청은 422가 된다.
- 같은 event를 다른 배열 순서와 다른 `requestedAt`으로 재시도해도 같은 run ID가 나온다.
- 같은 `batchId`에 다른 event 내용은 409가 된다.
- label 하향, `userSatisfied=false`, `piiRedacted=false`, 중복 event ID, non-UTC 시각은 422다.
- Parquet 12개 column/type과 manifest SHA-256이 ML validator 계약과 일치한다.
- 202 이후 running, terminal failure, promoted, activation active 상태를 구분한다.
- 새 model load 실패 시 현재 known-good model이 계속 `/router/classify`를 처리한다.

복사 가능한 request fixture는
[`examples/escalation-feedback-inline-v1/escalate-request.example.json`](examples/escalation-feedback-inline-v1/escalate-request.example.json)에 있다.
조회 response fixture는 [`examples/retraining-api-v1/`](examples/retraining-api-v1/)에
있다.
