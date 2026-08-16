# 백엔드 연동 안내

대상: Spring Boot 백엔드 팀

기준일: 2026-08-16

이 문서는 Spring 서버가 FastAPI Router 서버를 호출하는 데 필요한 것만 모았다.
정확한 schema와 전체 오류 목록은 [`api-contract.md`](api-contract.md)가 기준이며, 이
문서와 충돌하면 `api-contract.md`를 따른다.

## 1. 서버 주소

| 항목 | 값 |
| --- | --- |
| 내부 IP | `10.138.0.2` |
| 포트 | `8000` |
| Base URL | `http://10.138.0.2:8000` |
| 리전 / 존 | `us-west1` / `us-west1-a` |
| VM 이름 | `llm-router-serving` |
| 네트워크 태그 | `llm-router-serving` |
| 외부 IP | **없음** |

> **상태: 아직 서비스 중이 아니다.** VM은 떠 있지만 컨테이너가 기동하지 못한 상태다.
> 서빙 서비스 계정에 Artifact Registry 읽기 권한이 없어 이미지를 받지 못한다. 권한이
> 부여되면 위 주소로 바로 응답한다. 연동 테스트는 그 이후에 시작한다.

외부 IP가 없으므로 인터넷에서는 접근할 수 없다. 같은 VPC(`default`) 안에서만 호출할 수
있으며, Spring이 있는 `asia-northeast3` 서브넷(`10.178.0.0/20`)에서 us-west1
서브넷(`10.138.0.0/20`)으로 가는 트래픽은 기존 `default-allow-internal` 규칙으로 이미
허용되어 있다. 별도 방화벽 작업은 필요 없다.

### 리전이 다른 점

Spring은 서울, Router는 오리건이다. 리전 간 왕복 지연이 **요청당 약 120~140ms** 붙는다.
`/router/embed`는 모든 요청에서 호출되고 `/router/classify`는 cache miss에서 호출되므로,
cache miss 한 건에 두 번의 왕복이 발생한다. 타임아웃을 잡을 때 이 값을 고려한다.

## 2. 인증

`Authorization: Bearer <token>` 헤더가 필요하다. 없거나 틀리면 `401 UNAUTHORIZED`다.

토큰은 Secret Manager에 있다.

```bash
gcloud secrets versions access latest \
  --secret=llm-router-internal-api-token \
  --project=project-763753da-fa8f-4afd-b31
```

Spring 서비스 계정에 `roles/secretmanager.secretAccessor`를 부여받아야 읽을 수 있다.
**토큰을 소스 코드나 저장소에 넣지 않는다.**

`/health`는 인증 없이 호출할 수 있다. 헬스체크용이다.

## 3. 요청 ID

모든 요청에 `X-Request-ID` 헤더를 넣는다. 서버가 로그와 오류 응답에 그대로 보존하고
응답 헤더로 되돌려준다. 없으면 서버가 UUID를 생성한다. 장애 추적할 때 양쪽 로그를
연결하는 유일한 키다.

## 4. 엔드포인트

### `GET /health`

배포 readiness와 장애 진단용. 인증 불필요.

```json
{"status": "ok", "embedModel": "loaded", "routerModel": "loaded"}
```

두 모델 중 하나라도 `loaded`가 아니면 `503`과 `status: "not_ready"`다.
**기동 직후 수십 초 동안은 503이다.** 모델을 메모리에 올리는 시간이며 정상이다.

### `POST /router/embed`

cache 조회 전에 **항상** 호출한다.

```json
// 요청
{"message": "질문 원문"}

// 응답 200
{"embedding": [0.0123, -0.0456, ...], "dim": 1024, "modelVersion": "bge-large-en-v1.5"}
```

- `embedding`은 L2 정규화된 1,024차원 벡터다. pgvector `vector(1024)`에 그대로 넣는다.
- 정규화되어 있으므로 cosine distance에서 `similarity = 1 - distance`가 성립한다.
- `modelVersion`을 `request_log`에 남긴다. 이 값이 바뀌면 기존 cache 벡터와 새 벡터를
  같은 기준으로 비교하면 안 된다.

**pgvector 조회와 cache hit 판정은 Spring이 소유한다.** 서버는 벡터만 만든다.

### `POST /router/classify`

cache miss일 때만 호출한다.

```json
// 요청
{"message": "질문 원문"}

// 응답 200
{
  "tier": "easy",
  "model": "haiku",
  "confidence": 0.87,
  "modelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d"
}
```

- `tier`는 `easy` / `medium` / `hard` 중 하나다.
- `model`은 **참고값**이다. 실제 tier→모델 매핑의 소유자는 Spring의 `application.yml`이다.
  둘이 다르면 Spring 설정이 우선한다. 매핑만 바꿀 때는 모델 재학습도 FastAPI 재배포도
  필요 없다.
- `confidence`가 낮아도 `tier`는 항상 온다. **confidence로 분기하지 않는다.** 기록과
  사후 분석에만 쓴다.
- `modelVersion`을 `request_log`에 남긴다. 재학습 전후 판정을 분리해 분석하는 데 필수다.

### `POST /escalate`

재학습 트리거. 자세한 건 6장.

## 5. 오류 처리

공통 오류 body:

```json
{"error": "사람이 읽는 설명", "code": "ROUTER_NOT_READY", "requestId": "req-123"}
```

`error` 문자열로 분기하지 않는다. `code`로 분기한다.

| Status | Code | 재시도 |
| --- | --- | --- |
| 400 | `INVALID_MESSAGE` | 아니오 — 요청이 잘못됨 |
| 401 | `UNAUTHORIZED` | 아니오 — 토큰 확인 |
| 403 | `FORBIDDEN` | 아니오 |
| 503 | `EMBED_MODEL_NOT_READY` / `ROUTER_MODEL_NOT_READY` | **예** — 기동 중이거나 적재 실패 |
| 500 | `EMBEDDING_INFERENCE_FAILED` / `ROUTER_INFERENCE_FAILED` | 한 번 정도 |

### FastAPI가 죽었을 때

Router가 응답하지 않으면 Spring이 어떻게 할지는 Spring 쪽 정책이다. 권장은 **상위 tier로
fallback**(예: `medium`으로 고정)이다. 라우팅 실패로 사용자 요청 전체를 실패시키는 것보다
조금 비싼 모델로 답하는 편이 낫다. 이때 fallback이 발생했다는 사실을 로그에 남긴다.

## 6. 재학습 트리거 — `POST /escalate`

유효한 escalation feedback이 **1,000건** 쌓이면 호출한다.

### 순서

1. Spring이 GCS에 immutable batch를 먼저 게시한다.
2. Spring이 `/escalate`에 **위치와 metadata만** 보낸다. 이벤트 본문은 보내지 않는다.
3. FastAPI가 검증하고 Vertex AI 재학습 pipeline을 제출한 뒤 `202`와 `pipelineRunId`를
   반환한다.

### 게시 위치

```text
gs://llm-router-mlserver/datasets/feedback/<batch-id>/manifest.json
gs://llm-router-mlserver/datasets/feedback/<batch-id>/events.parquet
```

Spring 서비스 계정에 `datasets/feedback/` 접두사 조건부 `roles/storage.objectCreator`가
필요하다.

### 요청

```json
{
  "batchId": "feedback-20260812-0001",
  "schemaVersion": "feedback-export-v0.1",
  "manifestUri": "gs://llm-router-mlserver/datasets/feedback/feedback-20260812-0001/manifest.json",
  "eventsUri": "gs://llm-router-mlserver/datasets/feedback/feedback-20260812-0001/events.parquet",
  "eventCount": 1000,
  "requestedAt": "2026-08-12T15:30:00Z"
}
```

### manifest.json 필수 필드 11개

```text
schema_version, batch_id, created_at, window_start, window_end,
event_count, data_format, data_uri, sha256, producer_version, watermark
```

- 타임스탬프는 UTC ISO 8601. `window_start` ≤ `window_end`.
- `data_format`은 `"parquet"`.
- `sha256`은 events 파일의 64자리 hex (`sha256:` 접두사 허용).
- `data_uri`는 요청의 `eventsUri`와 정확히 같아야 한다.

### events.parquet 필수 컬럼 12개

```text
event_id, request_id, question,
initial_label, initial_model, final_label, final_model,
user_satisfied, escalation_count, created_at,
router_model_version, pii_redacted
```

행 수가 `eventCount`와 정확히 일치해야 한다. 질문 원문에서 PII를 제거하고
`pii_redacted`로 표시한다.

### 응답

```json
// 202 Accepted
{"status": "accepted", "batchId": "...", "pipelineRunId": "projects/.../pipelineJobs/..."}
```

`accepted`는 **제출됐다**는 뜻이지 학습이 성공했다는 뜻이 아니다. 학습·평가·승인은
몇 시간이 걸린다.

### 멱등성 — 재시도 규칙

같은 `batchId`로 같은 내용을 다시 보내면 새 학습을 시작하지 않고 기존 `pipelineRunId`를
담아 `202`를 돌려준다.

**멱등성 키에 `requestedAt`은 포함되지 않는다.** 타임아웃 후 재시도할 때 타임스탬프를
새로 찍어도 되고 그대로 보내도 된다. 둘 다 같은 결과다. 키는 다음 5개다:

```text
batchId, schemaVersion, manifestUri, eventsUri, eventCount
```

### 오류

| Status | Code | 재시도 | 의미 |
| --- | --- | --- | --- |
| 409 | `BATCH_ID_CONFLICT` | 아니오 | 같은 batchId에 다른 내용. 새 batchId를 쓴다 |
| 409 | `RETRAINING_IN_PROGRESS` | **예** | 다른 재학습이 진행 중. GCS batch를 지우지 말고 나중에 같은 요청으로 재시도 |
| 422 | `UNSUPPORTED_SCHEMA_VERSION` | 아니오 | |
| 422 | `GCS_OBJECT_NOT_FOUND` | 아니오 | 게시부터 확인 |
| 422 | `MANIFEST_MISMATCH` | 아니오 | 필수 필드 누락, window 순서 오류, 또는 요청과 불일치 |
| 422 | `UNSUPPORTED_DATA_FORMAT` | 아니오 | `data_format`이 parquet이 아님 |
| 422 | `CHECKSUM_MISMATCH` | 아니오 | |
| 422 | `EVENT_SCHEMA_MISMATCH` | 아니오 | 필수 컬럼 누락 |
| 422 | `EVENT_COUNT_MISMATCH` | 아니오 | |
| 422 | `REQUEST_VALIDATION_FAILED` | 아니오 | body가 schema 위반 (`eventCount != 1000` 포함) |
| 502 | `PIPELINE_SUBMISSION_FAILED` | 아니오 | Vertex가 거부. ML 팀 확인 필요 |
| 503 | `GCS_BACKEND_UNAVAILABLE` / `PIPELINE_BACKEND_UNAVAILABLE` | **예** | |

동시에 실행 가능한 재학습은 하나다.

## 7. 아직 없는 것

**재학습 상태 조회 endpoint가 없다.** `202`를 받은 뒤 그 학습이 성공했는지 실패했는지
조회할 방법이 현재 없다. `GET /pipelines/{pipelineRunId}` 형태로 만들 예정이며 확정 전까지는
`pipelineRunId`를 로그에 남겨두고 ML 팀에 문의한다.

## 8. 모델 교체

승인된 새 모델이 나오면 FastAPI를 **재시작**해서 교체한다. 실행 중 교체하는 API는 없다.
교체 시 `/health`가 잠시 503이 되고, 이후 응답의 `modelVersion`이 바뀐다. Spring은
`modelVersion`을 기록만 하면 되고 별도 대응은 필요 없다.

현재 서빙 중인 모델:

```text
router-modernbert-base-routerbench-20260814-45dcc6d
```

## 9. 계약 변경 절차

이 API는 versioned contract다. 필드명, enum 값, 오류 코드를 바꾸려면 양쪽 contract test를
함께 갱신해야 한다. 변경이 필요하면 [`api-contract.md`](api-contract.md)를 기준으로
논의한다.
