# Inline escalation feedback 계약 v1

상태: 교체 예정 FastAPI 구현 기준

HTTP schema version: `feedback-inline-v1`

ML artifact schema version: `feedback-export-v0.1` (FastAPI 내부 생성)

대상: Spring 백엔드의 `POST /escalate` 구현

복사 가능한 request fixture:
[`examples/escalation-feedback-inline-v1/escalate-request.example.json`](examples/escalation-feedback-inline-v1/escalate-request.example.json)

## 1. 책임 경계

Spring은 재학습 대상으로 확정한 event를 JSON 배열로 한 번 전송한다. Spring은 Parquet,
manifest, checksum 또는 GCS URI를 생성하지 않고 재학습용 GCS 권한도 갖지 않는다.

FastAPI는 다음을 담당한다.

1. HTTP schema와 event 의미 검증
2. event 수 계산과 `eventId` 중복 검사
3. camelCase JSON을 ML용 snake_case Parquet으로 변환
4. SHA-256과 `feedback-export-v0.1` manifest 생성
5. 3-call cycle과 compact replay 구성
6. 파생 artifact를 GCS에 immutable하게 저장
7. Vertex AI PipelineJob 제출

## 2. HTTP 요청

```http
POST /escalate
Content-Type: application/json
Authorization: Bearer <internal-token>
X-Request-ID: <요청 추적 ID>
```

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

### Top-level 필드

| 필드 | JSON 타입 | 필수 | 규칙 |
| --- | --- | :---: | --- |
| `batchId` | string | Y | 1~200자, `[A-Za-z0-9._-]+`, immutable 멱등성 키 |
| `schemaVersion` | string | Y | `feedback-inline-v1` |
| `requestedAt` | string | Y | UTC ISO 8601, `Z` 사용 |
| `events` | array | Y | 1개 이상, 고정 최대 건수 없음 |

`eventCount`, `manifestUri`, `eventsUri`는 보내지 않는다. 알 수 없는 top-level 필드도
보내지 않는다.

### `events[]` 필드

| 필드 | JSON 타입 | 필수 | 규칙 |
| --- | --- | :---: | --- |
| `eventId` | string | Y | 비공백, 요청 배열 안에서 고유 |
| `requestId` | string | Y | 원 사용자 요청 추적 ID |
| `question` | string | Y | 비공백, 최대 100,000자, 실제 PII 제거 완료 |
| `initialLabel` | string | Y | `easy`, `medium`, `hard` |
| `initialModel` | string | Y | 최초 model routing key/version |
| `finalLabel` | string | Y | `easy`, `medium`, `hard`, initial보다 높은 tier |
| `finalModel` | string | Y | 만족스러운 답변을 만든 model key/version |
| `userSatisfied` | boolean | Y | JSON boolean `true`만 허용 |
| `escalationCount` | integer | Y | 1 이상 |
| `createdAt` | string | Y | UTC ISO 8601, `Z` 사용 |
| `routerModelVersion` | string | Y | 최초 `/router/classify` 응답의 `modelVersion` |
| `piiRedacted` | boolean | Y | 실제 PII 제거 후 JSON boolean `true`만 허용 |

허용되는 tier 상승은 `easy -> medium`, `easy -> hard`, `medium -> hard`다. 문자열
`"true"`, timezone 없는 시각, UTC가 아닌 offset, 같은 `eventId`의 반복은 거부한다.

## 3. FastAPI 내부 변환

FastAPI는 event를 다음 Parquet 컬럼으로 변환한다.

```text
event_id, request_id, question,
initial_label, initial_model, final_label, final_model,
user_satisfied, escalation_count, created_at,
router_model_version, pii_redacted
```

`created_at`은 `timestamp[ms, UTC]`, `escalation_count`는 int32, boolean 두 필드는 bool,
나머지는 UTF-8 string이다. FastAPI는 변환된 Parquet bytes에서 SHA-256을 계산하고 ML
pipeline이 요구하는 `feedback-export-v0.1` manifest를 자동 생성한다.

Spring의 `feedback-inline-v1`은 HTTP transport schema이고, FastAPI가 생성하는
`feedback-export-v0.1`은 ML pipeline artifact schema다. 두 version은 역할이 다르다.

## 4. 멱등성과 재시도

fingerprint에는 다음이 들어간다.

```text
batchId + schemaVersion + events 전체 내용(eventId 순 정렬)
```

`requestedAt`은 제외한다.

- 같은 `batchId`와 같은 events 재시도: 기존 `pipelineRunId` 반환
- 같은 `batchId`와 다른 events: `409 BATCH_ID_CONFLICT`
- 다른 재학습 실행 중: `409 RETRAINING_IN_PROGRESS`; 같은 JSON으로 backoff 후 재시도
- timeout: 새 batch ID를 만들지 말고 기존 batch와 events로 재시도

FastAPI는 Parquet과 manifest를 GCS에 기록하고 Vertex 제출이 성립한 뒤에만
`202 Accepted`를 반환한다. `202`는 학습 완료나 모델 반영을 뜻하지 않는다.

## 5. 검증 오류

| HTTP | Code | 대표 조건 |
| --- | --- | --- |
| 409 | `BATCH_ID_CONFLICT` | 같은 batch ID에 다른 events |
| 409 | `RETRAINING_IN_PROGRESS` | 다른 pipeline 실행 중 |
| 422 | `UNSUPPORTED_SCHEMA_VERSION` | `feedback-inline-v1`이 아님 |
| 422 | `REQUEST_VALIDATION_FAILED` | 빈 배열, 필드·타입·label·UTC·중복 ID·boolean 조건 위반 |
| 502 | `PIPELINE_SUBMISSION_FAILED` | Vertex 제출 실패 |
| 503 | `GCS_BACKEND_UNAVAILABLE` | FastAPI의 artifact 저장/조회 실패 |
| 503 | `PIPELINE_BACKEND_UNAVAILABLE` | Vertex backend 연결 실패 |

## 6. Spring 구현 체크리스트

1. 한 DB snapshot에서 보낼 event를 1건 이상 선택한다.
2. 실제 PII를 제거하고 `piiRedacted=true`로 설정한다.
3. 12개 event 필드를 camelCase와 정확한 JSON 타입으로 직렬화한다.
4. top-level 네 필드와 `X-Request-ID`를 보낸다.
5. `batchId`와 전송한 events를 재시도 가능하게 보존한다.
6. 현재 `202` 응답의 `batchId`와 `pipelineRunId`를 저장한다.
7. 상태 API가 구현·배포된 뒤에는 추가되는 version/status URL도 저장하고 polling한다.
8. `202`를 학습 완료로 처리하지 않는다.

Spring에 Parquet writer, Google Cloud Storage SDK, checksum 또는 manifest 생성 코드를
추가하지 않는다.
