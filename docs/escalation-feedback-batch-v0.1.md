# Escalation feedback batch 계약 v0.1

> **폐기 예정 legacy 계약:** Spring이 GCS에 manifest와 Parquet을 직접 올리는 이 방식은
> `feedback-inline-v1`으로 대체되었다. 새 Spring 구현은
> [`escalation-feedback-inline-v1.md`](escalation-feedback-inline-v1.md)를 따른다.

상태: 현재 외부 배포본과의 호환성 확인용. 교체 예정 저장소 코드에서는 폐기됨.

배포 호환성: 2026-09-12 기준 외부 배포본은 아직 `eventCount = 1000`만 허용한다. 이
문서의 1행 이상 계약은 새 FastAPI 배포가 완료된 뒤 활성화한다.

schema version: `feedback-export-v0.1`

대상: Spring 백엔드의 feedback export 및 `/escalate` 호출 구현

## 1. 결론

`POST /escalate`에 Escalation Log를 JSON 배열로 보내지 않는다. 백엔드는 다음 세
요소를 하나의 immutable batch로 만든다.

1. GCS의 `events.parquet`: Spring이 확정한 1행 이상의 Escalation Log batch
2. GCS의 `manifest.json`: Parquet의 위치, 행 수, checksum 등
3. `POST /escalate` request JSON: 위 두 객체의 URI와 batch 식별자

```text
events.parquet 업로드
  -> 실제 파일의 SHA-256 계산
  -> manifest.json 업로드
  -> POST /escalate 호출
```

JSON event 배열, 단일 `.json` event 파일, JSONL event 파일은 현재 `/escalate` 계약에서
지원하지 않는다. ML 전처리 함수 자체는 JSONL을 읽을 수 있지만 FastAPI의 사전 검증은
`data_format = "parquet"`만 허용한다.

권장 GCS 경로는 다음과 같다.

```text
gs://llm-router-mlserver/datasets/feedback/<batchId>/events.parquet
gs://llm-router-mlserver/datasets/feedback/<batchId>/manifest.json
```

같은 `batchId`의 객체를 덮어쓰지 않는다.

## 2. `/escalate` request JSON

HTTP header:

```http
Content-Type: application/json
Authorization: Bearer <internal-token>
X-Request-ID: <요청 추적 ID>
```

body 예시:

```json
{
  "batchId": "feedback-20260912-0001",
  "schemaVersion": "feedback-export-v0.1",
  "manifestUri": "gs://llm-router-mlserver/datasets/feedback/feedback-20260912-0001/manifest.json",
  "eventsUri": "gs://llm-router-mlserver/datasets/feedback/feedback-20260912-0001/events.parquet",
  "eventCount": 3,
  "requestedAt": "2026-09-12T06:30:00Z"
}
```

| 필드 | JSON 타입 | 규칙 |
| --- | --- | --- |
| `batchId` | string | 1~200자. 영문 대소문자, 숫자, `.`, `_`, `-`만 허용. immutable batch ID이자 멱등성 키 |
| `schemaVersion` | string | 반드시 `feedback-export-v0.1` |
| `manifestUri` | string | `gs://<bucket>/<object>` 형식의 `manifest.json` URI |
| `eventsUri` | string | `gs://<bucket>/<object>` 형식의 `events.parquet` URI |
| `eventCount` | integer | 1 이상의 숫자. Spring이 정한 batch 크기이며 문자열로 보내지 않는다. |
| `requestedAt` | string | UTC ISO 8601. `Z`로 끝내는 형식을 사용 |

이 JSON은 camelCase이며 위 6개 필드만 보낸다. 누락 필드와 알 수 없는 추가 필드는
`REQUEST_VALIDATION_FAILED`로 거부된다.

`manifestUri`는 batch 설명서의 위치이고 `eventsUri`는 실제 event 데이터의 위치다.
FastAPI는 manifest 안의 `data_uri`가 request의 `eventsUri`와 정확히 같은지도 확인한다.

## 3. `manifest.json`

완성 예시:

```json
{
  "schema_version": "feedback-export-v0.1",
  "batch_id": "feedback-20260912-0001",
  "created_at": "2026-09-12T06:29:58Z",
  "window_start": "2026-09-12T05:14:22Z",
  "window_end": "2026-09-12T06:27:41Z",
  "event_count": 3,
  "data_format": "parquet",
  "data_uri": "gs://llm-router-mlserver/datasets/feedback/feedback-20260912-0001/events.parquet",
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "producer_version": "spring-feedback-exporter-1.0.0",
  "watermark": "284391"
}
```

위 `sha256`은 형식 설명용 값이다. 실제로는 **업로드할 `events.parquet` 파일 바이트의
SHA-256**으로 교체한다.

`manifest.json`은 snake_case다. request JSON의 camelCase 이름을 그대로 쓰면 안 된다.

| 필드 | JSON 타입 | 규칙과 의미 |
| --- | --- | --- |
| `schema_version` | string | request의 `schemaVersion`과 정확히 같아야 함 |
| `batch_id` | string | request의 `batchId`와 정확히 같아야 함 |
| `created_at` | string | manifest 생성 시각, UTC ISO 8601 |
| `window_start` | string | batch에 포함된 event 중 가장 이른 `created_at`, UTC ISO 8601 |
| `window_end` | string | batch에 포함된 event 중 가장 늦은 `created_at`, UTC ISO 8601 |
| `event_count` | integer | request의 `eventCount` 및 Parquet 실제 행 수와 같은 1 이상의 값 |
| `data_format` | string | 반드시 `parquet` |
| `data_uri` | string | request의 `eventsUri`와 정확히 같아야 함 |
| `sha256` | string | Parquet 파일 바이트의 64자리 SHA-256 hex. 소문자, 접두사 없는 형식 권장 |
| `producer_version` | string | 이 파일을 만든 Spring exporter의 배포 버전. 예: `spring-feedback-exporter-1.0.0` |
| `watermark` | string | 이번 snapshot에 포함한 마지막 Escalation Log cursor. DB 증가 ID라면 10진 문자열 사용 |

모든 timestamp는 `Z`가 붙은 UTC로 만든다. 현재 수신 코드는 timezone 없는 manifest 시각도
UTC로 해석하지만, producer 계약에서는 허용하지 않는다. `window_start`는 `window_end`보다
늦을 수 없다.

현재 FastAPI와 ML 전처리는 manifest의 추가 필드를 무시할 수 있지만, v0.1 producer는 위
11개 필드만 작성한다. 새 metadata가 필요하면 schema version을 올려 양쪽 계약을 함께
변경한다.

`watermark`는 학습 코드가 해석하는 값이 아니라 export 재개 및 감사용 opaque cursor다.
백엔드는 batch 생성용 DB snapshot을 잡은 뒤 그 snapshot의 마지막 cursor를 기록하고,
다음 batch에서는 이전 watermark 이후의 로그부터 선택한다.

## 4. `events.parquet`

### 4-1. 한 행의 논리적 JSON 표현

아래 JSON은 백엔드 DTO와 의미를 설명하기 위한 것이다. 이 JSON을 그대로 GCS에 올리는
것이 아니라 1건 이상의 객체를 아래 Parquet schema로 직렬화한다.

```json
{
  "event_id": "esc-284391",
  "request_id": "req-019c1234",
  "question": "Explain why the sky appears blue.",
  "initial_label": "easy",
  "initial_model": "haiku",
  "final_label": "medium",
  "final_model": "sonnet",
  "user_satisfied": true,
  "escalation_count": 1,
  "created_at": "2026-09-12T06:27:41Z",
  "router_model_version": "router-modernbert-base-routerbench-20260814-45dcc6d",
  "pii_redacted": true
}
```

### 4-2. Parquet schema

| 컬럼 | Parquet 논리 타입 | 필수 | 규칙과 의미 |
| --- | --- | --- | --- |
| `event_id` | UTF-8 string | Y | 비어 있지 않은 Escalation Log 고유 ID. batch 안에서 중복 금지 |
| `request_id` | UTF-8 string | Y | 원래 사용자 요청의 추적 ID. 비어 있으면 안 됨 |
| `question` | UTF-8 string | Y | PII를 제거한 원 질문. 공백 문자열 금지 |
| `initial_label` | UTF-8 string | Y | 최초 router 판정. `easy`, `medium`, `hard` 중 하나 |
| `initial_model` | UTF-8 string | Y | 최초로 호출한 model routing key 또는 version. 비어 있으면 안 됨 |
| `final_label` | UTF-8 string | Y | 만족스러운 답을 만든 최종 tier. `easy`, `medium`, `hard` 중 하나 |
| `final_model` | UTF-8 string | Y | 최종 model routing key 또는 version. 비어 있으면 안 됨 |
| `user_satisfied` | boolean | Y | v0.1 export 대상은 반드시 `true` |
| `escalation_count` | int32 | Y | 1 이상의 정수 |
| `created_at` | timestamp millis, UTC | Y | Escalation Log 생성 시각 |
| `router_model_version` | UTF-8 string | Y | 최초 판정을 반환한 `/router/classify`의 `modelVersion` |
| `pii_redacted` | boolean | Y | 실제 PII 제거를 수행한 뒤 반드시 `true` |

컬럼명은 정확히 snake_case로 작성한다. v0.1에서는 위 12개 컬럼만 게시한다. 현재 수신
코드는 추가 컬럼을 읽을 수 있지만 이를 안정된 계약으로 보장하지 않으며, 불필요한 컬럼은
PII 유출 범위도 넓힌다.

### 4-3. event 선택 규칙

ML 전처리가 학습 후보로 인정하는 행은 다음 조건을 모두 만족해야 한다. Spring은 GCS에
게시하기 전에 같은 규칙으로 후보 행을 만든다. 재학습 trigger 크기는 Spring이 결정한다.

- 모든 필수 문자열이 비어 있지 않다.
- `event_id`가 batch 안에서 유일하다.
- `initial_label`과 `final_label`은 소문자 `easy`, `medium`, `hard` 중 하나다.
- 최종 tier가 최초 tier보다 엄격히 높다. 허용 조합은 `easy -> medium`, `easy -> hard`,
  `medium -> hard`뿐이다.
- `user_satisfied`는 JSON/Parquet boolean `true`다. 문자열 `"true"`는 허용하지 않는다.
- `pii_redacted`는 boolean `true`이며, flag만 설정한 것이 아니라 `question`에서 실제 PII를
  제거해야 한다.
- `escalation_count`는 1 이상의 정수다.
- `created_at`은 유효한 UTC 시각이다.

같은 질문의 문장부호·공백·대소문자 차이처럼 정규화 후 동일한 질문이 여러 번 나타나는
것은 허용한다. 전처리는 이를 학습 1행으로 합치고 반복 횟수에 따라 sample weight를 높인다.
반대로 정규화된 같은 질문에 서로 다른 `final_label`이 있으면 해당 질문의 행들은 모두
학습에서 제외된다.

## 5. `eventCount`의 정확한 의미

현재 구현에서 `eventCount`는 다음 세 숫자가 같다는 뜻이다.

```text
/escalate request.eventCount
= manifest.json event_count
= events.parquet 물리적 행 수 >= 1
```

이는 **전처리 후 최종 학습 행 수가 eventCount와 같다는 뜻은 아니다.** 정규화 중복은 한 행으로
합쳐지고, 충돌 label 및 기존 validation/test holdout과 겹치는 질문은 제외될 수 있다.
FastAPI는 최소 1행과 artifact 정합성만 검사한다. 신규 유효 feedback의 재학습 trigger
기준은 Spring이 소유한다.

따라서 `eventCount`는 현재 v0.1 request에서 **필수지만 본질적으로 중복된 값**이다. 이를
생략하면 FastAPI request schema가 422로 거부한다. 후속 v0.2에서는 `manifestUri`만 받고
manifest의 `event_count`와 `data_uri`를 실제 Parquet과 비교하도록 단순화할 수 있다.
FastAPI와 Spring의 schema 및 contract test를 함께 올리기 전에는 v0.1 필드를 생략하지
않는다.

특정 최종 unique training row 수를 보장해야 한다면 사전 정규화·충돌 검사 또는 refill
절차를 새 schema version과 함께 설계해야 한다. 현재 version trigger에는 이 값을 쓰지 않는다.

개발자 정책이 향후 `validatedUniqueBatch`로 전환되면 위 세 숫자의 일치 조건은 그대로
유지된다. 추가로 ML 전처리가 만든 accepted canonical row가 1건 이상이고, 그 canonical
batch digest가 이전에 commit된 batch와 달라야 version count 자격을 얻는다. 이 추가 검증은
Spring이 구현하지 않으며 HTTP payload도 바뀌지 않는다.

## 6. 게시 및 호출 체크리스트

1. 한 DB snapshot에서 event 선택 규칙과 Spring trigger 정책을 통과한 행을 1건 이상 고른다.
2. 정의된 12개 컬럼과 타입으로 `events.parquet`을 만든다.
3. 로컬 파일의 실제 행 수가 `eventCount`와 같고 1 이상인지 확인한다.
4. 완성된 파일 바이트의 SHA-256을 계산한다. 파일을 다시 만들면 checksum도 다시 계산한다.
5. 새 `batchId` 경로에 `events.parquet`을 먼저 업로드한다.
6. 동일한 ID, URI, count, checksum으로 `manifest.json`을 만들고 두 번째로 업로드한다.
7. 두 객체가 모두 조회되는 것을 확인한 뒤 `/escalate`를 호출한다.
8. `202 Accepted` 응답의 `pipelineRunId`를 `batchId`, `X-Request-ID`와 함께 저장한다.
9. timeout 또는 재시도 가능한 오류에는 GCS 객체를 바꾸지 않고 같은 5개 멱등성 필드로
   다시 호출한다. `requestedAt`은 새로 만들어도 된다.

FastAPI는 호출 시 schema version, GCS 접근, manifest 형식과 request 일치 여부, data
format, checksum, 필수 Parquet 컬럼 및 행 수를 검사한다. 행 내부의 의미 규칙과
중복·충돌은 현재 ML pipeline 전처리에서 검사하므로, `202 Accepted`가 모든 입력 행의 최종
학습 사용을 보장하지는 않는다.

## 7. 성공 응답

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

HTTP status는 `202 Accepted`다. 이는 pipeline 제출 성공만 의미하며 재학습 및 모델 승인을
의미하지 않는다. 이후 `statusUrl`을 polling하고 `status=succeeded`와
`activation.status=active`를 각각 확인한다. 전체 응답 계약은
[`api-contract.md`](api-contract.md)의 6장과 7장을 따른다.

## 8. 구현 근거

- FastAPI request 및 manifest model: `src/llm_router_serving/schemas.py`
- GCS, checksum, Parquet schema 사전 검증: `src/llm_router_serving/retraining/gcs.py`
- ML manifest 및 행 단위 전처리: `LLM_Router/ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py`
- ML acceptance 설정: `LLM_Router/ml/config/feedback_retraining_preprocessing.json`

## 9. 복사 가능한 예시 파일

- [`escalate-request.example.json`](examples/escalation-feedback-v0.1/escalate-request.example.json):
  `/escalate` request body
- [`manifest.example.json`](examples/escalation-feedback-v0.1/manifest.example.json): GCS manifest
- [`event-row.example.json`](examples/escalation-feedback-v0.1/event-row.example.json): Parquet 변환 전
  한 행의 논리적 표현
- [`README.md`](examples/escalation-feedback-v0.1/README.md): 사용 시 교체할 값과 주의사항
