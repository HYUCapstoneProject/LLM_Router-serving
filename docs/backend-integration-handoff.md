# Spring 백엔드용 FastAPI 연동 명세

이 문서 하나와 OpenAPI 파일만으로 Spring 연동을 구현할 수 있도록 작성한 v1 계약이다.

- OpenAPI: [backend-api-v1/openapi.yaml](backend-api-v1/openapi.yaml)
- 요청 fixture:
  [examples/escalation-feedback-inline-v1/escalate-request.example.json](examples/escalation-feedback-inline-v1/escalate-request.example.json)
- 상태별 response fixture: [examples/retraining-api-v1/](examples/retraining-api-v1/)

## 1. 설계 원칙과 책임 경계

Spring에는 학습 인프라 책임을 두지 않는다. Spring이 HTTP event DTO와 ML용
Parquet/manifest를 함께 관리하면 같은 데이터를 두 번 가공하게 되고, GCS SDK·bucket
경로·checksum·artifact schema·IAM 변화가 Spring 코드까지 전파된다.

Spring이 담당하는 범위:

1. 사용자 요청, 최초 routing 결과, 상향 결과와 만족 여부를 DB에 기록한다.
2. 재학습 대상으로 보낼 event를 선택한다.
3. 질문의 개인정보를 실제로 제거한다.
4. event 전체를 feedback-inline-v1 JSON으로 FastAPI에 보낸다.
5. 접수 결과와 재학습 상태를 저장·조회한다.

FastAPI와 ML system이 담당하는 범위:

1. HTTP schema와 event 의미를 검증한다.
2. event를 정규화하고 ML용 Parquet으로 변환한다.
3. SHA-256과 manifest를 생성한다.
4. current cycle과 compact replay를 구성한다.
5. FastAPI service account로 GCS에 저장한다.
6. Vertex AI PipelineJob을 제출한다.
7. 학습·평가·gate·승격·model 활성화 상태를 Spring용 API로 변환한다.
8. 승인 model을 검증하고 in-memory router를 자동 교체한다.

Spring에는 Parquet writer, Google Cloud Storage SDK, GCS bucket 정보, Vertex SDK 또는
재학습 GCS IAM을 추가하지 않는다.

## 2. 접속과 공통 규칙

환경별 설정:

~~~text
llm-router.fastapi.base-url=https://<fastapi-host>
llm-router.fastapi.internal-token=<secret>
llm-router.fastapi.connect-timeout=3s
llm-router.fastapi.read-timeout=30s
~~~

VM IP, GCP project와 region을 Java 코드에 하드코딩하지 않는다. HTTP connection pooling과
keep-alive를 사용한다.

인증이 필요한 endpoint의 공통 header:

~~~http
Content-Type: application/json; charset=utf-8
Authorization: Bearer <internal-token>
X-Request-ID: <Spring trace ID>
~~~

- 날짜·시각은 UTC ISO 8601과 Z를 사용한다.
- request JSON의 알 수 없는 필드는 허용되지 않는다.
- Spring은 모든 요청에 고유 X-Request-ID를 보낸다.
- FastAPI는 같은 ID를 response header에 반환한다.
- token과 질문 원문은 application log에 남기지 않는다.

공통 오류 body:

~~~json
{
  "error": "request body does not match the API contract",
  "code": "REQUEST_VALIDATION_FAILED",
  "requestId": "0f1d3f52-f560-4b0b-93d0-acde58745ae1"
}
~~~

Spring은 error 문자열이 아니라 code로 분기한다.

## 3. Endpoint 목록

| Method/path | 용도 | 정상 HTTP |
| --- | --- | ---: |
| POST /router/embed | 질문 embedding | 200 |
| POST /router/rerank | pgvector 후보의 선택적 정밀 점수 | 200 |
| POST /router/classify | 질문 난이도 분류 | 200 |
| GET /health | embedding/router readiness | 200 또는 503 |
| POST /escalate | inline feedback 접수·재학습 제출 | 202 |
| GET /retraining-runs/{batchId} | 학습 단계·최종 결과·model 활성화 조회 | 200 |
| GET /router/model/current | 승인 목표 model과 실제 serving model 조회 | 200 |

GET/PUT /developer/retraining/version-policy는 개발자 운영 도구용이다. Spring에서는 호출하지
않는다.

외부 계약 확인:

~~~text
GET /openapi.json
GET /docs
~~~

/openapi.json은 code generation과 contract test의 기계 판독 기준이며, /docs는 Swagger
UI다.

## 4. 온라인 routing API

### 4.1 GET /health

Ready:

~~~json
{
  "status": "ok",
  "embedModel": "loaded",
  "routerModel": "loaded",
  "rerankModel": "disabled"
}
~~~

BGE와 router 중 하나라도 준비되지 않으면 HTTP 503과 status=not_ready를 반환한다.
rerankModel은 disabled | loading | loaded | error이며 선택 계층이므로 readiness를 막지
않는다. /health는 model version 조회용이 아니다.

### 4.2 POST /router/embed

Request:

~~~json
{
  "message": "How does photosynthesis work?"
}
~~~

Response:

~~~json
{
  "embedding": [0.0123, -0.0456],
  "dim": 1024,
  "modelVersion": "bge-large-en-v1.5-d4aa690"
}
~~~

- message: 공백이 아닌 문자열
- embedding: L2 정규화된 1,024차원 vector
- dim: embedding 길이와 같아야 함
- modelVersion: embedding cache namespace에 포함

다른 embedding modelVersion의 vector를 같은 cache 공간에서 비교하지 않는다.

### 4.3 POST /router/rerank

Spring이 pgvector에서 찾은 후보를 query와 함께 보낸다. 후보 검색, threshold와 최종
cache hit/miss 판정은 계속 Spring이 소유한다.

~~~json
{
  "query": "How does photosynthesis work?",
  "candidates": [
    "Why do plants need sunlight?",
    "How do plants convert light into chemical energy?"
  ]
}
~~~

Response:

~~~json
{
  "scores": [0.013, 0.941],
  "modelVersion": "cross-encoder/quora-distilroberta-base@f62e7a4"
}
~~~

- scores[i]는 candidates[i]에 대응하며 FastAPI가 정렬하지 않는다.
- 배포 기본 후보 상한은 16개다.
- threshold는 modelVersion별로 관리한다. BGE cosine threshold와 혼용하지 않는다.
- 503 RERANK_MODEL_NOT_READY, 추론 오류 또는 비유한 점수는 cache miss로 처리한다.
- 리랭커는 기본 비활성이며 별도 운영 승인 후에만 환경변수로 켠다.

### 4.4 POST /router/classify

Request:

~~~json
{
  "message": "How does photosynthesis work?"
}
~~~

Response:

~~~json
{
  "tier": "medium",
  "model": "sonnet",
  "confidence": 0.87,
  "modelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d"
}
~~~

- tier: easy | medium | hard
- model: haiku | sonnet | opus, 참고값
- confidence: 0.0 이상 1.0 이하
- modelVersion: 분류에 실제 사용된 router model

실제 provider model 선택은 Spring의 versioned routing policy가 결정한다. Spring은
modelVersion을 원 요청 log에 보존하고 escalation event의 routerModelVersion으로 사용한다.

## 5. Escalation event 생성 규칙

event 하나는 다음 사실을 나타낸다.

1. router가 최초 tier/model을 선택했다.
2. 최초 응답으로 사용자가 만족하지 않아 더 높은 tier로 상향했다.
3. 최종 상향 응답에는 사용자가 만족했다.
4. 질문에서 PII를 제거했다.

허용 label 이동:

~~~text
easy -> medium
easy -> hard
medium -> hard
~~~

같은 label, 하향 label, 만족 확인이 없는 사례, PII 제거 전 질문은 보내지 않는다.

## 6. POST /escalate

### 6.1 Request

~~~json
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
~~~

Top-level은 정확히 네 필드다.

| 필드 | Java 타입 | 규칙 |
| --- | --- | --- |
| batchId | String | 1~200자, 영문·숫자·점·밑줄·하이픈; immutable 멱등성 키 |
| schemaVersion | String | 상수 feedback-inline-v1 |
| requestedAt | Instant | 요청 생성 시각, UTC |
| events | List<EscalationEvent> | 1개 이상, 배열 안에서 eventId 고유 |

event는 정확히 12개 필드다.

| 필드 | Java 타입 | 규칙과 출처 |
| --- | --- | --- |
| eventId | String | 고유 escalation ID, 비공백, 최대 200자 |
| requestId | String | 최초 사용자 요청 추적 ID, 최대 200자 |
| question | String | PII 제거가 끝난 질문, 비공백, 최대 100,000자 |
| initialLabel | RouterTier | 최초 /router/classify.tier |
| initialModel | String | 최초 실제 응답 model, 최대 300자 |
| finalLabel | RouterTier | 만족 응답의 더 높은 tier |
| finalModel | String | 만족 응답 model, 최대 300자 |
| userSatisfied | boolean | 반드시 JSON boolean true |
| escalationCount | int | 최초 응답 이후 상향 횟수, 1 이상 |
| createdAt | Instant | event 생성 시각, UTC |
| routerModelVersion | String | 최초 classify response의 modelVersion |
| iRedactedpi | boolean | 실제 PII 제거 완료 후 true |

다음 필드는 존재하지 않는다.

~~~text
eventCount
manifestUri
eventsUri
~~~

FastAPI가 events 길이를 계산하므로 eventCount는 필요 없다. GCS URI와 manifest도 FastAPI가
내부에서 생성한다.

FastAPI 최소 batch 조건은 event 1개다. 애플리케이션 수준의 고정 최대 건수는 없다. 다만
reverse proxy와 HTTP body-size 제한은 배포 설정이므로 매우 큰 payload를 자동 분할하지
말고 운영 한도를 협의한다. HTTP 호출 1회가 version trigger 1회이기 때문이다.

piiRedacted=true는 Spring의 완료 선언이다. FastAPI가 보장하는 것은 해당 boolean 검사이며,
질문 본문의 실제 개인정보 탐지·마스킹은 Spring 책임이다.

### 6.2 Java DTO

~~~java
public enum RouterTier {
    easy, medium, hard
}

public record EscalationEvent(
    String eventId,
    String requestId,
    String question,
    RouterTier initialLabel,
    String initialModel,
    RouterTier finalLabel,
    String finalModel,
    boolean userSatisfied,
    int escalationCount,
    Instant createdAt,
    String routerModelVersion,
    boolean piiRedacted
) {}

public record EscalateRequest(
    String batchId,
    String schemaVersion,
    Instant requestedAt,
    List<EscalationEvent> events
) {}
~~~

Instant를 사용하고 timezone이 없는 LocalDateTime, epoch number 또는 문자열 boolean을
사용하지 않는다.

### 6.3 Accepted response

HTTP 202 Accepted:

~~~json
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
~~~

202는 GCS artifact 생성과 Vertex PipelineJob 제출 완료다. 학습·평가·승격 또는 serving
반영 성공이 아니다.

Spring이 저장할 필드:

~~~text
batch_id
pipeline_run_id
logical_model_version
candidate_model_version
training_mode
version_policy_mode
status_url
submitted_at
request_payload 또는 재생성 가능한 event snapshot
~~~

pipelineRunId는 Google resource name이며 HTTP URL이 아니다. statusUrl만 FastAPI base
URL에 resolve해 호출한다.

### 6.4 멱등성과 재시도

batch fingerprint:

~~~text
batchId + schemaVersion + events 전체 내용(eventId 순 정렬)
~~~

requestedAt과 events 배열 순서는 fingerprint에서 제외된다.

- 같은 batchId와 같은 event 내용: 기존 pipelineRunId로 202
- 같은 batchId와 다른 event 내용: 409 BATCH_ID_CONFLICT
- 다른 batch 학습 중: 409 RETRAINING_IN_PROGRESS
- client timeout, 502 또는 503: 같은 batchId와 같은 payload로 재시도

재시도 때문에 새 batch ID를 만들거나 payload를 나누지 않는다. 한번 제출을 시작한
batchId의 event 집합은 수정하지 않는다.

## 7. 즉시 오류 처리

| HTTP/code | Spring 처리 |
| --- | --- |
| 400 INVALID_MESSAGE | routing 입력 검증 후 수정 |
| 401 UNAUTHORIZED | token 설정 점검, 자동 무한 재시도 금지 |
| 403 FORBIDDEN | FastAPI의 GCS/Vertex IAM 운영 알림 |
| 404 RETRAINING_RUN_NOT_FOUND | batch ID와 저장 상태 점검 |
| 409 BATCH_ID_CONFLICT | 자동 재시도 금지, 동일 ID payload 조사 |
| 409 RETRAINING_IN_PROGRESS | 원 payload를 유지하고 backoff |
| 409 GCS_OBJECT_CONFLICT | 자동 덮어쓰기 금지, 운영 조사 |
| 422 REQUEST_VALIDATION_FAILED | 데이터/DTO 수정 또는 dead-letter |
| 422 UNSUPPORTED_SCHEMA_VERSION | client/server schema version 점검 |
| 500 EMBEDDING_INFERENCE_FAILED | fallback과 운영 알림 |
| 500 ROUTER_INFERENCE_FAILED | 상위 tier fallback과 운영 알림 |
| 502 PIPELINE_SUBMISSION_FAILED | 같은 batch로 제한 재시도 |
| 503 EMBED_MODEL_NOT_READY | cache/routing 장애 정책 적용 |
| 503 ROUTER_MODEL_NOT_READY | 상위 tier fallback |
| 503 GCS_BACKEND_UNAVAILABLE | 같은 batch로 backoff |
| 503 PIPELINE_BACKEND_UNAVAILABLE | 같은 batch로 backoff |
| 503 PIPELINE_STATUS_UNAVAILABLE | polling backoff |

transient 재시도는 5초, 15초, 30초, 이후 60초 상한과 jitter를 권장한다.

## 8. GET /retraining-runs/{batchId}

호출 예:

~~~http
GET /retraining-runs/feedback-20260912-0001
Authorization: Bearer <internal-token>
X-Request-ID: req-retrain-status-1
~~~

top-level 상태:

~~~text
queued | running | cancelling | paused |
succeeded | rejected | failed | cancelled | unknown
~~~

terminal=true인 상태는 succeeded | rejected | failed | cancelled다. 실패·거부·취소도 조회
자체는 성공이므로 HTTP 200으로 반환한다.

stage:

~~~text
initializing | feedbackPreparation | feedbackMerge | training |
evaluation | packaging | gate | publishing | complete
~~~

Spring 분기:

- terminal=false: polling 계속
- status=succeeded: gate 통과, approved package와 CURRENT.json 게시 완료
- status=rejected: 성능 gate 거부
- status=failed: 기술 실패, result.failureCode로 분기
- status=cancelled: 취소 완료
- stale=true 또는 status=unknown: 완료로 확정하지 않음
- activation.status=active: FastAPI가 새 model로 실제 추론 중

안정된 failure code:

~~~text
FEEDBACK_VALIDATION_FAILED
NO_UNIQUE_FEEDBACK
TRAINING_FAILED
EVALUATION_FAILED
DEPLOY_GATE_REJECTED
PACKAGE_FAILED
PUBLISH_FAILED
PIPELINE_CANCELLED
UNKNOWN_PIPELINE_FAILURE
~~~

activation status:

| 값 | 의미 |
| --- | --- |
| pending | 승격 전 또는 FastAPI가 새 package를 적재 중 |
| active | 실제 serving model이 candidate와 같음 |
| loadFailed | 새 model 적재 실패, 이전 known-good model 계속 사용 |
| superseded | 더 최신 model이 이미 serving 중 |
| notApplicable | 거부·실패·취소로 반영 대상 아님 |

polling은 202 수신 10초 후 시작하고 15초, 30초, 이후 60초 상한으로 backoff한다.

DB column 권장안:

~~~text
batch_id, pipeline_run_id,
logical_model_version, candidate_model_version,
pipeline_status, pipeline_stage,
result_outcome, failure_code, retryable,
activation_status,
submitted_at, completed_at, last_polled_at
~~~

전체 response 필드와 null 규칙은 OpenAPI의 RetrainingStatusResponse schema를 따른다.

## 9. GET /router/model/current

이 API는 승인 pointer와 실제 메모리 model을 구분한다.

- desired.modelVersion: CURRENT.json이 가리키는 승인 model
- serving.modelVersion: FastAPI process가 실제 요청에 사용하는 model
- previousKnownGood.modelVersion: rollback용 보존 model

status:

~~~text
inSync | loading | loadFailed | static | unavailable
~~~

inSync이고 desired와 serving version이 같을 때 자동 반영이 끝난 것이다. loadFailed에서는
이전 known-good model이 계속 정상 응답할 수 있다.

## 10. URI 처리

| 값 | 예 | Spring 처리 |
| --- | --- | --- |
| FastAPI base URL | https://router.example.com | HTTP client 설정 |
| statusUrl | /retraining-runs/feedback-... | base URL에 안전하게 resolve해 GET |
| pipelineRunId | projects/.../pipelineJobs/... | DB/log 식별자로만 저장 |
| GCS URI | gs://bucket/object | Spring API 계약에 없음 |

상대 statusUrl은 문자열 단순 결합 대신 HTTP client URI builder로 resolve한다.
pipelineRunId에 https://를 붙이거나 Spring이 Vertex API를 호출하지 않는다.

## 11. Version 진행 방식

승격까지 성공한 고유 HTTP batch 3회가 한 major cycle이다.

~~~text
n.0
  -> 첫 성공 batch: n.1, minor_fine_tune
  -> 둘째 성공 batch: n.2, minor_fine_tune
  -> 셋째 성공 batch: (n+1).0, major_consolidation
~~~

major cycle 5회마다 pristine_rebase를 실행한다. 같은 batch의 멱등 재시도와 실패한 pipeline은
성공 횟수를 늘리지 않는다. event 수는 version 횟수와 무관하다.

개발자 API는 successfulHttpRequest와 validatedUniqueBatch mode를 전환하지만, Spring은 이
endpoint를 호출하거나 version을 직접 계산하지 않는다. /escalate response와 status
response의 version을 그대로 저장한다.

## 12. Spring 완료 체크리스트

- [ ] FastAPI base URL과 internal token을 환경 설정으로 분리했다.
- [ ] OpenAPI 파일로 DTO/client 또는 contract test를 생성했다.
- [ ] reranker 비활성·503·오류를 cache miss로 처리하고 modelVersion별 threshold를 관리한다.
- [ ] top-level 4필드와 event 12필드를 정확히 구현했다.
- [ ] eventCount, manifestUri, eventsUri가 없다.
- [ ] question의 실제 PII 제거 후에만 piiRedacted=true를 설정한다.
- [ ] 최초 classify modelVersion을 escalation까지 보존한다.
- [ ] batchId와 event snapshot을 재시도용으로 보존한다.
- [ ] 202를 학습 완료로 취급하지 않는다.
- [ ] statusUrl로 terminal 상태와 activation을 polling한다.
- [ ] HTTP/code별 retry, fallback과 dead-letter 정책이 있다.
- [ ] pipelineRunId를 URL로 호출하지 않는다.
- [ ] Spring service account에 재학습 GCS 권한을 추가하지 않았다.
- [ ] /developer/**를 호출하지 않는다.
