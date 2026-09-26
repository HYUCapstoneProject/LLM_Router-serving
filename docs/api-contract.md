# FastAPI 내부 API·Spring 연동 계약

기준일: 2026-09-23. 현행 source의 HTTP 계약과 해석 범위를 한곳에 정리한다. 실제 배포 상태를 보증하지 않는다.
설치·운영은 [README](../README.md), 학습 정책은 [ML 최종 명세](../../LLM_Router/docs/ml/README.md#retraining), 과거 계약은 [deprecated settings](../../LLM_Router/docs/deprecated-settings.md)를 따른다.

기계 판독 자료는 [OpenAPI YAML](backend-api-v1/openapi.yaml)·[JSON](backend-api-v1/openapi.json), 필드 구현은 [schemas.py](../src/llm_router_serving/schemas.py), 실제 route와 오류 처리는 [main.py](../src/llm_router_serving/main.py)다. 고정 OpenAPI는 전달 시점의 snapshot이므로 변경 시 live `/openapi.json`과 대조한다. Spring client는 `Routing`, `Health`, `Retraining`, `Model` 범위를 사용하고 `Developer`는 제외한다. 예시의 용도는 [fixture 안내](examples/README.md)에서 구분한다.

## 1. 책임·접속·공통 규칙

Spring은 pgvector 조회, cache threshold/hit/miss, 실제 provider 모델 ID·LLM 호출, feedback 선택·PII 제거·전송을 소유한다. FastAPI는 추론, inline 검증, 내부 Parquet/manifest/GCS, Vertex 제출과 승인 모델 적재를 소유한다. Spring DTO에 GCS·Vertex SDK, URI 조립·checksum·Parquet writer를 추가하지 않는다.

- Base URL/token은 환경 설정이다. 권장 초기 connect/read timeout은 3초/30초이며 운영 측정으로 조정한다.
- `Content-Type: application/json; charset=utf-8`, 내부 API에는 `Authorization: Bearer <internal-token>`을 보낸다. `/health`는 인증 없이 readiness를 제공한다. token 미설정 시 내부 인증은 비활성이다.
- `X-Request-ID`는 전달값을 응답 header에 되돌려주고, 없으면 UUID를 생성한다.
- 요청의 미지 필드는 거부한다. 시각은 UTC ISO 8601 `Z` 문자열로 보내고 Java에서는 `Instant`를 사용한다. Boolean은 JSON boolean으로 보낸다.
- 질문·token을 클라이언트 로그에 남기지 않는다. 서버의 검증 오류 로그 redaction 한계는 마지막 절에 명시한다.
- 오류 body는 `{ "error": "설명", "code": "안정된 코드", "requestId": "추적 ID" }`다. 문자열 `error` 대신 `code`로 분기한다. GCS 관련 오류 설명에 내부 URI가 포함될 수 있으므로 최종 사용자에게 raw body를 전달하지 않는다.

| Method / path | 정상 HTTP | 용도 |
| --- | ---: | --- |
| `POST /router/embed` | 200 | BGE 임베딩 |
| `POST /router/rerank` | 200 | 선택적 cache 후보 점수 |
| `POST /router/classify` | 200 | R0 난이도 분류 또는 R1 후보 선택을 기존 3단계로 반환 |
| `GET /health` | 200/503 | 추론 readiness |
| `POST /escalate` | 202 | inline feedback 접수·재학습 제출 |
| `GET /retraining-runs/{batchId}` | 200 | 상태·집계·활성화 조회 |
| `GET /router/model/current` | 200/503 | 승인 목표·실제 serving 모델 |
| `GET/PUT /developer/retraining/version-policy` | 200 | 별도 개발자 정책 조회·예약 |

## 2. 온라인 추론

### `POST /router/embed`

요청은 `{ "message": "How does photosynthesis work?" }`다. 공백이 아닌 문자열이며 기본 문자 상한은 100,000이다. 초과·누락·잘못된 필드는 `400 INVALID_MESSAGE`다.

응답은 `embedding: number[]`, `dim: integer`, `modelVersion: string`이다. 현재 BGE 출력은 L2 정규화한 유한수 1,024개, `dim=1024`다. BGE는 CLS pooling, prefix 없이 기본 512 token까지 입력한다. `modelVersion`은 설정에서 정한 값으로 revision suffix를 자동 생성하지 않는다. Spring은 embedding version이 다른 vector를 같은 cache namespace에서 비교하지 않는다.

### `POST /router/rerank`

```json
{
  "query": "How does photosynthesis work?",
  "candidates": ["Why do plants need sunlight?", "How do plants convert light into chemical energy?"]
}
```

응답 예: `{ "scores": [0.013, 0.941], "modelVersion": "cross-encoder/quora-distilroberta-base@f62e7a4" }`.

`query`와 후보는 비공백 문자열이다. schema 상한은 후보 32개, 설정 기본 상한은 16개다. schema 범위 안에서 설정 상한을 초과하면 `400 TOO_MANY_CANDIDATES`, 빈 배열·비공백 조건 위반·32개 초과는 `422 REQUEST_VALIDATION_FAILED`다. 각 문자열의 문자 상한도 적용한다.

`scores[i]`는 `candidates[i]`와 대응하며 정렬하지 않는다. 점수 scale/threshold는 model version별로 관리하고 BGE cosine과 혼용하지 않는다. 기본 비활성·미로딩의 `503 RERANK_MODEL_NOT_READY`, 추론 실패·비유한 점수의 `500 RERANK_INFERENCE_FAILED`는 Spring에서 cache miss로 처리해 분류·LLM 경로로 진행한다.

### `POST /router/classify`

요청은 embed와 같다.

```json
{
  "tier": "medium",
  "model": "sonnet",
  "confidence": 0.87,
  "modelVersion": "router-modernbert-example-version"
}
```

`tier`는 `easy | medium | hard`, 참고값 `model`은 `haiku | sonnet | opus`다. 실제 provider ID는 Spring의 versioned routing policy가 결정한다. `modelVersion`은 실제 추론 classifier의 version이며 최초 요청 기록과 feedback의 `routerModelVersion`에 보존한다.

응답은 R0·R1 모두 위 네 필드로 고정한다. `confidence`는 정책 없는 legacy R0에서는
3-class logits의 raw softmax 최댓값이고, 명시적 정책을 사용하는 R0에서는 선택된
후보의 softmax 점수, R1에서는 선택된 후보의 sigmoid 점수다. 모두 비용 차감 전
점수이며 0~1 범위다. R1 값을 난이도 confidence나 보정된 서비스 성공 확률로 해석하지
않는다. legacy 모델의 설정 기본 최대 길이는 8,192 token이다. 명시적 입력·길이 계약이
있는 패키지는 검증된 패키지의 추론 길이를 사용한다. 512로 학습한 이력을 8,192로
수정하지 않으며, 새 패키지 승격 없이 운영 길이를 자동 변경하지 않는다.

### `GET /health`

```json
{ "status": "ok", "embedModel": "loaded", "routerModel": "loaded", "rerankModel": "disabled" }
```

BGE/router의 상태는 `loading | loaded | error`, reranker는 여기에 `disabled`가 추가된다. 두 필수 모델이 모두 loaded일 때 200/`ok`, 아니면 같은 schema로 503/`not_ready`다. reranker는 readiness를 막지 않는다. 모델 version은 current-model API에서 조회한다.

## 3. `POST /escalate`

Spring은 한 DB snapshot에서 선택한 event의 실제 PII를 제거하고 전송한다. 전송 크기·시점은 Spring 정책이다. 현행 HTTP는 1건 이상이며 고정 1,000건 조건이나 애플리케이션 수준 최대 event 수가 없다. proxy/body-size 한도는 별도 배포 설정이다.

```json
{
  "batchId": "feedback-20260923-0001",
  "schemaVersion": "feedback-inline-v1",
  "requestedAt": "2026-09-23T06:30:00Z",
  "events": [{
    "eventId": "esc-284391",
    "requestId": "req-019c1234",
    "question": "Explain why the sky appears blue.",
    "initialLabel": "easy",
    "initialModel": "haiku",
    "finalLabel": "medium",
    "finalModel": "sonnet",
    "userSatisfied": true,
    "escalationCount": 1,
    "createdAt": "2026-09-23T06:27:41Z",
    "routerModelVersion": "router-modernbert-example-version",
    "piiRedacted": true
  }]
}
```

Top-level은 정확히 다음 네 필드다.

| 필드 | JSON / Java | 규칙 |
| --- | --- | --- |
| `batchId` | string / String | 1~200자, `[A-Za-z0-9._-]+`, immutable 멱등성 키 |
| `schemaVersion` | string / String | `feedback-inline-v1` |
| `requestedAt` | string / Instant | UTC 요청 생성 시각 |
| `events` | array / List | 1개 이상, 배열 내 eventId 고유 |

각 event는 정확히 다음 12개 필드다.

| 필드 | JSON / Java | 규칙·출처 |
| --- | --- | --- |
| `eventId` | string / String | 비공백, 최대 200자, escalation ID |
| `requestId` | string / String | 비공백, 최대 200자, 최초 사용자 요청 ID |
| `question` | string / String | 비공백, 최대 100,000자, 실제 PII 제거 완료 |
| `initialLabel` | string / enum | 최초 `easy | medium | hard` |
| `initialModel` | string / String | 비공백, 최대 300자, 최초 응답 model 식별자 |
| `finalLabel` | string / enum | 만족 응답의 더 높은 tier |
| `finalModel` | string / String | 비공백, 최대 300자, 만족 응답 model 식별자 |
| `userSatisfied` | boolean / boolean | strict JSON `true` |
| `escalationCount` | integer / int | 상향 호출 횟수, 1 이상 |
| `createdAt` | string / Instant | UTC event 생성 시각 |
| `routerModelVersion` | string / String | 비공백, 최대 300자, 최초 classify의 version |
| `piiRedacted` | boolean / boolean | 실제 PII 제거 후 strict JSON `true` |

허용 이동은 `easy→medium`, `easy→hard`, `medium→hard`다. 두 boolean은 문자열 `"true"`를 허용하지 않는다. timezone 없는 값과 UTC가 아닌 offset, 중복 eventId, 동급·하향 이동은 거부한다. PII flag는 Spring의 완료 선언이며 FastAPI가 본문을 탐지·마스킹하지 않는다.

`eventCount`, `manifestUri`, `eventsUri`는 HTTP 필드가 아니며 보내면 422다. 과거 `feedback-export-v0.1` HTTP 입력도 현재 endpoint에서 지원하지 않는다.

### 접수·내부 artifact

FastAPI는 eventId 순 정렬 → Parquet → SHA-256 → 내부 `feedback-export-v0.1` manifest → 현재 cycle/replay → immutable GCS 저장 → Vertex PipelineJob 제출을 수행한다. HTTP `feedback-inline-v1`과 내부 artifact `feedback-export-v0.1`은 역할이 다르다.

Parquet column은 `event_id, request_id, question, initial_label, initial_model, final_label, final_model, user_satisfied, escalation_count, created_at, router_model_version, pii_redacted`다. `created_at`은 `timestamp[ms, UTC]`, 횟수는 int32, 두 flag는 boolean, 나머지는 UTF-8 string이다.

```text
<derived-root>/<candidate-version>/feedback/events.parquet
<derived-root>/<candidate-version>/feedback/manifest.json
<derived-root>/<candidate-version>/replay/events.parquet
<derived-root>/<candidate-version>/replay/manifest.json
```

현재 cycle은 이전 성공 호출들과 합치므로 pipeline 입력 event 수가 이번 HTTP 배열 길이보다 클 수 있다. 정규화·상충 라벨·holdout 검사 후 실제 학습 행 수는 다시 줄 수 있다. 경로와 manifest/checksum은 Spring 계약에 노출하지 않는다.

`202 Accepted`의 필드는 `schemaVersion=escalation-acceptance-v1`, `status=accepted`, `batchId`, `pipelineRunId`, `logicalModelVersion`, `candidateModelVersion`, `trainingMode`, `versionPolicyMode`, `statusUrl`이다. 전체 예시는 [accepted fixture](examples/retraining-api-v1/escalate-accepted.example.json)에 있다. 이는 제출 완료이며 학습·승격·serving 반영 성공이 아니다.

### 멱등성·저장·재시도

Fingerprint는 batchId + schemaVersion + eventId로 정렬한 events 전체 내용이다. `requestedAt`과 배열 순서는 제외한다.

| 요청 | 결과 |
| --- | --- |
| 같은 batchId + 같은 events | 새 job 없이 기존 run의 202 |
| 같은 batchId + 다른 events | 409 `BATCH_ID_CONFLICT` |
| 다른 batch가 실행 중 | 409 `RETRAINING_IN_PROGRESS` |
| 같은 immutable GCS URI에 다른 bytes | 409 `GCS_OBJECT_CONFLICT` |

Spring은 원 event snapshot, 접수 응답의 모든 ID/version/mode/statusUrl, 제출 시각을 저장한다. timeout·502·503에는 같은 batchId와 내용으로 제한 재시도한다. 네트워크 재시도로 새 ID를 만들거나 batch를 분할하지 않는다. 권장 backoff는 5→15→30→최대 60초+jitter다.

SQLite lease와 Vertex의 결정적 job ID로 단일 pipeline을 관리한다. 같은 batch의 멱등 재시도는 version을 추가하지 않는다. 실패한 run을 같은 ID로 재전송해 새 학습을 시작하는 기능도 아니다. 단일 FastAPI replica 전제와 lease 복구는 [README](../README.md#재학습-운영-설정)에 있다.

## 4. `GET /retraining-runs/{batchId}`

Spring은 접수된 `statusUrl`을 같은 base URL에 안전하게 resolve해서 조회한다. `pipelineRunId`는 `projects/.../pipelineJobs/...` 식별자이며 HTTP URL이 아니다. Spring은 GCS·Vertex를 직접 조회하지 않는다.

응답의 전체 JSON shape는 [running fixture](examples/retraining-api-v1/retraining-running.example.json)와 OpenAPI `RetrainingStatusResponse`를 따른다.

| 필드 | 의미 |
| --- | --- |
| `schemaVersion`, `batchId`, `pipelineRunId` | `retraining-status-v1`와 실행 식별 |
| `status`, `terminal`, `stage` | 전체 상태·완료 여부·현재 단계 |
| `version` | logical/candidate version, training/initialization/policy mode |
| `stageProgress` | step/total/unit/percent/epoch/totalEpochs/heartbeatAt; 없으면 null |
| `stages` | 아래 일곱 실행 단계 순서, 각 name/status/startedAt/endedAt |
| `filtering`, `merge` | 계산된 전처리·병합 집계, 없으면 null |
| `result` | terminal 결과, 진행 중 null |
| `activation` | 해당 process의 candidate 반영 상태 |
| `submittedAt`, `startedAt`, `endedAt`, `lastUpdatedAt` | 제출/실행/완료/저장 상태 시각; 미관측 시작·종료는 null |
| `stale`, `warnings` | 상태 refresh 실패·부분 관측 경고 |

허용 `status`는 `queued | running | cancelling | paused | succeeded | rejected | failed | cancelled | unknown`이다. `terminal=true`는 succeeded/rejected/failed/cancelled이며 실패·거부·취소 조회도 HTTP 200이다. 현행 Vertex adapter는 활성 상태를 하나로 축약하므로 queued/cancelling/paused의 정밀 구분을 보장하지 않는다.

`stage`는 `initializing | feedbackPreparation | feedbackMerge | training | evaluation | packaging | gate | publishing | complete`다. `stages`는 feedbackPreparation부터 publishing까지 일곱 개를 순서대로 반환하고, 각 `status` enum은 `pending | running | succeeded | failed | cancelled | skipped | unknown`이다. 완료 상태의 top-level stage는 complete다. 현재 단계는 GCS artifact로 추정하므로 Vertex task 상세 상태·정확한 단계별 시간과 같다고 가정하지 않는다.

`stageProgress`는 GPU 학습의 optimizer step 비율이며 pipeline 전체 퍼센트가 아니다. terminal에서도 이전 heartbeat가 남을 수 있으므로 진행 여부는 top-level status로 판단한다. null은 미관측이지 0건·0%가 아니다.

### 집계와 최종 결과

- `inputEventCount = acceptedEventCount + excludedEventCount`
- `collapsedDuplicateEventCount = acceptedEventCount - canonicalTrainingRowCount`
- `feedbackInputRowCount = acceptedFeedbackRowCount + excludedFeedbackRowCount`

첫째는 행 조건 제외, 둘째는 같은 정규화 질문의 병합, 셋째는 base와 합칠 때 제외다. Spring 화면의 최종 채택 feedback 수는 `acceptedFeedbackRowCount`, base까지 포함한 최종 데이터 행 수는 `mergedTrainingRowCount`다. 현행 API는 upstream 수치를 변환하며 모든 산술 불변식을 별도로 검증하는 것은 아니다.

`exclusionDistribution`에는 충돌 final label, 중복 eventId, 만족/PII 조건 등의 사유가 들어간다. `decisionDistribution`의 주요 key는 `excluded_base_holdout_overlap`, `excluded_same_label_already_in_base`, `replace_exact_base_train_rows`, `append_new_question`이다. 새 key도 보존해 집계한다.

`result`는 `outcome: promoted | rejected | failed | cancelled`, candidate/logical version, nullable `gatePassed`, `failedGateChecks`, nullable `failureCode`/`failureStage`, `retryable`, nullable `message`/`approvedAt`을 포함한다. schema의 failureCode 목록은 다음과 같다.

```text
FEEDBACK_VALIDATION_FAILED | NO_UNIQUE_FEEDBACK | TRAINING_FAILED |
EVALUATION_FAILED | DEPLOY_GATE_REJECTED | PACKAGE_FAILED |
PUBLISH_FAILED | PIPELINE_CANCELLED | UNKNOWN_PIPELINE_FAILURE
```

현행 변환기는 gate/고유 batch 거부와 취소를 구분하지만 일반 기술 실패는 `UNKNOWN_PIPELINE_FAILURE`로 반환한다. schema에 코드가 있다는 이유로 단계별 상세 분류가 완료됐다고 가정하지 않는다.

### 활성화와 polling

| `activation.status` | 의미·현재 판정 |
| --- | --- |
| `pending` | 승격 전 또는 candidate 적재 대기 |
| `active` | serving version이 해당 candidate와 같음 |
| `loadFailed` | desired가 candidate인데 watcher 적재 실패, 직전 모델 유지 |
| `superseded` | 성공 run의 candidate와 현재 desired가 다름; 더 최신 version인지 순서를 검증하지 않음 |
| `notApplicable` | 거부·실패·취소 |

`202` 수신 10초 뒤 조회하고 15→30→최대 60초로 backoff한다. `stale=true`/unknown/HTTP 장애는 완료로 단정하지 않는다. 실패·거부·취소는 polling을 끝낸다. 성공이어도 activation pending이면 반영 확인을 계속하고 active/superseded에서 종료한다. loadFailed는 운영 확인 대상으로 남긴다. **terminal 성공과 실제 serving 활성화는 별개**다.

권장 저장 필드는 batch/run/logical/candidate ID, pipeline status/stage, result outcome/failure code/retryable, activation status, submitted/completed/last-polled 시각이다.

## 5. `GET /router/model/current`

응답 schema는 `router-model-status-v1`이다. `source`는 `currentPointer | fixedPackage | localPath`, `status`는 `inSync | loading | loadFailed | static | unavailable`이다.

- `desired`: pointer의 model/logical version, trainingMode, manifestSha256, approvedAt; 관측되지 않았으면 null.
- `serving`: 실제 model/logical version과 loadedAt; 미적재면 null. logical version을 확인할 수 없으면 null.
- `previousKnownGood`: 이전 modelVersion 또는 null.
- `lastPointerCheckAt`, `lastSuccessfulSwapAt`, `lastLoadError`: 마지막 확인·교체·일반화된 오류 정보.

전체 예시는 [current-model fixture](examples/retraining-api-v1/current-model.example.json)에 있다. `unavailable`은 같은 schema로 HTTP 503, 그 밖에는 200이다. fixed/local 또는 watcher 비활성은 static이 될 수 있다. loadFailed에서도 직전 loaded classifier가 추론을 처리할 수 있다.

## 6. 개발자 version policy

Spring은 이 API를 호출하지 않는다. `LLM_ROUTER_ENABLE_DEVELOPER_API=true`와 별도 `LLM_ROUTER_DEVELOPER_API_TOKEN`이 필요하고 network 제한은 배포가 담당한다. 비활성·token 미설정은 403, 잘못된 credential은 401이다.

`GET /developer/retraining/version-policy`는 `schemaVersion=retraining-version-policy-v1`, mode, callsPerMajor, rebaseEveryMajorCycles, committedSuccessfulRuns, cyclePosition, current/nextLogicalModelVersion, revision, nullable pendingChange, updatedAt/updatedBy를 반환한다.

`PUT`은 다음 네 필드를 모두 요구한다.

```json
{
  "mode": "validatedUniqueBatch",
  "activation": "nextMajorBoundary",
  "expectedRevision": 4,
  "reason": "고유 batch 정책 검증 후 전환"
}
```

`mode`는 `successfulHttpRequest | validatedUniqueBatch`, activation은 정확히 `nextMajorBoundary`, expectedRevision은 0 이상 정수, reason은 1~500자다. revision 불일치는 `409 REVISION_CONFLICT`다. 변경은 다음 major 성공 경계에 예약하며 즉시 active mode를 바꾸거나 기존 version count·제출 run을 재해석하지 않는다. `updatedBy`/audit actor는 현재 `developer-api`로 기록되며 개인 사용자 인증 기능은 아니다.

기본 successfulHttpRequest에서는 성공 run 세 개가 한 cycle이며 `1.1→1.2→2.0`으로 진행한다. 각 cycle은 같은 `.0` anchor를 사용하고 5번째 major 성공은 pristine rebase다. event 수와 retry는 version 횟수가 아니다. replay·rehearsal·기준 checkpoint와 실험의 정확한 정의는 [ML 재학습 계약](../../LLM_Router/docs/ml/README.md#retraining)이 소유한다.

validatedUniqueBatch는 canonical batch digest를 이용하는 대안 정책이다. 전체 batch digest가 다르다는 사실은 모든 row가 신규라는 보장이 아니며, 적용 가능한 pipeline·control root가 필요하다. 실제 검증 한계는 [ML 구현 차이표](../../LLM_Router/docs/ml/README.md#implementation-gaps)를 따른다. 이 PUT으로 callsPerMajor나 rebase 주기를 수정하지 않는다.

## 7. 오류 처리표

| HTTP / code | 처리 |
| --- | --- |
| 400 `INVALID_MESSAGE` | message 또는 rerank 문자열 길이 수정 |
| 400 `TOO_MANY_CANDIDATES` | 설정 후보 상한에 맞춤 |
| 401 `UNAUTHORIZED` | 내부/developer credential 또는 GCS identity 점검 |
| 403 `FORBIDDEN` | developer 비활성 또는 Vertex/GCS IAM 점검 |
| 404 `RETRAINING_RUN_NOT_FOUND` | store에 해당 run ID를 가진 batch가 없음 |
| 409 `BATCH_ID_CONFLICT`, `GCS_OBJECT_CONFLICT` | 자동 변경/덮어쓰기 없이 원인 조사 |
| 409 `RETRAINING_IN_PROGRESS` | 같은 batch와 payload로 backoff |
| 409 `REVISION_CONFLICT` | 개발자 policy 재조회 후 의도 확인 |
| 422 `REQUEST_VALIDATION_FAILED`, `UNSUPPORTED_SCHEMA_VERSION` | DTO/schema 수정 |
| 422 `GCS_OBJECT_NOT_FOUND`, `MANIFEST_MISMATCH` | 내부 artifact 경로·정합성 점검 |
| 500 `EMBEDDING_INFERENCE_FAILED`, `ROUTER_INFERENCE_FAILED` | 운영 알림 및 Spring의 장애 fallback |
| 500 `RERANK_INFERENCE_FAILED` | cache miss 처리 |
| 500 `INTERNAL_SERVER_ERROR` | requestId로 운영 조사 |
| 502 `PIPELINE_SUBMISSION_FAILED` | 같은 batch로 제한 재시도 |
| 503 `EMBED_MODEL_NOT_READY`, `ROUTER_MODEL_NOT_READY` | Spring 장애 정책 적용 |
| 503 `RERANK_MODEL_NOT_READY` | cache miss 처리 |
| 503 `GCS_BACKEND_UNAVAILABLE`, `PIPELINE_BACKEND_UNAVAILABLE` | 같은 batch로 backoff |
| 503 `PIPELINE_STATUS_UNAVAILABLE` | 재학습 backend 미설정; polling 장애 처리 |

`/health`의 503은 HealthResponse, `/router/model/current`의 미적재 503은 CurrentModelResponse다. 일반 ErrorResponse와 구분한다.

<a id="implementation"></a>
## 8. 관측 artifact와 현행 구현 범위

[관측 reader](../src/llm_router_serving/retraining/observability.py)는 `<observability-root>/<batchId>/observability-v1/` 아래 다음 객체를 읽는다.

```text
metrics/feedback-prepare.json    metrics/feedback-merge.json
progress/training.json          metrics/training-final.json
metrics/evaluation.json         metrics/packaging.json
gate/gate-result.json           result/publish-result.json
```

입력 Parquet/manifest와 관측 경로는 분리한다. reader는 JSON object/batch identity/candidate identity를 검사하고, 없거나 잘못된 artifact는 null과 필요 시 warning으로 표시한다. 상태 API는 원 질문·개별 event 대신 집계를 제공한다. 과거 계획을 구현 완료로 오해하지 않도록 다음 범위를 구분한다.

| 항목 | 현재 source 동작 |
| --- | --- |
| Vertex lifecycle | 전체 terminal 상태 조회. queued/running/cancelling/paused는 adapter에서 active로 축약; task detail 조회 없음 |
| 상태 refresh 장애 | 저장된 job state를 이용해 응답하고 `stale=true`와 예외 종류 경고. 완전한 response snapshot cache나 10초 TTL은 없음 |
| GCS 일부 장애 | 해당 집계 null·warnings; 이것만으로 top-level stale이 true가 되지는 않음 |
| 성공 해석 | Vertex success를 바탕으로 promoted 반환. publish-result 존재를 API가 필수 조건으로 재검증하지 않음. serving 적용은 activation으로 확인 |
| 단계·heartbeat | artifact 존재로 단계 추정. heartbeat sequence 단조 검사·10분 경과 자동 warning은 없음 |
| 기술 실패 | 일반 실패를 UNKNOWN_PIPELINE_FAILURE로 표현; 모든 세부 enum 분류 구현은 아님 |
| 검증 오류 로그 | `main.py`가 `exc.errors()`를 기록하므로 입력값이 로그에 포함될 가능성이 있음. 전체 입력 redaction 완료를 주장하지 않음 |

이 표는 API 소비자가 의존할 수 있는 범위를 명시하며 기능 구현 완료 선언이 아니다. 남은 연구·실행 변경은 [통합 ML 실행 차이표](../../LLM_Router/docs/ml/README.md#implementation-gaps)에서 관리한다.

Spring/FastAPI 계약 검증은 field/UTC/strict boolean, inline 1건·빈 배열, 미지 legacy 필드 거부, 재시도 멱등성과 충돌, rerank 순서·fallback, terminal/activation 분리, 새 모델 적재 실패 시 기존 추론 유지를 포함한다. JSON fixture는 필드 예시이며 실배포·GPU E2E 성공을 입증하는 로그가 아니다.

<a id="routing-policy-v2"></a>
## 9. 명시적 서비스 정책을 사용하는 routing v2 계약

이 문서는 구현된 `model-package-v2` 적재와 명시적 정책의 동작을 설명한다.
R0·R1 교체는 기존 `POST /router/classify`의 요청과 네 응답 필드를 유지한다. 별도의
`POST /router/classify/v2`는 정책 메타데이터를 조회하는 기존 선택적 경로로 유지한다.
패키지의 v2 schema가 서비스 클라이언트의 v2 endpoint 전환을 요구하지 않는다.
정책은 R0/R1/R2의 점수 의미와 서비스 선택 기준을 명시하며 기존 3단계 라우팅을 유지한다.

이 구현은 로컬 검증을 마친 지원 코드다. 실제 서비스 후보 모델, 비용, λ를 확정하거나
GCS의 `CURRENT.json`을 변경하지 않았으며 운영 배포도 수행하지 않았다. 학습 결과나
모델 파일의 GCS 게시만으로 서비스 정책 승인·모델 활성화가 이루어지지는 않는다.

### 9.1. 요청·버전 호환성

두 endpoint 모두 JSON body `{"message":"질문"}`를 받는다. 기존 내부 인증,
`X-Request-ID`, 요청 길이 제한과 공통 오류 형식을 사용한다. 알 수 없는 요청 필드는
거부한다.

| 적재된 모델 | `/router/classify` | `/router/classify/v2` |
| --- | --- | --- |
| 명시적 v2 정책이 없는 legacy ordinal 모델 | 기존 `tier`, `model`, `confidence`, `modelVersion` 응답 | `503 ROUTER_POLICY_NOT_CONFIGURED` |
| 검증된 명시적 정책이 있는 R0 ordinal·R1 pointwise 모델 | 정책 선택을 투영한 기존 네 필드 응답 | 메타데이터를 포함한 v2 응답 |
| 검증된 명시적 정책이 있는 R2 pairwise 모델 | `409 ROUTER_API_VERSION_REQUIRED` | nullable confidence를 포함한 v2 응답 |
| 모델 미적재 | `503 ROUTER_MODEL_NOT_READY` | `503 ROUTER_MODEL_NOT_READY` |

정책이 있으면 두 endpoint 모두 같은 `route_scores()` 경로로 후보를 선택한다.
R0에서 비용 정책이 argmax와 다른 후보를 고르더라도 기존 endpoint가 그 정책을
우회하지 않는다. R1 confidence는 선택된 후보의 sigmoid 점수이며 R0 softmax로
바꾸지 않는다. R1/R2 checkpoint는 legacy v1 패키지나 정책 없는 classifier로
적재할 수 없다. R2는 기존 API의 필수 숫자 confidence를 제공할 수 없어 v2 경계를 유지한다.

구조 보존 입력이 정규화 후 비어 있으면 `400 INVALID_MESSAGE`, 추론 출력이 유효하지
않으면 `500 ROUTER_INFERENCE_FAILED`다. 기존 endpoint에 objective나 정책 정보를
요청 필드로 추가하지 않는다. R0·R1 선택은 배포 패키지와 내부 설정으로 결정한다.

### 9.2. 공통 3단계와 목표 함수별 점수

| action | 응답 tier | 응답 model: 논리 힌트 | 실제 호출 모델 |
| --- | --- | --- | --- |
| `low` | `easy` | `haiku` | `candidateModelIds.low` |
| `medium` | `medium` | `sonnet` | `candidateModelIds.medium` |
| `high` | `hard` | `opus` | `candidateModelIds.high` |

`model`은 provider의 실제 모델 ID가 아니다. 정책에는 세 후보의 서로 다른 실제
`model_id`가 필요하며, 응답의 `candidateModelIds`는 선택되지 않은 후보까지 세 ID를
모두 전달한다. Spring은 이를 자신이 실제 호출하는 `llm.models.easy/medium/hard`와
완전히 일치하는지 검사한다. `haiku`, `sonnet`, `opus` 같은 논리 힌트만 `model_id`로
기재한 정책은 거부한다. 로더의 ID 검증은 형식·명시성 검사이며 provider의 모델 존재나
현재 호출 가능성을 원격으로 확인하는 기능은 아니다.

| 비교군 / objective | checkpoint 출력 순서 | actionScores 생성 | scoreSemantics | confidence |
| --- | --- | --- | --- | --- |
| R0 / `ordinal` | `easy, medium, hard` | 3개 logit의 softmax | `difficulty_class_probabilities` | 선택된 action의 비용 적용 전 softmax 값 |
| R1 / `pointwise` | `low, medium, high` | 각 logit의 독립 sigmoid | `independent_success_probabilities` | 선택된 action의 비용 적용 전 sigmoid 값 |
| R2 / `pairwise` | `low, medium, high` | 원래 상대 품질 점수 | `relative_quality_scores` | 반드시 `null` |

R0 확률 합은 1이다. R1은 후보별 성공 추정치이므로 합이 1일 필요가 없으며, 학습·평가와
같이 logit을 `[-30, 30]`으로 제한한 뒤 sigmoid를 적용한다. R2 점수는 유한수여야 하지만
`[0, 1]`에 제한되지 않는다. R2에 softmax를 씌워 성공 확률이나 confidence로 표시하지 않는다.
이 점수들에 대한 추가 확률 보정은 구현되어 있지 않다.

모든 v2 목표 함수는 같은 선택 규칙을 사용한다.

```text
utility[action] = actionScores[action] - lambda_cost * decision_cost[action]
선택 = utility가 가장 큰 action
정확히 동률이면 low → medium → high 순서로 선택
```

따라서 비용 때문에 최대 점수를 가진 후보와 실제 선택 후보가 다를 수 있다. R0/R1의
`confidence`는 최대 점수나 비용을 뺀 utility가 아니라 **선택된 후보의 원래 점수**다.
`utilities`는 내부 계산 결과이며 현재 HTTP 응답에는 포함되지 않는다.

### 9.3. v2 응답

필수 메타데이터는 `schemaVersion`, `objective`, `policyVersion`, `policySha256`,
`candidatePoolVersion`, `candidateModelIds`, `action`, `scoreSemantics`, `actionScores`다.
기존 `tier`, `model`, `modelVersion`을 유지하고 `confidence`만 목표 함수에 따라 nullable로
정의한다. `modelVersion`은 물리적 checkpoint/package 버전이고 `policyVersion`은 정책 버전이다.

다음은 **SYNTHETIC 테스트 fixture 예시**다. 모든 provider ID, 모델·정책·pool 버전과
점수는 설명용이며 실험 결과·운영 설정이 아니다. 세 logit이 모두 0인 R1 fixture는 세
점수가 각각 0.5이고, fixture 비용 정책에 따라 `low`를 선택한다. 해시는 이 합성 fixture의
canonical policy 해시이며 실제 서비스 설정에 사용하면 안 된다.

```json
{
  "schemaVersion": "router-decision-v2",
  "tier": "easy",
  "model": "haiku",
  "modelVersion": "router-synthetic-policy-v2",
  "objective": "pointwise",
  "policyVersion": "fixture-policy-v2",
  "policySha256": "be1db3bffe07d6acd2ad2da5896edfaa878144308104d1c7995dd7461d432e7d",
  "candidatePoolVersion": "fixture-service-pool-v1",
  "candidateModelIds": {
    "low": "fixture-provider/small-model-v1",
    "medium": "fixture-provider/mid-model-v1",
    "high": "fixture-provider/large-model-v1"
  },
  "action": "low",
  "scoreSemantics": "independent_success_probabilities",
  "actionScores": {"low": 0.5, "medium": 0.5, "high": 0.5},
  "confidence": 0.5
}
```

### 9.4. 패키지와 명시적 서비스 정책

`model-package-v2` manifest는 `objective`, `input_contract`, `tokenizer_max_length`,
`routing_policy_file: "routing_policy.json"`, `routing_policy_sha256`와 파일별 SHA-256을
포함한다. `config.json`의 `router_objective`, 실제 `id2label`·`label2id`,
`training_config.json`의 목표 함수 및 길이 제한을 정책과 대조한다. R0의 training config에서
목표 함수가 생략된 경우만 기존 ordinal로 해석하며, v2 checkpoint config의
`router_objective`는 명시되어 있어야 한다.

`routing_policy.json`은 정확히 정해진 필드를 사용한다. 누락·추가 필드와 NaN/Inf를 거부한다.

| 영역 | 필수 내용 |
| --- | --- |
| 식별 | `schema_version: routing-policy-v2`, `scope: service-explicit`, `policy_version`, `objective`, `policy_sha256` |
| 입력·출력 | `input_contract: structure-preserving-v2`, `max_length: 512` 또는 새 길이 계약의 `8192`, `action_order: [low, medium, high]`, 목표 함수에 맞는 `score_semantics` |
| 선택 | 유한한 0 이상의 `lambda_cost` |
| 후보 pool | `version`, `price_version`, `decision_cost_version`, `decision_cost_basis: pre-request-fixed`, `cost_unit`, 순서가 고정된 세 `candidates` |
| 각 후보 | `action`, `tier_projection`, 논리 `model`, 실제 `model_id`, 유한한 0 이상의 `decision_cost` |

비용 단위는 `normalized-0-1` 또는 `usd-per-request`다. 비용은 low≤medium≤high여야 하고,
정규화 단위에서는 low=0, high=1이어야 한다. λ는 그 비용 단위에 대응하는 값이어야 한다.
현재 요청의 응답 토큰 수나 사후 청구 금액으로 선택 비용을 바꾸지 않는다.

RouterBench와 xRouteBench의 후보 pool과 source별 λ는 그 평가 범위에 묶여 있다.
서비스 요청에서 benchmark source를 추측하거나 source-local 정책을 서비스 정책으로
자동 변환·평균하는 기능은 없다. 공개 데이터의 `low/medium/high` 출력 슬롯을 실제
Haiku/Sonnet/Opus provider ID에 투영하는 것이 타당한지도 별도의 서비스 검증 대상이다.
필드가 완비된 합성 정책이 로드된다는 사실만으로 이 의미적 타당성이 검증되지는 않는다.

#### 세 종류의 해시

| 위치 | 해시 대상 | 용도 |
| --- | --- | --- |
| `CURRENT.json.manifest_sha256` | `model_manifest.json` 실제 바이트 | 원하는 승인 패키지를 고정 |
| manifest의 `routing_policy_sha256` 및 `files["routing_policy.json"]` | `routing_policy.json` 실제 바이트 | 파일 무결성 검증; 두 값은 같아야 함 |
| policy 내부 `policy_sha256` 및 응답 `policySha256` | 내부 digest 필드를 뺀 검증·정규화된 canonical policy JSON | 정책 의미와 클라이언트 pin을 고정 |

Canonical JSON은 key를 정렬하고 공백 없이 직렬화한다. 유효한 숫자를 float로 정규화하고
`-0.0`을 `0.0`으로 통일한다. 파일의 들여쓰기만 바꿔도 파일 해시는 달라질 수 있지만
canonical 해시는 유지될 수 있다. 두 해시를 서로 대신 사용하지 않는다. 정책 파일의
파일 해시를 먼저 확인하고 내부 canonical digest를 별도로 검증한다.

`CURRENT.json`의 schema는 기존 `router-current-model-v1`을 유지한다. pointer의
`id2label`은 `easy/medium/hard` 서비스 alias이며, R1/R2 패키지·checkpoint의 실제 출력
label인 `low/medium/high`와 구별해 검증한다. pointer의 형식을 새 v2 응답으로 바꾸지 않는다.

### 9.5. 입력 정규화와 학습·추론 길이 분리

v2 classifier와 `input_contract: structure-preserving-v2`를 명시한 v1 classifier는
tokenizer 전에 `clean_text`를 적용한다.
NFKC 정규화, CRLF/CR→LF 변환, newline·tab을 제외한 Unicode category C 문자 제거,
각 줄 끝 공백 제거, 연속 빈 줄 축약과 바깥 newline 제거를 수행한다. 학습과 동일하게
첫 줄의 특정 객관식 안내 문장과 마지막 줄의 특정 단일 선택지 출력 안내도 제거한다.
정확한 규칙은 [structured_text.py](../src/llm_router_serving/retraining/structured_text.py)에 있다.

줄 앞의 indentation, 줄 경계, 연산자와 대소문자는 보존한다. 일반 `strip()`으로 입력
본문의 indentation을 지우지 않는다. NFKC가 정규화하는 전각 문자는 바뀔 수 있다.
정규화 결과가 공백뿐인지는 별도 검사해 거부한다.

R0·R1 신규 배포 프로필의 추론 목표는 8,192토큰이다. 추가 학습 없이 길이를 확장하는
패키지는 아래 계약을 사용한다. 학습 길이는 보존한 실제 기록과 비교하며, 추론 길이를
따라서 바꾸지 않는다.

| 위치 | 신규 8,192토큰 패키지 |
| --- | --- |
| manifest | `length_contract_version: train-inference-lengths-v1`, `training_max_length: 512`, `inference_max_length: 8192`, `tokenizer_max_length: 8192` |
| `training_config.json` | 기존 학습의 `max_length: 512` 유지; manifest의 `training_max_length`와 정확히 일치 |
| v2 policy | 같은 `length_contract_version`과 `max_length: 8192`; policy에는 학습 길이를 넣지 않음 |
| checkpoint config | 양의 정수 `max_position_embeddings`; 학습·추론 길이가 이 한도를 넘으면 거부 |

길이 필드에 boolean·문자열·실수를 허용하지 않는다. 새 계약은 추론 길이 512도
명시할 수 있어 같은 schema로 rollback 패키지를 검증할 수 있다. marker가 없는 기존
v2 policy/package는 512만 허용한다. marker 또는 필드를 일부만 추가·제거하거나
manifest와 policy marker·추론 길이가 다르면 거부한다. 새로운 계약의 policy SHA는
기존 512 policy와 다르므로 Spring의 pin도 함께 갱신해야 한다.

시작 적재와 hot reload 모두 검증한 추론 길이로
`truncation=True, max_length=<inference_max_length>`를 호출한다.
`router_max_length` 환경 설정은 명시적 패키지 길이를 덮어쓰지 않는다. 입력 계약이
없는 legacy v1은 원래 raw 입력과 기존 길이 설정을 유지한다. 512 길이의 기존 구조
보존 v1 패키지도 계속 읽을 수 있다.

이 계약 지원만으로 8,192의 운영 승격이 완료되지는 않는다. 실제 서비스 후보에 대한
동일 추론 길이 gate, 운영 VM의 지연·메모리 검증과 immutable 새 모델 버전·manifest
생성이 필요하다. 추론 길이를 바꾸며 기존 `CURRENT.json`이나 패키지를 덮어쓰지 않는다.
검증: [길이 계약과 hot reload 테스트](../tests/test_length_contract.py).

### 9.6. 적재·교체와 롤백

패키지 파일 checksum, 승인 상태(기본 요구), 목표 함수·label·정책·입력 계약을 검증하고
모델을 로드한 다음에 활성 classifier를 원자적으로 교체한다. 진행 중인 요청은 시작할 때
얻은 이전 classifier 참조를 유지한다. 다운로드·검증·로드 실패 시 기존 정상 모델을
계속 제공한다. 이 동작은 기존 `ModelManager`의 load-before-swap 경계를 유지한다.

pointer watcher는 현재 적재된 동일 물리적 `model_version`에 다른 manifest digest가
나타나면 거부한다. 캐시 재사용 때도 pointer가 지정한 manifest digest를 다시 검사한다.
가중치 또는 정책을 바꿀 때 같은 물리적 버전을 덮어쓰는 방식은 지원하지 않는다.
과거 승인 버전으로의 rollback도 그 버전의 원래 manifest와 정책을 사용한다.

Spring의 R0·R1 공통 서비스 경로는 기존 `/router/classify`를 사용한다. 이 네 필드
응답에는 정책 hash·후보 ID가 없으므로 배포 시 FastAPI의 policy/package와 Spring의
실제 모델 mapping이 같은 release 설정을 사용하는지 별도로 확인해야 한다.
선택적 v2 클라이언트는 `llm.router.classify-contract=v2`를 명시했을 때 다음 설정과
응답을 함께 검사한다.

- `llm.router.expected-policy-sha256`: 파일 해시가 아닌 canonical policy 해시
- `llm.router.candidate-pool-version`: 응답의 pool 버전
- `llm.models.easy/medium/hard`: 응답의 세 실제 후보 ID 전체
- action↔tier↔논리 model, 목표 함수↔score semantics, 점수·confidence 계약

정책을 바꾼 승격이나 rollback 시 FastAPI package와 Spring의 pool·실제 모델 mapping을
함께 맞춰야 한다. v2 클라이언트에서는 policy pin도 맞춘다. 단일 pin을 사용하는 v2
클라이언트는 전환 중 두 정책을 동시에 허용하거나 설정을 자동 갱신하지 않는다.
기존 endpoint를 사용하는 R0·R1 전환에는 클라이언트 요청·응답 변경이 없다.
HTTP 오류나 계약 불일치에 대한 자동 endpoint 전환은 없다.

### 9.7. 운영 활성화 전에 남은 조건

이 구현은 목표 함수별 추론 경계를 지원한다. 다음 작업의 완료를 뜻하지 않는다.

- 실제 서비스 후보 ID와 pool·가격·고정 비용 버전, 비용 단위, λ의 확정 및 서비스 데이터 검증
- 서비스 평가에서 승인 기준을 충족한 immutable package 생성·승격과 클라이언트 pin 동기화
- R1의 후보별 성공 target 또는 R2의 비교 가능한 품질·선호 target을 만들 feedback 계약과 loss 정의
- 선택한 목표 함수에 맞는 replay, 평가·승격 gate와 재학습 pipeline 검증

현재 feedback의 `initialLabel/finalLabel/userSatisfied`만으로는 공개 데이터와 동일한
세 후보별 성공 target 또는 완전한 pairwise 품질 target을 자동 재현할 수 없다. 관측되지
않은 후보의 outcome을 생성하거나 R1/R2 checkpoint에 legacy CE loss를 적용하면 안 된다.
D_dev의 비교 승자 선정, 모델 파일 보관, sealed test 진단, 서비스 승인·배포는 서로 다른
상태다. 이 문서와 v2 지원 코드에는 정책값 자동 선택, `CURRENT.json` 수정 또는 운영
활성화 기능이 추가되지 않았다.

구현 근거: [policy engine](../src/llm_router_serving/routing_policy.py),
[package loader](../src/llm_router_serving/model_source.py),
[classifier](../src/llm_router_serving/inference.py),
[HTTP endpoint](../src/llm_router_serving/main.py),
[v2 계약 테스트](../tests/test_routing_policy_v2.py).
