# 재학습 진행상황 및 데이터 필터링 관측 계획

상태: API·pipeline·GCS 관측 artifact 구현 및 배포 완료, 실제 GPU run 검증은 별도 수행

대상 schema: `retraining-status-v1`

대상 시스템: Spring Server, `LLM_Router-serving`, Vertex AI Pipelines,
`LLM_Router/ml`

## 1. 목표

Spring Server가 `POST /escalate`의 `202 Accepted` 이후 원할 때 다음 내용을 FastAPI를 통해
조회할 수 있게 한다.

- Pipeline이 대기·전처리·학습·평가·gate 중 어디까지 진행됐는가
- 현재 단계가 실행 중인지, 성공했는지, 실패했는지
- ModernBERT 학습의 현재 optimizer step과 전체 step
- 입력 batch 중 행 단위 조건으로 제외된 event 수와 사유
- 중복 질문 병합 후 실제 canonical 학습 행 수
- 기존 RouterBench 데이터와 병합할 때 추가로 제외·교체된 수
- Pipeline의 최종 결과가 candidate 게시, gate 거부, 기술적 실패 중 무엇인가
- gate 통과 모델이 `CURRENT.json`에 승격됐는가
- 승격된 model이 FastAPI 메모리에 실제로 적재됐는가

질문 원문, event ID, request ID와 개인정보는 상태 API에 노출하지 않는다. 숫자와 집계만
반환한다.

## 2. 현재 구현에서 부족한 점

| 영역 | 현재 상태 | 문제 |
| --- | --- | --- |
| Pipeline 조회 | FastAPI가 다음 `/escalate` 시점에만 Vertex 전체 상태를 확인하며 성공·실패·취소는 구분 | 실시간 조회, 단계·gate 거부·오류 원인을 알 수 없음 |
| 단계 조회 | Vertex task detail을 읽지 않음 | 전처리·학습·평가 중 어느 단계인지 모름 |
| 상태 API | `POST /escalate`만 있고 조회 endpoint가 없음 | Spring이 `202` 이후 결과를 알 수 없음 |
| 로컬 저장 | SQLite에 제출 및 축약 terminal 상태와 version 계획을 기록 | 단계, 오류, metric, activation snapshot이 없음 |
| feedback prepare | `metrics.json`에 필터링 수치가 이미 생성됨 | Pipeline artifact 안에만 있어 Spring이 접근할 수 없음 |
| feedback merge | `metrics.json`에 병합 제외·교체 수치가 이미 생성됨 | 위와 동일 |
| GPU 학습 | 최종 `metrics.json`만 생성 | 실행 중 step/epoch heartbeat가 없음 |
| deploy gate | `gate_result.json` 작성 후 실패 exit | gate 거부와 시스템 오류를 외부에서 안정적으로 구분하기 어려움 |

따라서 필터링 로직을 다시 구현하지 않는다. 기존 `feedback prepare`와 `merge`가 계산하는
metric을 안정적인 GCS run 경로에 공개하고, FastAPI가 Vertex 상태와 함께 읽어 API 형태로
조합한다.

## 3. 전체 구조

```text
Spring Server
  └─ GET /retraining-runs/{batchId}
       └─ FastAPI
            ├─ SQLite: batch ↔ pipelineRunId, 마지막 정상 snapshot
            ├─ Vertex PipelineJob: 전체 상태, task별 상태·시각·오류
            └─ GCS observability prefix
                 ├─ feedback prepare/merge 집계
                 ├─ ModernBERT step heartbeat
                 ├─ 평가 metric
                 ├─ deploy gate 결과
                 └─ candidate 게시 결과
```

Spring은 Vertex AI 또는 GCS에 직접 접근하지 않는다. GCP 권한과 내부 artifact 구조는
FastAPI 뒤에 숨기고, FastAPI가 안정된 camelCase API 계약으로 변환한다.

## 4. 상태 조회 API 계약

필드와 enum의 단일 기준은 [`api-contract.md`](api-contract.md)의 7장이다. 이 문서는 그
계약을 만들기 위한 Vertex/GCS/FastAPI 구현 방법을 설명한다.

### 4-1. Endpoint

```http
GET /retraining-runs/{batchId}
Authorization: Bearer <internal-token>
X-Request-ID: <request-id>
```

`pipelineRunId`는 `/`가 포함된 전체 Vertex resource name이므로 URL path key로 쓰기
불편하다. 이미 영문·숫자·점·밑줄·하이픈으로 제한된 `batchId`를 조회 key로 사용하고,
응답에 `pipelineRunId`를 포함한다.

`POST /escalate`의 목표 `escalation-acceptance-v1` 응답에는 `statusUrl`을 필수로 추가한다.
FastAPI 배포와 Spring contract test를 함께 전환한다.

```json
{
  "schemaVersion": "escalation-acceptance-v1",
  "status": "accepted",
  "batchId": "feedback-20260904-0001",
  "pipelineRunId": "projects/123/locations/us-west1/pipelineJobs/retrain-feedback-20260904-0001-a1b2c3d4",
  "logicalModelVersion": "2.1",
  "candidateModelVersion": "router-modernbert-feedback-v2.1-20260904-a1b2c3d4-21c2f1b",
  "trainingMode": "minor_fine_tune",
  "versionPolicyMode": "successfulHttpRequest",
  "statusUrl": "/retraining-runs/feedback-20260904-0001"
}
```

### 4-2. 상태 응답 예시

```json
{
  "schemaVersion": "retraining-status-v1",
  "batchId": "feedback-20260904-0001",
  "pipelineRunId": "projects/123/locations/us-west1/pipelineJobs/retrain-feedback-20260904-0001-a1b2c3d4",
  "status": "running",
  "stage": "training",
  "terminal": false,
  "version": {
    "logicalModelVersion": "2.1",
    "candidateModelVersion": "router-modernbert-feedback-v2.1-20260904-a1b2c3d4-21c2f1b",
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
    "heartbeatAt": "2026-09-04T07:12:30Z"
  },
  "stages": [
    {
      "name": "feedbackPreparation",
      "status": "succeeded",
      "startedAt": "2026-09-04T06:31:04Z",
      "endedAt": "2026-09-04T06:32:11Z"
    },
    {
      "name": "feedbackMerge",
      "status": "succeeded",
      "startedAt": "2026-09-04T06:32:13Z",
      "endedAt": "2026-09-04T06:33:02Z"
    },
    {
      "name": "training",
      "status": "running",
      "startedAt": "2026-09-04T06:34:20Z",
      "endedAt": null
    },
    {
      "name": "evaluation",
      "status": "pending",
      "startedAt": null,
      "endedAt": null
    },
    {
      "name": "packaging",
      "status": "pending",
      "startedAt": null,
      "endedAt": null
    },
    {
      "name": "gate",
      "status": "pending",
      "startedAt": null,
      "endedAt": null
    },
    {
      "name": "publishing",
      "status": "pending",
      "startedAt": null,
      "endedAt": null
    }
  ],
  "filtering": {
    "inputEventCount": 3,
    "acceptedEventCount": 932,
    "excludedEventCount": 68,
    "collapsedDuplicateEventCount": 57,
    "canonicalTrainingRowCount": 875,
    "exclusionDistribution": {
      "conflicting_final_labels": 31,
      "duplicate_event_id": 7,
      "pii_not_redacted": 12,
      "user_not_satisfied": 18
    },
    "labelDistribution": {
      "easy": 0,
      "medium": 511,
      "hard": 364
    }
  },
  "merge": {
    "feedbackInputRowCount": 875,
    "acceptedFeedbackRowCount": 830,
    "excludedFeedbackRowCount": 45,
    "replacedBaseRowCount": 23,
    "mergedTrainingRowCount": 19607,
    "decisionDistribution": {
      "append_new_question": 807,
      "excluded_base_holdout_overlap": 19,
      "excluded_same_label_already_in_base": 26,
      "replace_exact_base_train_rows": 23
    }
  },
  "result": null,
  "activation": {
    "status": "pending",
    "desiredModelVersion": null,
    "servingModelVersion": "router-modernbert-base-routerbench-20260814-45dcc6d",
    "checkedAt": "2026-09-04T07:12:32Z"
  },
  "submittedAt": "2026-09-04T06:30:02Z",
  "startedAt": "2026-09-04T06:30:45Z",
  "endedAt": null,
  "lastUpdatedAt": "2026-09-04T07:12:32Z",
  "stale": false,
  "warnings": []
}
```

아직 완료되지 않은 단계의 집계는 `null` 또는 해당 object 자체를 생략한다. `0`과 “아직
계산되지 않음”을 혼동하지 않는다.

### 4-3. Status와 stage

전체 `status`는 다음 값으로 제한한다.

```text
queued | running | cancelling | paused |
succeeded | rejected | failed | cancelled | unknown
```

- `succeeded`: deploy gate 통과, approved package 게시와 `CURRENT.json` 전환까지 성공.
  FastAPI 메모리 반영은 별도 `activation.status`로 표시
- `rejected`: 학습과 평가는 완료됐지만 deploy gate가 후보를 거부함
- `failed`: 인프라, 데이터 처리, 학습 또는 게시 과정의 기술적 실패
- `unknown`: Vertex가 새로운 상태를 반환하거나 상태 backend를 확인할 수 없음

Vertex PipelineJob 상태는 다음과 같이 대응한다.

| Vertex 상태 | API `status` |
| --- | --- |
| `PIPELINE_STATE_QUEUED`, `PIPELINE_STATE_PENDING` | `queued` |
| `PIPELINE_STATE_RUNNING` | `running` |
| `PIPELINE_STATE_CANCELLING` | `cancelling` |
| `PIPELINE_STATE_PAUSED` | `paused` |
| `PIPELINE_STATE_CANCELLED` | `cancelled` |
| `PIPELINE_STATE_FAILED` + 정상적인 `gatePassed=false` artifact | `rejected` |
| 그 밖의 `PIPELINE_STATE_FAILED` | `failed` |
| `PIPELINE_STATE_SUCCEEDED` + approve/pointer publish result 존재 | `succeeded` |
| `PIPELINE_STATE_SUCCEEDED`인데 approve/pointer result 없음 | `unknown` + warning |
| 미지정 또는 새로운 값 | `unknown` + warning |

현재 `stage`는 다음 값으로 제한한다.

```text
initializing | feedbackPreparation | feedbackMerge | training |
evaluation | packaging | gate | publishing | complete
```

각 stage의 `status`는
`pending | running | succeeded | failed | cancelled | skipped | unknown`이다.
`stages` 배열에는 일곱 개 실행 stage를 항상 같은 순서로 반환해 Spring이 누락과 대기를
구분할 수 있게 한다. `stage=complete`는 terminal 상태에서만 top-level에 사용한다.

Pipeline 전체 퍼센트는 반환하지 않는다. 전처리 1단계와 GPU 학습 1단계가 같은 시간이
걸리지 않으므로 “8단계 중 3단계 = 37.5%”는 잘못된 기대 시간을 만든다. 정확한 분모가 있는
GPU 학습 구간에만 `stageProgress`를 제공한다.

### 4-4. 오류 응답과 실패 표현

존재하는 run이 `failed`, `rejected`, `cancelled`인 것은 조회 API 오류가 아니다. HTTP 200과
해당 terminal `status`를 반환한다.

| HTTP | code | 조건 |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | 내부 인증 실패 |
| 404 | `RETRAINING_RUN_NOT_FOUND` | 해당 `batchId`가 FastAPI store에 없음 |
| 503 | `PIPELINE_STATUS_UNAVAILABLE` | Vertex/GCS 조회 실패이며 저장된 snapshot도 없음 |

Vertex가 일시적으로 응답하지 않아도 이전 정상 snapshot이 있으면 HTTP 200으로 반환하되
`stale=true`, `lastUpdatedAt`과 `warnings=["PIPELINE_STATUS_STALE"]`를 포함한다. Spring은
이를 pipeline 실패로 바꾸지 않고 다음 polling에서 다시 확인한다.

오류 메시지는 task name과 안정된 error code만 노출한다. stack trace, GCP credential,
질문 원문과 내부 환경변수는 반환하지 않는다.

## 5. 필터링 수치의 의미

한 개의 `filteredCount`만 반환하면 어느 단계에서 왜 감소했는지 알 수 없다. 다음 세 구간을
분리한다.

### 5-1. 행 단위 제외

기존 feedback prepare metric을 그대로 사용한다.

```text
inputEventCount
= acceptedEventCount + excludedEventCount
```

`exclusionDistribution`에는 현재 전처리의 다음 사유가 들어갈 수 있다.

```text
missing_event_or_request_id, duplicate_event_id, empty_question,
invalid_label, missing_model, missing_router_model_version,
invalid_user_satisfied, user_not_satisfied,
invalid_pii_redacted, pii_not_redacted,
invalid_escalation_count, insufficient_escalation,
invalid_created_at, not_strictly_escalated,
conflicting_final_labels
```

### 5-2. 정규화 중복 병합

행 조건을 통과한 동일 질문들은 하나의 canonical training row로 합쳐진다.

```text
collapsedDuplicateEventCount
= acceptedEventCount - canonicalTrainingRowCount
```

이는 품질 오류가 아니라 반복 feedback을 sample weight로 반영하는 정상 처리다.

### 5-3. 기존 학습 데이터와 병합

`feedback merge` 단계에서 다음 값을 별도로 제공한다.

```text
feedbackInputRowCount
= acceptedFeedbackRowCount + excludedFeedbackRowCount
```

- `excluded_base_holdout_overlap`: 기존 validation/test와 겹쳐 feedback을 제외
- `excluded_same_label_already_in_base`: 동일 질문·동일 label이 이미 있어 feedback을 제외
- `replace_exact_base_train_rows`: 기존 train 질문의 label을 feedback으로 교체한 정상 채택
- `append_new_question`: 새로운 질문을 train에 추가한 정상 채택

Spring 화면에서 “최종 재학습에 반영된 feedback 수”는 `acceptedFeedbackRowCount`를 사용한다.
`mergedTrainingRowCount`는 RouterBench base까지 포함한 전체 학습 데이터 행 수다.

## 6. GCS 관측 artifact 계약

입력 feedback batch는 immutable이므로 그 경로에 실행 상태를 덧쓰지 않는다. 별도 root를
설정한다.

```text
gs://<bucket>/retraining-runs/<batchId>/observability-v1/
├── metrics/feedback-prepare.json
├── metrics/feedback-merge.json
├── progress/training.json
├── metrics/training-final.json
├── metrics/evaluation.json
├── gate/gate-result.json
└── result/publish-result.json
```

FastAPI 설정에 `LLM_ROUTER_RETRAINING_OBSERVABILITY_ROOT_URI`를 추가하고 `/escalate` 제출 시
pipeline parameter로 다음 값을 전달한다.

```text
batch_id
observability_root_uri
```

규칙:

- 완료 metric과 gate/publish result는 한 번만 쓰는 immutable object로 게시한다.
- `progress/training.json`만 실행 중 갱신 가능한 heartbeat object다.
- 모든 object에 `schema_version`, `batch_id`, `candidate_model_version`, `updated_at`을 넣는다.
- GCS 단일 object upload의 atomic visibility를 이용해 FastAPI가 부분 JSON을 읽지 않게
  한다. Spring은 이 artifact를 직접 읽지 않고 상태 API만 호출한다.
- progress에는 증가하는 `sequence`와 `optimizer_step`을 넣고 FastAPI는 이전보다 작은 값을
  무시한다.
- 관측 artifact에는 질문, event ID와 request ID를 넣지 않는다.

## 7. ML pipeline 변경 계획

### 7-1. Feedback prepare/merge

현재 다음 metric은 이미 계산된다.

- prepare: 입력·채택·제외 event 수, training row 수, 제외 사유와 label 분포
- merge: 입력·채택·제외 feedback row 수, base 교체 수, 최종 train 수와 decision 분포

변경 사항:

1. `feedback_prepare_component`와 `feedback_merge_component`에 observability URI input을 추가한다.
2. 각 CLI가 기존 `metrics.json`을 쓴 직후 같은 내용을 versioned status envelope에 넣어
   결정적 GCS URI에 게시한다.
3. prepare metric 게시 실패는 GPU 학습 전에 pipeline을 실패시킨다. 필터 결과를 잃은 채
   비용이 큰 학습을 시작하지 않기 위함이다.
4. 기존 KFP artifact 안의 `metrics.json`과 GCS 공개본의 checksum 일치를 test한다.

### 7-2. ModernBERT 학습 heartbeat

공통 학습 코어에 GCS 의존성을 넣지 않는다.

1. `run_training()`에 선택적인 `on_progress(TrainingProgressState)` callback을 추가한다.
2. callback state에는 epoch, optimizer step, total steps, elapsed seconds와 최신 구간 loss를
   넣는다.
3. Vertex용 CLI가 callback을 받아 `progress/training.json`을 갱신한다.
4. 첫 학습 step 전 `0 / totalSteps`, 이후 30초 또는 25 optimizer step 중 더 늦게 충족되는
   시점마다 갱신하고 epoch 종료 및 학습 종료 시 반드시 갱신한다.
5. heartbeat 게시 일시 실패는 경고를 남기고 학습을 계속한다. Vertex task 상태는 계속
   조회할 수 있고, 관측 장애 때문에 이미 사용 중인 GPU 학습을 버리지 않는다.
6. Vertex가 `running`인데 heartbeat가 10분 이상 오래되면 FastAPI는 실패로 단정하지 않고
   `TRAINING_HEARTBEAT_STALE` warning을 반환한다.

### 7-3. 평가, gate, 게시

- RouterArena의 기존 `metrics.json`을 `metrics/evaluation.json`에 복사한다.
- deploy gate는 `passed=false`일 때 exit code 2로 task를 실패시키기 전에
  `gate/gate-result.json`을 GCS에 반드시 게시한다.
- FastAPI는 Vertex PipelineJob이 failed여도 gate result가 정상이고 `passed=false`이면
  기술적 `failed`가 아니라 비즈니스 결과인 `rejected`로 표시한다.
- candidate 자동 승인과 `CURRENT.json` 갱신 성공 후 `result/publish-result.json`을
  게시한다. 이 object가 있을 때만 top-level `status=succeeded`를 반환한다. 내부 GCS URI는
  Spring API에 노출하지 않는다.

## 8. FastAPI 변경 계획

### 8-1. Vertex 상세 조회

현재 `PipelineClient.state()`는 모든 terminal 상태를 하나로 합친다. 이를 다음 정보가 있는
`describe(run_id)` snapshot으로 확장한다.

- PipelineJob의 원래 state, create/start/end/update time과 sanitized error
- `job_detail.task_details`의 task name, state, start/end time
- Custom Training task가 가리키는 CustomJob resource name

task name을 API에 그대로 노출하지 않고 versioned mapping으로
`feedbackPreparation`~`publishing` stage에 대응시킨다. 새 pipeline template에서 알 수 없는
task가 생기면 `stage=initializing` 또는 `unknown`으로 안전하게 처리하고 warning을 남긴다.

### 8-2. 집계 service

`RetrainingStatusService`를 추가한다.

1. SQLite에서 `batchId`와 `pipelineRunId`를 찾는다.
2. Vertex에서 PipelineJob과 task detail을 읽는다.
3. 해당 batch의 GCS observability object 중 존재하는 것만 읽는다.
4. 수치 불변식과 `batch_id` 일치를 검증한다.
5. 내부 snake_case metric을 API camelCase response로 변환한다.
6. 정상 snapshot을 SQLite에 저장하고 응답한다.

Vertex 호출 폭주를 막기 위해 10초 TTL cache를 둔다. 서로 다른 Spring instance가 동시에
조회해도 TTL 안에서는 같은 snapshot을 반환한다.

### 8-3. SQLite 변경

기존 `retraining_jobs` record에 migration을 적용해 최소한 다음을 보존한다.

```text
candidate_model_version
observability_uri
pipeline_status
current_stage
status_snapshot_json
last_synced_at
started_at
ended_at
```

기존 database도 기동할 수 있도록 destructive migration이 아닌 `ALTER TABLE ADD COLUMN`
방식과 migration test를 사용한다. FastAPI 재시작 뒤에도 마지막 상태와 metric을 조회할 수
있어야 한다.

### 8-4. Pydantic/API model

`RetrainingStatusResponse`, `StageStatus`, `StageProgress`, `FilteringSummary`, `MergeSummary`,
`RetrainingResult`를 `extra="forbid"` model로 정의한다. API response에는 GCS 내부 manifest
전체나 raw Vertex protobuf를 그대로 반환하지 않는다.

## 9. Spring 연동 계획

1. `/escalate`에서 받은 `batchId`, `pipelineRunId`, `statusUrl`을 저장한다.
2. `202` 직후 한 번 조회하고 terminal 상태 전까지 30초 간격으로 polling한다.
3. 장시간 GPU 학습 중에는 60~120초까지 backoff해도 된다.
4. `queued`, `running`, `cancelling`, `paused`, `unknown`은 계속 polling한다.
5. `rejected`, `failed`, `cancelled`에서 polling을 종료한다.
6. HTTP timeout이나 `503`은 pipeline 실패로 저장하지 않고 재시도한다.
7. `stale=true` 응답은 마지막 관측값으로 화면에 표시하되 “최종 상태”로 사용하지 않는다.
8. `succeeded`이면서 `activation=pending`이면 model 반영까지 계속 polling하고,
   `active` 또는 `superseded`에서 완료한다. `loadFailed`는 운영 경보를 만든다.
9. 사용자가 보는 화면에는 현재 단계, 학습 step 퍼센트, 필터링 요약과 최종 결과를 표시한다.

MVP에서는 webhook을 추가하지 않는다. 내부 callback 인증, 재시도, 중복 전송과 SSRF 방어가
필요해 범위가 커진다. 먼저 polling 계약을 안정화하고, 이후 실시간 알림이 필요하면 Pub/Sub
또는 서명된 callback을 별도 version으로 설계한다.

## 10. 구현 순서

### Phase 1 — 계약 및 fixture

- `retraining-status-v1` JSON fixture와 Pydantic response model 작성
- Vertex state와 task name → API status/stage mapping 표 고정
- Spring/FastAPI가 공유할 contract test fixture 작성

완료 조건: running, succeeded, rejected, failed, cancelled 예시를 양쪽에서 같은 값으로 읽음.

### Phase 2 — 필터링 metric 공개

- observability root와 pipeline parameter 추가
- prepare/merge metric을 결정적 GCS URI에 게시
- pipeline IR 재컴파일 및 artifact checksum 검증

완료 조건: 여러 크기의 합성 batch에서 제외 사유, 중복 병합 수, merge 제외 수가 GCS JSON과 기존
`metrics.json`에서 정확히 같음.

### Phase 3 — 학습 heartbeat

- 공통 코어 progress callback과 Vertex CLI GCS reporter 구현
- step throttle, 단조 증가, stale heartbeat test 추가

완료 조건: bounded GPU/CPU smoke에서 step이 0부터 total까지 증가하고 마지막 percent가
100.0이며 학습 결과가 callback 미사용 실행과 동일함.

### Phase 4 — FastAPI status endpoint

- Vertex `describe`, GCS metric reader, SQLite snapshot migration 구현
- `GET /retraining-runs/{batchId}`와 인증·오류 처리 구현
- TTL cache와 stale fallback 구현

완료 조건: FastAPI 재시작, Vertex 일시 장애, 부분 metric, gate 거부를 포함한 API test 통과.

### Phase 5 — Spring polling

- DB record, polling scheduler와 terminal 처리 구현
- 관리자/운영 화면 또는 내부 조회 API에 상태 및 집계 표시
- timeout·503·stale 응답 재시도 정책 적용

완료 조건: Spring이 GCP credential 없이 FastAPI만 통해 전체 lifecycle을 추적함.

### Phase 6 — Vertex end-to-end 검증

- 실제 feedback fixture로 PipelineJob 1회 실행
- Vertex task state와 FastAPI stage가 일치하는지 확인
- prepare/merge 수치, training heartbeat, gate 및 candidate 결과 대조
- GCS/IAM, polling 부하, 로그의 PII 미포함 확인

완료 조건: 제출부터 terminal 결과까지 Spring에서 누락 없이 조회되고, Google Cloud Console의
수치 및 상태와 일치함.

## 11. 필수 테스트 목록

- 모든 Vertex PipelineJob state의 API status mapping
- component task가 queued/running/succeeded/failed/skipped일 때 stage mapping
- feedback metric 산술 불변식 및 unknown exclusion reason 보존
- `acceptedEventCount - canonicalTrainingRowCount` 중복 병합 수 계산
- merge accepted/excluded 불변식
- step/epoch progress 단조 증가와 0~100 범위
- heartbeat가 없거나 오래됐을 때 warning 처리
- gate `passed=false`를 `rejected`로, 기술 오류를 `failed`로 구분
- publish result가 없는 PipelineJob success를 불완전 상태로 경고
- Vertex 장애 시 cached snapshot과 `stale=true` 반환
- snapshot이 없을 때만 503 반환
- 존재하지 않는 batch 404, 인증 실패 401
- API와 observability artifact에 질문·event ID·request ID가 없는지 검사
- 기존 SQLite file migration 및 FastAPI 재시작 후 조회
- 같은 batch를 여러 번 조회해도 PipelineJob을 추가 제출하지 않는지 검사

## 12. 이번 계획에서 확정한 계약

- 조회 key는 `pipelineRunId`가 아니라 `batchId`를 사용한다.
- Spring은 polling 방식으로 FastAPI만 조회한다.
- Vertex는 lifecycle의 source of truth, GCS는 ML 집계·heartbeat의 source of truth로 삼는다.
- 전체 진행률은 만들지 않고 GPU 학습 단계에만 정확한 퍼센트를 제공한다.
- 필터링 수치는 행 제외, 중복 병합, base 병합 제외의 세 구간으로 나눈다.
- gate 품질 거부는 기술 실패와 분리해 `rejected`로 반환한다.
- 상태 API는 집계만 제공하고 원 질문이나 개별 event 정보는 제공하지 않는다.
