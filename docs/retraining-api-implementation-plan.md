# 재학습 API 구현 및 교체 배포 계획

상태: 핵심 계약 구현 및 GCP 교체 배포 완료, 실제 유효 batch E2E 학습 검증은 별도 수행

기준일: 2026-09-12

> 2026-09-16 통합 주석: 2~6장의 “현재/목표 차이”는 구현 작업을 정의할 당시의 gap
> 기록이다. 현재 source에는 acceptance 확장, 상태/current-model/developer-policy endpoint와
> hot-swap이 구현되어 있다. offline 모델군 비교는 Colab notebook에 한정하고 KFP/Vertex
> 경로는 ModernBERT만 사용한다. 통합 데이터 전환의 최신 계획은
> sibling `LLM_Router/docs/ml/integrated-routing-data-and-model-comparison-plan.md` 10~15장을
> 우선한다. 실제 유효 batch의 Vertex/GCS E2E 검증은 여전히 별도 작업이다.

단일 API 기준: [`api-contract.md`](api-contract.md)

Spring 전달 기준: [`backend-integration-handoff.md`](backend-integration-handoff.md)

HTTP feedback 기준:
[`escalation-feedback-inline-v1.md`](escalation-feedback-inline-v1.md)

Legacy 배포 참고:
[`escalation-feedback-batch-v0.1.md`](escalation-feedback-batch-v0.1.md)

## 1. 이번 단계의 범위

FastAPI·ML pipeline·관측 artifact·자동 승격 코드를 구현했고, digest 고정 이미지와 v3
Vertex pipeline template 및 FastAPI container를 GCP에 배포했다. 로컬 contract/test와
배포된 OpenAPI, 모델 동기화, GCS 파생 데이터 쓰기 권한까지 검증했다. 비용이 발생하는 실제
유효 feedback batch의 GPU E2E 학습은 운영 smoke로 분리한다.

## 2. 현재 구현과 목표 계약의 차이

| 영역 | 현재 구현 | 목표 계약 | 필요한 변경 |
| --- | --- | --- | --- |
| `/escalate` 입력 | inline `events.length >= 1` 구현됨 | 유지 | Spring은 GCS/Parquet/manifest 작업 없음 |
| `/escalate` 응답 | `status`, `batchId`, `pipelineRunId`만 반환 | logical/candidate version, training/policy mode, `statusUrl` 추가 | FastAPI schema와 route 수정 |
| 최종 상태 갱신 | 다음 `/escalate`가 올 때 active job만 reconcile | 조회할 때 Vertex/GCS와 동기화 | 독립 status service 필요 |
| 상태 조회 API | 없음 | `GET /retraining-runs/{batchId}` | route, schema, cache, snapshot 구현 |
| 성공/실패 구분 | SQLite에 축약 상태만 저장 | gate 거부, 기술 실패, 취소, 단계와 code 구분 | Vertex task detail 및 publish/gate artifact 필요 |
| 학습 진행률 | 없음 | GPU optimizer step heartbeat | ML callback 및 GCS progress artifact 필요 |
| 현재 모델 조회 | inference engine 내부 artifact version만 접근 가능 | desired/serving/logical version과 swap 상태 조회 | pointer/manager/watcher 상태 확장 |
| `CURRENT.json` 읽기 | artifact version, URI, manifest checksum만 읽음 | logical version, training mode, previous known-good, approved time도 읽음 | pointer dataclass/parser 확장 |
| version 정책 | 성공 job 수 3회로 고정 | `successfulHttpRequest`와 `validatedUniqueBatch` 전환 | durable policy store와 developer API 필요 |
| 고유 batch 검증 | request fingerprint와 batch ID 멱등성만 있음 | canonical training content digest 중복 확인 | ML digest 출력과 committed digest ledger 필요 |
| 개발자 인증 | 일반 internal token만 있음 | 별도 token, actor, feature flag, audit | config/auth/middleware 추가 |

이미 구현되어 재사용할 부분은 다음과 같다.

- `/escalate` inline event의 schema·label 상승·만족·PII 보증 검증과 FastAPI 내부 artifact화
- `batchId` 멱등성, persistent SQLite lease, 단일 pipeline 동시 실행 제한
- `n.1 -> n.2 -> (n+1).0`, 5번째 major `pristine_rebase` 계획
- cycle feedback + compact replay materialization
- gate 통과 package 자동 승인과 atomic `CURRENT.json` 갱신
- watcher의 package checksum/label 검증 및 load-before-swap
- current, previous known-good, active anchor 보존

## 3. 작업 묶음 A — FastAPI response와 상태 조회

대상 저장소: `LLM_Router-serving`

주요 파일:

```text
src/llm_router_serving/schemas.py
src/llm_router_serving/main.py
src/llm_router_serving/retraining/pipeline.py
src/llm_router_serving/retraining/service.py
src/llm_router_serving/retraining/store.py
tests/test_api.py
tests/test_retraining.py
```

변경 내용:

1. `EscalateResponse`를 `escalation-acceptance-v1`으로 확장한다.
2. `PipelineClient.state()` 외에 create/start/end/update 시각, 원래 Vertex 상태, task detail과
   sanitized error를 반환하는 `describe()`를 추가한다.
3. `RetrainingStatusService`가 SQLite, Vertex와 GCS observability 결과를
   `retraining-status-v1`으로 집계한다.
4. 정상 snapshot을 SQLite에 저장하고 Vertex/GCS 장애 시 `stale=true`로 재사용한다.
5. 조회 폭주를 막기 위해 같은 batch에 10초 TTL cache를 둔다.
6. `GET /retraining-runs/{batchId}`를 추가한다.
7. pipeline terminal 상태는 HTTP 오류가 아니라 HTTP 200 body로 반환한다.

SQLite에는 최소 다음 column이 추가된다.

```text
created_at, submitted_at, started_at, ended_at,
pipeline_status, current_stage, failure_code, failure_stage,
observability_uri, status_snapshot_json, last_synced_at,
version_policy_mode, canonical_batch_digest
```

기존 DB를 유지하기 위해 destructive migration 없이 `ALTER TABLE ADD COLUMN`과 migration
test를 사용한다. raw 질문과 event/request ID는 SQLite status snapshot에 넣지 않는다.

완료 조건:

- queued/running/succeeded/rejected/failed/cancelled fixture가 계약과 일치한다.
- FastAPI 재시작 후에도 마지막 terminal 결과를 조회할 수 있다.
- Vertex 장애에서 snapshot이 있으면 200/stale, 없으면 503이다.
- 동일 batch polling이 새 pipeline을 제출하거나 version count를 바꾸지 않는다.

## 4. 작업 묶음 B — 현재 모델과 hot-swap 상태 조회

대상 저장소: `LLM_Router-serving`

주요 파일:

```text
src/llm_router_serving/model_source.py
src/llm_router_serving/model_manager.py
src/llm_router_serving/model_updates.py
src/llm_router_serving/schemas.py
src/llm_router_serving/main.py
tests/test_model_source.py
tests/test_model_updates.py
tests/test_api.py
```

변경 내용:

1. `CurrentModelPointer`가 기존 `CURRENT.json`의 `logical_model_version`, `training_mode`,
   `previous_model_version`, `approved_at`을 검증해 읽는다.
2. `ModelManager` snapshot에 실제 serving artifact/logical version, loaded time와 마지막 load
   error를 추가한다.
3. `ModelPointerWatcher`가 desired pointer, 마지막 check/swap 시각, load 진행과 실패를
   thread-safe snapshot으로 유지한다.
4. `GET /router/model/current`를 추가한다.
5. load 실패 시 직전 model을 계속 쓰고 API는 `loadFailed`와 sanitized error만 노출한다.

ML publisher는 필요한 current logical version, training mode, previous model metadata를 이미
`CURRENT.json`에 쓰므로 이 작업만을 위한 pointer schema 변경은 필요하지 않다.

완료 조건:

- 동일 pointer는 `inSync`, 새 pointer 적재 중은 `loading`, 적재 오류는 `loadFailed`다.
- load 실패 중 `/router/classify`는 직전 known-good version으로 계속 성공한다.
- status API의 `activation`과 current model API가 같은 watcher snapshot을 사용한다.

## 5. 작업 묶음 C — 개발자 version policy

대상 저장소: `LLM_Router-serving`, canonical digest는 `LLM_Router`

Serving 주요 파일:

```text
src/llm_router_serving/config.py
src/llm_router_serving/schemas.py
src/llm_router_serving/main.py
src/llm_router_serving/retraining/store.py
src/llm_router_serving/retraining/strategy.py
src/llm_router_serving/retraining/service.py
tests/test_config.py
tests/test_api.py
tests/test_retraining.py
```

변경 내용:

1. `retraining_version_policy`와 `retraining_version_policy_audit` table을 만든다.
2. 환경변수의 mode는 최초 DB bootstrap에만 사용하고 이후 durable DB revision을 기준으로
   삼는다.
3. `GET/PUT /developer/retraining/version-policy`를 별도 인증으로 추가한다.
4. 제출 시 활성 policy revision/mode를 job에 snapshot한다.
5. `nextMajorBoundary` pending change는 `.0` run commit 직후 transaction 안에서 적용한다.
6. 이미 제출된 run과 과거 history는 새 mode로 다시 계산하지 않는다.
7. 동일 batch retry, rejected/failed/cancelled run은 commit count에 포함하지 않는다.

추가 설정:

```text
LLM_ROUTER_ENABLE_DEVELOPER_API=false
LLM_ROUTER_DEVELOPER_API_TOKEN=<separate secret>
LLM_ROUTER_RETRAINING_VERSION_POLICY_MODE=successfulHttpRequest
LLM_ROUTER_RETRAINING_CONTROL_ROOT_URI=gs://<artifact-bucket>/retraining-control
```

`callsPerMajor=3`, `rebaseEveryMajorCycles=5`는 이 API에서 수정하지 않는다. 이 값까지 한
endpoint에서 바꾸면 version 의미와 anchor 보존 규칙이 동시에 달라지므로 별도 schema
revision과 migration 없이 열지 않는다.

완료 조건:

- revision compare-and-set, active run 충돌, boundary 예약과 audit test가 통과한다.
- mode 변경 전후 job이 자신이 제출된 mode를 그대로 보고한다.
- FastAPI 재시작 후 mode/revision/pending change가 유지된다.

## 6. 작업 묶음 D — ML observability와 canonical batch digest

대상 저장소: `LLM_Router`

주요 파일:

```text
ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py
ml/src/llm_router_labeling/training/feedback_retraining_cli.py
ml/src/llm_router_labeling/training/modernbert_training.py
ml/src/llm_router_labeling/training/modernbert_training_cli.py
ml/src/llm_router_labeling/deployment/deploy_gate.py
ml/src/llm_router_labeling/deployment/model_publish.py
ml/kubeflow/components/feedback_retraining.py
ml/kubeflow/components/modernbert_training.py
ml/kubeflow/components/model_evaluation_promotion.py
ml/kubeflow/pipelines/modernbert_retraining.py
ml/kubeflow/compile_training_workflows.py
ml/tests/
```

GCS run artifact:

```text
gs://<artifact-bucket>/retraining-runs/<batchId>/observability-v1/
├── metrics/feedback-prepare.json
├── metrics/feedback-merge.json
├── progress/training.json
├── metrics/training-final.json
├── metrics/evaluation.json
├── gate/gate-result.json
└── result/publish-result.json
```

Validated batch control artifact:

```text
gs://<artifact-bucket>/retraining-control/validated-batches/
├── pending/<canonicalBatchDigest>.json
└── committed/<canonicalBatchDigest>.json
```

변경 내용:

1. prepare/merge의 기존 집계를 위 결정적 URI에도 게시한다.
2. accepted canonical row의 안정된 content digest와 row count를 prepare 결과에 추가한다.
3. validated mode에서는 GPU 제출 전에 GCS control ledger의 committed digest를 확인하고,
   중복이면 `NO_UNIQUE_FEEDBACK`으로 조기 종료한다.
4. 신규 digest는 generation precondition으로 pending marker를 만들고, 자동 승격 성공 뒤
   committed marker로 전환한다. 중단된 marker는 FastAPI가 PipelineJob과 `CURRENT.json`의
   `pipeline_run_id`를 대조해 commit 또는 폐기한다.
5. 공통 학습 코어에 optional progress callback을 넣고 Vertex CLI만 GCS heartbeat를 쓴다.
6. gate 거부는 실패 exit 전에 `gate-result.json`을 반드시 쓴다.
7. 자동 승인과 `CURRENT.json` 갱신 뒤 `publish-result.json`을 쓴다.
8. 관측 JSON과 digest ledger에는 질문, event/request ID, credential을 넣지 않는다.

Canonical digest 규칙은 API 계약과 동일하게 정규화된 질문, final label, sample weight의
stable JSON 직렬화와 정렬 순서를 test fixture로 고정한다. FastAPI는 commit된 digest만
ledger에 보관한다.

완료 조건:

- 같은 canonical content를 Parquet row 순서만 바꿔도 digest가 같다.
- 질문 또는 label/sample weight가 바뀌면 digest가 다르다.
- 이미 commit된 digest는 GPU job 생성 전에 거부된다.
- pending marker 도중 process가 중단돼도 다음 reconcile에서 영구 lock이 남지 않는다.
- heartbeat sequence와 optimizer step은 단조 증가한다.
- gate rejection, publish success와 기술 실패가 API failure code로 결정적으로 대응된다.

## 7. 작업 묶음 E — Spring 연동

대상: Spring 백엔드 팀

1. [`backend-integration-handoff.md`](backend-integration-handoff.md)의 12-field inline event와
   4-field HTTP JSON을 구현한다. Spring에는 GCS/Parquet/manifest 코드를 추가하지 않는다.
2. `202`의 ID, logical/candidate version, status URL을 DB에 저장한다.
3. polling scheduler는 HTTP 장애/stale/unknown과 terminal pipeline 실패를 구분한다.
4. `succeeded`와 `activation.active`를 별도 상태로 저장한다.
5. `/router/classify.modelVersion`을 원 request log에 계속 기록한다.
6. 개발자 policy endpoint는 호출하지 않는다.

권장 Spring 상태 column:

```text
batch_id, pipeline_run_id, logical_model_version, candidate_model_version,
pipeline_status, pipeline_stage, result_outcome, failure_code,
activation_status, submitted_at, completed_at, last_polled_at
```

## 8. GCP/GCS 승인 및 교체 순서

각 단계는 앞 단계의 로컬 test와 artifact 확인 후 별도로 승인한다.

### 승인 1 — GCS observability와 lifecycle

- 기존 bucket 아래 `retraining-runs/` prefix 사용 여부 확정
- 기존 bucket 아래 `retraining-control/validated-batches/` digest ledger 사용 승인
- FastAPI read, Vertex pipeline write 권한 확인
- 완료 metric/result 보존 기간과 mutable heartbeat lifecycle 설정
- pipeline root 임시 artifact 1일 TTL 및 soft-delete 비용 정책 확인

새 bucket은 필수가 아니다. 기존 `us-west1` bucket의 prefix와 IAM 조건으로 분리할 수 있다.
Digest marker는 batch당 수백 byte 수준이라 model package나 Parquet에 비해 저장비 영향이
사실상 없지만, 운영 lineage이므로 model과 같은 보존기간을 적용한다.

### 승인 2 — Secret과 network

- 별도 `llm-router-developer-api-token` Secret 생성
- FastAPI runtime service account에 secret access 부여
- `/developer/**`를 VPC 또는 관리자 source range로 제한
- Spring service account에는 developer secret access를 부여하지 않음

### 승인 3 — ML component image

- 변경된 data-processing, training, evaluation/promotion image build
- immutable digest로 Artifact Registry push
- 각 image SBOM/vulnerability 및 로컬 test 결과 확인

학습 progress callback이 training image에 들어가므로 FastAPI image만 교체해서는 전체 계약이
완성되지 않는다.

### 승인 4 — Vertex pipeline template

- 새 component image digest로 KFP v2 IR 재컴파일
- observability URI parameter와 publish result 단계 확인
- GCS의 production template object를 새 immutable version으로 업로드
- 기존 template URI는 rollback용으로 보존

### 승인 5 — FastAPI image와 VM 설정

- 새 serving image build/push
- observability root, developer feature flag/token, status cache 설정 추가
- persistent SQLite volume backup 및 non-destructive migration 확인
- GCE VM container를 새 digest로 교체
- `/health`, 기존 embed/classify, 새 model/status API smoke 확인

### 승인 6 — 1행 end-to-end smoke

- 별도 smoke prefix에 유효 feedback 1행 게시
- `/escalate` 202, running status, terminal result와 activation 확인
- 같은 batch retry가 새 run/version을 만들지 않는지 확인
- 개발자 API로 mode 조회 및 boundary 예약/취소가 아닌 실제 전환 test는 별도 test DB에서 수행
- Cloud Console의 PipelineJob, GCS artifact, `CURRENT.json`, API 응답을 대조

Smoke가 실제 T4 학습과 자동 승격을 유발하므로 이 단계는 별도 비용·모델 변경 승인 후
실행한다.

## 9. 교체 배포의 rollback 기준

- 새 pipeline template 실패: 기존 immutable IR URI로 FastAPI 설정을 되돌린다.
- 새 FastAPI API 실패: 기존 serving container digest와 SQLite backup으로 되돌린다.
- 새 모델 load 실패: container rollback 없이 watcher가 직전 loaded model을 계속 사용한다.
- 잘못된 모델이 승격됨: `CURRENT.json`을 `previousKnownGood` approved package로 rollback하고
  watcher 반영을 확인한다.

어떤 rollback에서도 이미 접수한 `batchId`의 fingerprint나 FastAPI의 immutable feedback
artifact를 덮어쓰지 않는다.

## 10. 승인 단위 권장안

한 번에 cloud 전체를 바꾸지 않고 다음 네 번으로 나누는 것이 안전하다.

1. **로컬 코드 승인:** 작업 묶음 A~D, unit/contract test, IR compile 결과
2. **GCS/IAM 승인:** observability prefix, lifecycle, service account와 developer secret
3. **Artifact/Vertex 승인:** component image push와 새 immutable pipeline template 게시
4. **Serving 교체 승인:** FastAPI image 교체 후 1행 end-to-end smoke와 rollback 확인

각 승인 단계에서 만들어질 정확한 command와 변경 대상 resource diff는 실행 직전에 다시
제시한다.
