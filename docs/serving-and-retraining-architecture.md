# LLM Router Serving 및 재학습 연동 아키텍처

상태: 현행 ModernBERT serving·재학습 구현 기준

기준일: 2026-09-16

## 1. 목적

`LLM_Router-serving`은 Spring Gateway와 ML 시스템 사이에서 다음 네 가지 역할을 맡는
FastAPI 서비스다.

1. 질문을 BGE 계열 모델로 임베딩한다.
2. Spring이 pgvector에서 찾은 cache 후보를 선택적 cross-encoder로 정밀 재채점한다.
3. 캐시 미스 질문을 현재 ModernBERT router로 분류해 `easy`, `medium`, `hard` 중 하나를
   반환한다. 향후 backbone 교체 뒤에도 이 HTTP 출력 계약은 유지한다.
4. Spring이 HTTP로 보낸 escalation feedback event를 검증·artifact화해 기존 Vertex AI
   Pipelines 재학습 파이프라인에 연결하고, gate를 통과한 새 모델을 이후 라우팅 추론에
   반영한다.

이 문서는 서비스의 책임 경계와 구현 방향을 정한다. 통합 데이터·최종 비교 모델군·threshold와
ModernBERT-only 재학습 경계는 sibling `LLM_Router` 저장소의
`docs/ml/integrated-routing-data-and-model-comparison-plan.md`를 우선한다.
다른 backbone 비교는 Colab notebook에서 끝나며 이 서비스나 Vertex 재학습 pipeline에
연결하지 않는다. 최초 ModernBERT checkpoint도 Colab에서 학습하며, 이 서비스가 제출하는
Vertex PipelineJob은 feedback 재학습만 수행한다.

Spring이 재학습용 GCS 객체까지 만들지 않는 이유는 온라인 백엔드를 ML artifact schema와
cloud storage 운영에 결합시키지 않기 위해서다. Spring은 business event와 trigger만
소유하고, FastAPI가 HTTP event를 ML 입력으로 바꾸는 anti-corruption boundary가 된다.
Parquet/manifest/checksum/replay/GCS 변경은 이 경계 안에서만 일어난다.

## 2. 전체 흐름

### 2-1. 온라인 요청

```text
사용자 질문
  -> Spring Gateway
       -> FastAPI: BGE embedding 요청
       <- 정규화된 embedding + embedding model version
       -> Spring: pgvector에서 cache 후보 검색
            -> FastAPI: 선택적으로 query + 후보 목록 재채점
            <- 순서가 보존된 cross-encoder scores + reranker model version
            -> Spring: model-version별 threshold로 cache hit 판정
            -> cache hit: 저장된 응답 반환
            -> cache miss
                 -> FastAPI: ModernBERT 난이도 분류 요청
                 <- easy | medium | hard + confidence + model version
                 -> Spring: versioned routing policy로 실제 LLM model ID 선택
                 -> LLM 호출, 로그 기록, 캐시 저장
```

FastAPI는 embedding, 선택적 후보 재채점과 난이도 분류를 담당한다. pgvector 조회,
reranker model-version별 cache threshold, 캐시 hit/miss 판정, 실제 LLM model ID 선택과
LLM 호출은 Spring의 책임이다.

### 2-2. Feedback 재학습

```text
Spring: 재학습할 escalation feedback 1건 이상 취합·PII 제거
  -> FastAPI: feedback-inline-v1 JSON으로 /escalate 요청
  -> FastAPI: schema·label 상승·만족·PII 보증·eventId 중복 검증
  -> FastAPI: eventId 순 정규화, Parquet·SHA-256·manifest 생성
  -> FastAPI service account로 cycle/replay artifact를 GCS에 immutable 저장
  -> Vertex AI PipelineJob 제출 (google.cloud.aiplatform)
       -> CPU: feedback 검증·정규화·중복 제거·compact rehearsal 병합
       -> CPU launcher: Vertex AI Custom Training job 제출·감시
       -> Vertex AI T4: 공통 ModernBERT 학습 코어로 재학습
       -> CPU: 평가·package·deploy gate
       -> gate 통과 시 candidate 게시·자동 승인·CURRENT.json 갱신
  -> FastAPI가 CURRENT.json을 감지해 새 model package를 검증·적재
  -> 실패 시 직전 loaded model을 계속 사용
```

FastAPI는 별도의 학습 loop를 구현하지 않는다. 재학습은 `LLM_Router` 저장소의
`ml/` 구현과 `docs/ml/` 명세를 그대로 사용한다. 공통 학습 CLI, checkpoint 형식과
라벨 순서는 로컬과 Vertex AI에서 동일해야 한다.

## 3. 컴포넌트 책임

| 컴포넌트 | 책임 | 책임이 아닌 것 |
| --- | --- | --- |
| Spring Gateway | 공개 chat API, cache 후보 조회·threshold·hit/miss 정책, 실제 LLM 선택·호출, escalation 수집·PII 제거·batch trigger·inline 전송 | GCS/Parquet/manifest, embedding·cross-encoder·ModernBERT 추론, ML 학습 loop |
| FastAPI serving | BGE inference, 선택적 cross-encoder 후보 점수, ModernBERT inference, 모델 version 노출, inline feedback 검증·artifact화·pipeline 제출, 승인 모델 반영 | pgvector 직접 조회, cache 판정, 실제 LLM 호출, Spring DB 직접 조회 |
| Vertex AI Pipelines | feedback 검증·병합, 학습 orchestration, 평가, package, deploy gate와 lineage | 온라인 요청 처리, Spring DB 직접 조회, 현재 serving process 직접 변경 |
| Vertex AI Custom Training | ModernBERT GPU 재학습 | pipeline 전체 orchestration, serving traffic 전환 |
| GCS/Artifact Registry | versioned dataset·model·metric과 immutable image 보관 | 승인 정책 결정 |

## 4. 온라인 추론 계약

### 4-1. BGE embedding

endpoint는 `POST /router/embed`다. cache 조회 전에 항상 호출하며, payload의 단일 기준은
[`api-contract.md`](api-contract.md)다.

구현 시 다음 값은 학습·캐시 실험과 완전히 같아야 한다.

- BGE의 정확한 model ID와 revision
- pooling 방식
- L2 normalization 여부
- query prefix 사용 여부
- vector 차원

현재 계약은 `BAAI/bge-large-en-v1.5` revision
`d4aa6901d3a41ba39fb536a557fa166f842b0e09`, CLS pooling, 1,024차원, FastAPI 측 L2
normalization, 질문 간 대칭 비교에서 prefix 미사용이다. embedding version이 달라지면
기존 cache vector와 섞지 않는다.

### 4-2. 선택적 cache reranker

endpoint는 `POST /router/rerank`다. Spring이 pgvector에서 찾은 query와 후보 목록을 보내면
FastAPI는 입력 순서와 같은 순서로 cross-encoder 점수와 `modelVersion`을 반환한다.
FastAPI는 후보를 정렬하거나 hit/miss를 결정하지 않는다.

현재 이미지는 `cross-encoder/quora-distilroberta-base`의 고정 revision을 포함하지만,
false-hit gate를 통과한 운영 승인이 없으므로 기본 비활성이다. 배포가
`LLM_ROUTER_RERANK_MODEL_ID=/opt/models/quora-distilroberta-base`를 명시한 경우에만
로딩한다. 비활성·미로딩 상태의 503, 추론 오류 또는 비유한 점수는 Spring에서 cache
miss로 처리한다. 리랭커 상태는 `/health.rerankModel`에 노출하지만 BGE/Router readiness를
막지 않는다.

threshold는 점수 scale과 checkpoint에 종속되므로 Spring은 반드시 `modelVersion`과 함께
관리한다. 서로 다른 version의 점수나 BGE cosine threshold를 혼용하지 않는다.

### 4-3. 3단계 router 분류

endpoint는 `POST /router/classify`다. 현재 적재 모델은 ModernBERT이며 cache miss일 때만
호출한다. payload의 단일 기준은
[`api-contract.md`](api-contract.md)다.

고정 ML 계약은 다음과 같다.

- `easy=0`
- `medium=1`
- `hard=2`
- FastAPI가 반환하는 `model`은 참고값이고 난이도 tier가 ML 판정의 기준이다.
- 실제 LLM 매핑은 Spring의 versioned routing policy가 관리한다.
- model package를 적재할 때 tokenizer, checksum, label 순서와 model version을 검증한다.

## 5. 재학습 API와 비동기 job

재학습 제출 endpoint는 `POST /escalate`다. Spring은 feedback event 전체를
`feedback-inline-v1` JSON request body에 넣어 보낸다. Spring은 Parquet, checksum,
manifest와 GCS URI를 생성하거나 GCS에 업로드하지 않는다.

FastAPI는 inline schema와 event 의미를 검증하고 이를 기존 ML 계약의 snake_case
Parquet으로 변환한다. FastAPI가 checksum과 `feedback-export-v0.1` manifest를 만든 뒤
GCS에 기록하고 Vertex AI pipeline을 제출한다. `202 Accepted`와 `pipelineRunId`는 pipeline
제출 성공만 의미하며 학습·평가·배포 성공을 의미하지 않는다.

제출 직전 FastAPI는 현재 3-call cycle 전체와 직전 compact replay를 결합한 파생
manifest/Parquet, 그리고 다음 호출에서 사용할 label별 최대 32행 replay snapshot을
`LLM_ROUTER_RETRAINING_DERIVED_DATA_ROOT_URI`에 immutable하게 기록한다. Spring DB에서
과거 데이터를 다시 요청하지 않는다.

동일한 `batchId`와 동일한 내용의 재시도에는 기존 `pipelineRunId`를 반환한다. 같은
`batchId`에 다른 내용이 오거나 다른 재학습이 진행 중이면 `409 Conflict`를 반환한다.
동시에 실행할 수 있는 재학습 pipeline은 하나다.

정확한 요청·응답 schema, 오류 코드와 검증 순서는
[`api-contract.md`](api-contract.md)를 따른다.

### 5-1. 재학습 trigger와 API batch 크기

Spring이 `/escalate`를 호출하는 양과 시점은 Spring이 소유한다. FastAPI는
`events.length >= 1`과 각 event의 schema·label 상승·만족·PII 제거 보증을 검사한다.
pipeline validator가
추가 품질 검사로 일부 행을 격리할 수 있으므로 `received_count`, `accepted_count`,
`quarantined_count`와 제외 사유별 count는 pipeline metric에 남긴다.

승격까지 성공한 HTTP 호출 3회마다 logical major version이 증가한다. 첫 두 호출은 같은
`.0` anchor에 cycle feedback을 누적한 minor fine-tuning이고, 세 번째는 동일 anchor와 세
batch 전체를 사용한 major consolidation이다. major consolidation 5회마다 순정
ModernBERT와 compact replay로 rebase한다. 상세 계약은 `LLM_Router` 저장소의
`docs/ml/continual-retraining-strategy.md`를 따른다.

### 5-2. Pipeline 입력 형식

FastAPI가 inline event를 기존 ML 계약에 맞춘 immutable artifact로 만들어 Vertex AI
pipeline에 전달한다.

```text
<derived-root>/<candidate-version>/feedback/manifest.json
<derived-root>/<candidate-version>/feedback/events.parquet
<derived-root>/<candidate-version>/replay/manifest.json
<derived-root>/<candidate-version>/replay/events.parquet
```

manifest에는 적어도 schema version, batch ID, event count, data URI, SHA-256, producer
version과 watermark를 FastAPI가 기록한다. Spring은 이 내부 artifact를 생성하거나 직접
읽지 않는다. FastAPI와 pipeline은 ML validator와 공통 학습 코어를 우회해서는 안 된다.

## 6. 모델 자동 승격과 FastAPI 반영

재학습 완료 checkpoint를 곧바로 운영 모델로 사용하지 않는다. 자동 deploy gate와 package
검증을 통과한 모델만 **pull 방식**으로 반영한다.

```text
Vertex AI Pipeline
  -> RouterArena 평가 -> model package -> deploy gate
  -> gs://<bucket>/models/router-modernbert/<version>/candidate/   (pipeline이 게시)
  -> pipeline이 modernbert-publish approve 자동 실행
  -> gs://<bucket>/models/router-modernbert/<version>/approved/ + CURRENT.json
  -> FastAPI가 CURRENT.json을 polling
  -> 새 package를 별도로 다운로드·검증·모델 로드
  -> 검증 완료 후 in-memory classifier 원자적 교체
```

각 단계의 계약은 다음과 같다.

1. pipeline이 평가, package 검증과 deploy gate를 수행한다. gate 실패 모델은 candidate
   경로에 올라가지 않는다.
2. gate 통과 package만 `candidate/`에 게시된다.
3. 다음 pipeline component가 `modernbert-publish approve`를 자동 수행한다. 이 명령이
   package를 `approved/`로 복사하며 `gate_status`를 `approved`로 바꾸고 `CURRENT.json`
   포인터를 GCS generation precondition으로 갱신한다.
4. FastAPI는 `CURRENT.json`을 시작 시 읽고 이후 기본 60초 간격으로 확인한다.
5. FastAPI는 package를 내려받아 `model_manifest.json` 기준으로 schema
   version, label 순서(`easy=0, medium=1, hard=2`), `model_version`, 파일별 SHA-256을
   검증한다. `gate_status`가 `approved`가 아니면 적재를 거부한다.
6. 검증과 모델 로딩이 모두 성공한 뒤에만 classifier 참조를 원자적으로 교체한다. 실패하면
   직전 모델이 계속 요청을 처리한다.
7. 진행 중 요청은 이미 얻은 이전 classifier 참조로 완료되고 새 요청부터 새 모델을 쓴다.
8. rollback은 `modernbert-publish rollback`으로 `CURRENT.json`을 이전 approved package로
   되돌리며 watcher가 같은 절차로 자동 반영한다.
9. 승인 완료 후 candidate 중복본을 제거하고 current, previous known-good, 현재 cycle의
   anchor만 보존해 model storage를 제한한다.

## 7. 오류·상태·관측성

Spring에서 재학습 lifecycle과 필터링 수치를 조회하는 endpoint는
`GET /retraining-runs/{batchId}`로 구현되어 있다. 정확한 response와 enum은
[`api-contract.md`](api-contract.md)의 7장이 기준이며, 실제 유효 feedback batch의
Vertex/GCS E2E와 activation은 별도 cloud smoke가 남아 있다.

Pipeline `succeeded`는 gate 통과와 approved package 게시, `CURRENT.json` 전환까지다. 실제
FastAPI 메모리 적재는 별도 `activation.status`로 표현한다. `GET /router/model/current`는
pointer가 원하는 `desired` model과 현재 추론에 쓰는 `serving` model을 함께 반환한다.

Version trigger mode는 별도 developer API로만 전환한다. 기본 연구 mode는 승격까지 성공한
HTTP-triggered run 3회이고, 대안은 전처리 후 canonical row가 존재하며 이전과 다른
canonical digest를 가진 `validatedUniqueBatch` 3회다. mode 변경은 version count를
초기화하지 않고 이미 제출된 run을 재해석하지 않는다. `/developer/**`는 Spring 연동 범위가
아니며 별도 token과 network allowlist를 사용한다.

공통 오류 응답 초안:

```json
{
  "error": "요청을 처리할 수 없습니다.",
  "code": "ROUTER_NOT_READY",
  "requestId": "req-123"
}
```

`/health`는 embedding/router 적재 readiness만 판정한다. 정확한 router artifact/logical
version과 hot-swap 상태는 `/router/model/current`가 노출한다. 둘 중 하나라도 준비되지
않았으면 readiness는 실패해야 한다.

로그와 metric에서 다음 식별자를 연결한다.

- `requestId`
- `batchId`, Vertex PipelineJob resource name, Vertex CustomJob ID
- embedding model version, router model version
- data version, training config version, evaluation version
- cache 요청 latency와 inference latency
- pipeline 단계별 상태, 제외 사유, gate 결과
- 이전·신규 serving model version과 승인·전환 시각

질문 원문, credential과 개인정보는 일반 로그에 기록하지 않는다.

## 8. 구현 전에 확정할 항목

- BGE model ID/revision, pooling, normalization, prefix와 차원 변경 여부
- 요청당 timeout과 concurrency
- feedback export 조건과 pipeline 추가 격리 정책
- Vertex PipelineJob·CustomJob service account와 IAM 주체
- FastAPI 장애 시 Spring의 상위 tier fallback 정책
- 여러 FastAPI replica의 `activation=active` 집계 기준과 SQLite shared store 전환 시점
- `validatedUniqueBatch` 검증 후 운영 기본 mode로 바꿀 시점

## 9. 연관 기준 문서

FastAPI의 정확한 endpoint와 payload 계약은 같은 저장소의
[`api-contract.md`](api-contract.md)를 기준으로 한다.

세부 ML 구현과 인프라 결정은 `LLM_Router` 저장소의 다음 문서를 기준으로 한다.

- `docs/mvp-scope-and-checklist.md`
- `docs/ml/backend-infrastructure-agreement.md`
- `docs/ml/backend-ml-integration-meeting-guide.md`
- `docs/ml/modernbert-training-spec.md`
- `docs/ml/workstream-checklist.md`
- `ml/kubeflow/pipelines/modernbert_retraining.py`
- `docs/ml/gcp-rollout-and-serving-integration-plan.md`

이 문서와 ML 기준 문서가 충돌하면 label/checkpoint/학습 pipeline은 `LLM_Router`의 현행
ML 명세를 우선하고, FastAPI API 변경은 Spring과의 versioned contract 및 양쪽 contract
test를 함께 갱신한다.
