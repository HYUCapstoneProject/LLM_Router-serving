# LLM Router Serving 및 재학습 연동 아키텍처

상태: 초기 구현 기준

기준일: 2026-08-13

## 1. 목적

`LLM_Router-serving`은 Spring Gateway와 ML 시스템 사이에서 다음 세 가지 역할을 맡는
FastAPI 서비스다.

1. 질문을 BGE 계열 모델로 임베딩한다.
2. 캐시 미스 질문을 ModernBERT로 분류해 `easy`, `medium`, `hard` 중 하나를 반환한다.
3. Spring이 GCS에 게시한 escalation feedback 1,000건의 metadata를 받아 기존
   Vertex AI Pipelines 재학습 파이프라인에 연결하고, 승인된 새 모델을 이후 라우팅
   추론에 반영한다.

이 문서는 서비스의 책임 경계와 구현 방향을 정한다. 아직 합의되지 않은 endpoint 이름,
세부 schema, 모델 교체 방식은 초안으로 표시한다.

## 2. 전체 흐름

### 2-1. 온라인 요청

```text
사용자 질문
  -> Spring Gateway
       -> FastAPI: BGE embedding 요청
       <- 정규화된 embedding + embedding model version
       -> Spring: pgvector 캐시 유사도 검사
            -> cache hit: 저장된 응답 반환
            -> cache miss
                 -> FastAPI: ModernBERT 난이도 분류 요청
                 <- easy | medium | hard + confidence + model version
                 -> Spring: versioned routing policy로 실제 LLM model ID 선택
                 -> LLM 호출, 로그 기록, 캐시 저장
```

FastAPI는 embedding과 난이도 분류만 담당한다. pgvector 조회, cache threshold,
캐시 hit/miss 판정, 실제 LLM model ID 선택과 LLM 호출은 Spring의 책임이다.

### 2-2. Feedback 재학습

```text
Spring: 유효 escalation feedback 1,000건 취합
  -> GCS: immutable manifest.json + events.parquet 게시
  -> FastAPI: batch metadata로 /escalate 요청
  -> GCS 객체·schema·checksum·행 수 검증
  -> Vertex AI PipelineJob 제출 (google.cloud.aiplatform)
       -> CPU: feedback 검증·정규화·중복 제거·기존 학습 데이터 병합
       -> CPU launcher: Vertex AI Custom Training job 제출·감시
       -> Vertex AI T4: 공통 ModernBERT 학습 코어로 재학습
       -> CPU: 평가·package·deploy gate
       -> rejected 또는 candidate
  -> 승인된 model package를 FastAPI serving에 반영
  -> readiness 확인 후 트래픽 전환
  -> 실패 시 직전 approved model로 rollback
```

FastAPI는 별도의 학습 loop를 구현하지 않는다. 재학습은 `LLM_Router` 저장소의
`ML/` 구현과 `docs/ml/` 명세를 그대로 사용한다. 공통 학습 CLI, checkpoint 형식과
라벨 순서는 로컬과 Vertex AI에서 동일해야 한다.

## 3. 컴포넌트 책임

| 컴포넌트 | 책임 | 책임이 아닌 것 |
| --- | --- | --- |
| Spring Gateway | 공개 chat API, cache 조회·정책, 실제 LLM 선택·호출, escalation 수집, 1,000건 trigger | embedding·ModernBERT 추론, ML 학습 loop |
| FastAPI serving | BGE inference, ModernBERT inference, 모델 version 노출, 재학습 요청 접수·상태 연결, 승인 모델 반영 | pgvector 직접 조회, cache 판정, 실제 LLM 호출, feedback DB 직접 조회 |
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

### 4-2. ModernBERT 분류

endpoint는 `POST /router/classify`다. cache miss일 때만 호출하며, payload의 단일 기준은
[`api-contract.md`](api-contract.md)다.

고정 ML 계약은 다음과 같다.

- `easy=0`
- `medium=1`
- `hard=2`
- FastAPI가 반환하는 `model`은 참고값이고 난이도 tier가 ML 판정의 기준이다.
- 실제 LLM 매핑은 Spring의 versioned routing policy가 관리한다.
- model package를 적재할 때 tokenizer, checksum, label 순서와 model version을 검증한다.

## 5. 재학습 API와 비동기 job

재학습 제출 endpoint는 `POST /escalate`다. Spring은 feedback event를 request body에
직접 넣지 않는다. 먼저 GCS에 변경 불가능한 `manifest.json`과 `events.parquet`을 게시한
뒤 FastAPI에는 `batchId`, schema version, 두 GCS URI, event count와 요청 시각만 보낸다.

FastAPI는 GCS 객체, manifest metadata, checksum과 행 수를 검증한 뒤 Vertex AI pipeline을
제출하고 `202 Accepted`와 `pipelineRunId`를 반환한다. 이 응답은 pipeline 제출 성공만
의미하며 학습·평가·배포 성공을 의미하지 않는다.

동일한 `batchId`와 동일한 내용의 재시도에는 기존 `pipelineRunId`를 반환한다. 같은
`batchId`에 다른 내용이 오거나 다른 재학습이 진행 중이면 `409 Conflict`를 반환한다.
동시에 실행할 수 있는 재학습 pipeline은 하나다.

정확한 요청·응답 schema, 오류 코드와 검증 순서는
[`api-contract.md`](api-contract.md)를 따른다.

### 5-1. 1,000건 trigger의 의미

Spring이 `/escalate`를 호출하는 기준은 raw DB row가 아니라 중복 제거, PII 제거, 만족
여부 등 export 조건을 통과한 유효 feedback 1,000건이다. pipeline 내부 validator가
추가 품질 검사로 일부 행을 격리할 수 있으므로 `received_count`, `accepted_count`,
`quarantined_count`와 제외 사유별 count는 pipeline metric에 남긴다.

### 5-2. Pipeline 입력 형식

Spring은 기존 ML 계약에 맞춘 immutable artifact를 GCS에 먼저 게시한다. FastAPI는 해당
artifact의 URI와 metadata만 받아 검증한 뒤 Vertex AI pipeline에 전달한다.

```text
feedback/<batch-id>/manifest.json
feedback/<batch-id>/events.parquet
```

manifest에는 적어도 schema version, batch ID, event count, data URI, SHA-256, producer
version과 watermark를 기록한다. FastAPI와 pipeline은 ML validator와 공통 학습 코어를
우회해서는 안 된다.

## 6. 모델 승인과 FastAPI 반영

재학습 완료 checkpoint를 곧바로 운영 모델로 사용하지 않는다. 전달은 **pull 방식**이며
실행 중인 모델을 교체하는 API는 존재하지 않는다.

```text
Vertex AI Pipeline
  -> RouterArena 평가 -> model package -> deploy gate
  -> gs://<bucket>/models/router-modernbert/<version>/candidate/   (pipeline이 게시)
  -> 승인자가 metrics와 gate_result 확인
  -> modernbert-publish approve                                    (LLM_Router 저장소)
  -> gs://<bucket>/models/router-modernbert/<version>/approved/ + CURRENT.json
  -> FastAPI 배포 설정에 package URI와 version 전달
  -> 프로세스 시작 시 다운로드·검증
  -> rolling restart
  -> GET /health 200 확인
```

각 단계의 계약은 다음과 같다.

1. pipeline이 평가, package 검증과 deploy gate를 수행한다. gate 실패 모델은 candidate
   경로에 올라가지 않는다.
2. gate 통과 package만 `candidate/`에 immutable하게 게시된다. 이 단계까지가 pipeline의
   책임이다.
3. 승인은 pipeline 밖의 사람이 `modernbert-publish approve`로 수행한다. 이 명령이
   package를 `approved/`로 복사하며 `gate_status`를 `approved`로 바꾸고 `CURRENT.json`
   포인터를 GCS generation precondition으로 갱신한다.
4. 배포 담당자가 `LLM_ROUTER_ROUTER_MODEL_PACKAGE_URI`와
   `LLM_ROUTER_ROUTER_MODEL_VERSION`을 approved 값으로 바꾼다.
5. FastAPI는 프로세스 시작 시 package를 내려받아 `model_manifest.json` 기준으로 schema
   version, label 순서(`easy=0, medium=1, hard=2`), `model_version`, 파일별 SHA-256을
   검증한다. `gate_status`가 `approved`가 아니면 적재를 거부한다.
6. 검증을 통과한 뒤에만 cache 디렉터리에 게시되므로, 실패한 다운로드는 사용 가능한
   모델 디렉터리를 남기지 않고 `/health`는 503을 유지한다.
7. rolling restart와 `GET /health` readiness probe가 무중단 전환을 보장한다. 새
   인스턴스는 모델 검증·적재를 마친 뒤에만 트래픽을 받는다.
8. rollback은 두 환경변수를 직전 approved version으로 되돌리고 재시작하는 것이다.
   `modernbert-publish rollback`이 `CURRENT.json`을 같은 방향으로 되돌린다.

hot reload와 자동 승인은 도입하지 않는다. 실행 중 모델 교체 대신 재시작을 쓰는 이유는
승인 시점과 트래픽 전환 시점이 배포 기록에 그대로 남고, 부분적으로 교체된 상태가
존재할 수 없기 때문이다.

## 7. 오류·상태·관측성

공통 오류 응답 초안:

```json
{
  "error": "요청을 처리할 수 없습니다.",
  "code": "ROUTER_NOT_READY",
  "requestId": "req-123"
}
```

최소 health endpoint는 embedding model과 router model의 적재 여부 및 version을 각각
노출해야 한다. 둘 중 하나라도 준비되지 않았으면 readiness는 실패해야 한다.

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

- FastAPI endpoint 경로와 camelCase/snake_case 규칙
- BGE model ID/revision, pooling, normalization, prefix와 차원
- 요청당 최대 질문 길이, batch 크기, timeout과 concurrency
- feedback export 조건과 pipeline 추가 격리 정책
- Vertex PipelineJob·CustomJob service account와 IAM 주체
- candidate 승인자와 approved GCS 경로
- FastAPI 장애 시 Spring의 상위 tier fallback 정책

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
