# R0 운영 배포

2026-09-27 결정: 기존 A 데이터로 학습한 R0 ModernBERT를 운영·재학습의 출발점으로 사용하고 R1 자동 재학습은 보류한다. `r0.env.example`은 기존 서비스 API를 유지하는 R0 운영 구성이다. GCS에는 활성 objective의 모델 그룹 하나만 둔다. R0 브랜치는 기존 v1 요청·응답, `feedback-inline-v1`, finalLabel CE proxy와 `successfulHttpRequest` 버전 정책을 유지한다. 학습 길이는 512, router 추론과 D_dev/D_gate 평가는 8192다.

실제 secret·승인 baseline version·full code SHA·immutable pipeline/release/data URI를 채운 별도 env를 준비하고 다음을 실행한다. 예제 그대로는 필수값이 없어 실패한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate-r0-profile.py --env-file deploy/r0.env
```

이 명령은 로컬 설정만 검사하고 cloud를 읽거나 수정하지 않는다. Settings와 같이 현재 프로세스 환경변수가 env 파일보다 우선한다. `profile_valid=true`는 실제 배포 승인이 아니며 지표·파일 SHA·동결 pilot은 ML 저장소의 `ml/scripts/prepare-r0-release.py`로 검증한다. 현재 실제 release/IR/approved baseline은 예제에 넣지 않았다.

기존 Docker 이미지의 BGE 설정을 유지한 채 router max length, R0 GCS/SQLite/cache 경로, 구조 보존 입력을 적용한다. runtime은 approved package의 길이·hash를 검증하고 기존 `/router/classify` 및 feedback/status 경로를 사용한다. 공통 증거 reader가 gate/CURRENT와 batch/job/SHA를 확인하기 전에는 Vertex 성공을 승격으로 보고하지 않는다. 실제 모델 활성화는 별도로 확인한다.

교체 시 제출·승격을 중지하고 진행 작업을 정리한다. 기존 모델 그룹 current/previous/anchor와 tokenizer/policy/manifest/gate·승격 증거, 복구용 상태를 개인 Drive에 보존한 뒤 **원격 재읽기와 파일별 SHA**를 검증한다. 기존 GCS 그룹을 generation 조건으로 제거한 후 새 그룹을 게시한다. Drive 동기화 폴더 복사만으로 보존을 완료 처리하지 않는다. 두 objective의 GCS 모델 그룹을 동시에 유지하지 않으므로 교체 유지보수 구간과 Drive 복구를 준비한다.

Spring은 `r0` profile에서 기존 v1 API와 `auto-submit=false`를 유지한다. 실제 VM의 ADC·secret·UID/볼륨, 8192 지연/메모리/동시 요청, Spring timeout, 재학습→gate→CURRENT→실제 로딩 및 rollback 검증 후 운영 활성화를 결정한다. R0 장문 품질 향상이나 운영 지연 gate 통과는 로컬 테스트로 주장하지 않는다.

## 실제 값 준비와 사용자 담당 최소 확인

예제 파일에는 실제 secret이나 모델 URI를 저장하지 않는다. 로컬 복사본은 Git에서 제외된다.

```bash
cp deploy/r0.env.example deploy/r0.env
chmod 600 deploy/r0.env
```

다음 값은 담당자가 실제 산출물에서 채워야 하며 테스트 fixture로 대체할 수 없다.

| 필요한 값 | 출처 |
| --- | --- |
| 내부 API token | 현재 Spring·FastAPI가 공유하는 secret. 채팅에 붙여 넣지 않는다. |
| R0 기준 모델 버전·approved package·CURRENT | 기존 A 실험 checkpoint의 SHA를 확인하고 8192 승인 절차로 만든 모델 패키지 |
| full code SHA·컨테이너 image digest | 검토된 배포 커밋·빌드 결과. 현재 로컬 변경을 반영한 후 생성 |
| pipeline template URI | R0의 실제 component 이미지 digest로 compile한 immutable IR |
| 학습·평가 데이터 URI | 보호된 분할을 유지한 실제 base 및 gate 데이터 |
| preprocessing/training/evaluation/package/baseline/gate URI | ML `prepare-r0-release.py`로 검증한 immutable release 결과 |

사용자가 직접 확인하면 좋은 것은 GCP credit 만료일·잔액, T4 quota, 현재 배포 secret 보관 위치와 Spring 담당자의 배포 가능 일정이다. 코드 수정·실험 결과 탐색·SHA 대조·IR 작성은 구현 담당자가 수행한다. 비용 조건을 확인하기 전 실제 GPU 재학습을 제출하지 않는다.

## 배포 및 전체 경로 인수 순서

1. Spring 자동 제출을 끄고 처리 중인 batch와 파이프라인을 확인한다. 실제 sqlite DB를 안전하게 백업하고 `/app/var/r0` 영구 볼륨의 컨테이너 UID 쓰기 권한을 확인한다. 기존 DB 위치를 새 경로로 바꿀 때 빈 DB로 시작하여 진행 중 batch를 잃지 않는다.
2. preflight를 실행한다. R0 profile은 approved ordinal 패키지와 실제 training 길이를 보존한 8192 inference 계약만 적재한다. 시작·hot reload에 같은 검사가 적용되며 잘못된 새 패키지는 현재 적재 모델을 바꾸지 않는다.
3. 모델 그룹이 바뀌면 위 Drive 보존 절차를 마친 뒤 GCS를 교체한다. 현재 이미 R0 그룹이면 불필요한 R1 전환·보존 작업을 하지 않는다.
4. 검증한 image digest와 release env·영구 볼륨으로 FastAPI를 배포한다. BGE 설정은 유지한다. env preflight 통과만으로 배포나 재학습 성공을 선언하지 않는다.
5. token이 이미 환경변수에 있는 **서버 컨테이너 내부에서** 아래 read-only 명령을 실행한다. `EXPECTED_R0_VERSION`은 실제 승인 모델 버전으로 설정한다. 도구는 token을 출력하지 않는다.

```bash
python scripts/check-r0-readiness.py --expected-model-version "$EXPECTED_R0_VERSION"
```

6. 실제 VM에서 짧은·긴 질문과 동시 요청을 측정하고 8192 추론 메모리·지연·Spring timeout을 확인한다. 전체 왕복이 통과하기 전 Spring 자동 제출은 끈다.
7. PII 제거와 실제 event provenance를 검증한 기존 `feedback-inline-v1` 배치를 수동 제출한다. **`POST /escalate`는 유료 GPU 학습을 시작할 수 있다.** 임의 합성 질문으로 운영 모델을 학습하지 않는다. 승인된 실배치를 재시도할 때 UUID와 payload를 그대로 사용한다.
8. 202의 batchId를 가지고 아래 명령으로 상태를 확인한다. Vertex 성공뿐 아니라 gate 통과·CURRENT 갱신·실제 serving version 일치를 확인한다. gate 실패, 같은 batch 재요청, 모델 로딩 실패, 재시작·복구도 시험하고 결과를 남긴다. 모든 조건 통과 후 Spring 자동 제출을 활성화한다.

```bash
python scripts/check-r0-readiness.py --expected-model-version "$EXPECTED_R0_VERSION" --batch-id "$EXISTING_BATCH_ID"
```

이 스크립트는 HTTP readiness와 기존 배치 조회만 수행한다. `pipeline_verified=false`는 의도적이며 GPU/IAM·모델 품질·승격·복구 인수를 자동으로 대신하지 않는다. R1/R2 모델이 적재된 상태의 기존 `/escalate`는 409 `R0_OBJECTIVE_REQUIRED`로 거절한다.

## 이미지 준비 (운영 VM은 linux/amd64)

Mac의 Docker가 arm64이므로 platform을 생략한 이미지를 VM에 올리지 않는다. 아래 명령은 로컬 빌드만 수행하며 push·배포하지 않는다. `SERVING_RELEASE_SHA`에는 검토·커밋을 완료한 현재 checkout의 전체 SHA를 넣는다. 빌드 시 고정된 BGE·reranker weights를 다운로드하므로 네트워크와 로컬 디스크가 필요하다. token은 build arg로 전달하지 않는다.

```bash
docker buildx build --platform linux/amd64 --load --tag "llm-router-serving:r0-$SERVING_RELEASE_SHA" .
docker image inspect "llm-router-serving:r0-$SERVING_RELEASE_SHA" --format '{{.Os}}/{{.Architecture}} {{.Id}}'
```

`.dockerignore`에서도 실제 `deploy/*.env`를 제외한다. 로컬 image ID는 registry digest가 아니다. 원격 registry 게시 후 배포에는 실제 `image@sha256:...` 값을 고정한다. 현재 문서의 명령을 작성했다는 사실은 빌드·업로드가 완료됐다는 뜻이 아니다.

새 환경파일로 배포할 때 이전 VM 환경변수를 무조건 합치지 않는다. Settings에서 process env가 env file보다 우선한다. 특히 예전 `baseline_metrics_empty.json`, `deploy_gate_rules_smoke.json`, 이전 IR·training config, 기존 publish root 값이 남지 않도록 실제 실행 컨테이너의 유효 설정을 preflight로 다시 검사해야 한다. 기존 모델은 R0이므로 새 A checkpoint로 바꾸는 것은 같은 그룹 내 버전 교체다. 기존 CURRENT와 복구용 모델은 승격 절차에 따라 보존한다.
