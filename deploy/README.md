# R0 교체 프로필

`r0.env.example`은 같은 서비스 API를 R0로 교체할 때 사용할 구성이다. 운영 기본 목표는 R1이며 GCS에는 활성 objective의 모델 그룹 하나만 둔다. R0 브랜치는 기존 v1 요청·응답, `feedback-inline-v1`, finalLabel CE proxy와 `successfulHttpRequest` 버전 정책을 유지한다. 학습 길이는 512, router 추론과 D_dev/D_gate 평가는 8192다.

실제 secret·승인 baseline version·full code SHA·immutable pipeline/release/data URI를 채운 별도 env를 준비하고 다음을 실행한다. 예제 그대로는 필수값이 없어 실패한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate-r0-profile.py --env-file deploy/r0.env
```

이 명령은 로컬 설정만 검사하고 cloud를 읽거나 수정하지 않는다. Settings와 같이 현재 프로세스 환경변수가 env 파일보다 우선한다. `profile_valid=true`는 실제 배포 승인이 아니며 지표·파일 SHA·동결 pilot은 ML 저장소의 `ml/scripts/prepare-r0-release.py`로 검증한다. 현재 실제 release/IR/approved baseline은 예제에 넣지 않았다.

기존 Docker 이미지의 BGE 설정을 유지한 채 router max length, R0 GCS/SQLite/cache 경로, 구조 보존 입력을 적용한다. runtime은 approved package의 길이·hash를 검증하고 기존 `/router/classify` 및 feedback/status 경로를 사용한다. 공통 증거 reader가 gate/CURRENT와 batch/job/SHA를 확인하기 전에는 Vertex 성공을 승격으로 보고하지 않는다. 실제 모델 활성화는 별도로 확인한다.

교체 시 제출·승격을 중지하고 진행 작업을 정리한다. 기존 모델 그룹 current/previous/anchor와 tokenizer/policy/manifest/gate·승격 증거, 복구용 상태를 개인 Drive에 보존한 뒤 **원격 재읽기와 파일별 SHA**를 검증한다. 기존 GCS 그룹을 generation 조건으로 제거한 후 새 그룹을 게시한다. Drive 동기화 폴더 복사만으로 보존을 완료 처리하지 않는다. 두 objective의 GCS 모델 그룹을 동시에 유지하지 않으므로 교체 유지보수 구간과 Drive 복구를 준비한다.

Spring은 `r0` profile에서 기존 v1 API와 `auto-submit=false`를 유지한다. 실제 VM의 ADC·secret·UID/볼륨, 8192 지연/메모리/동시 요청, Spring timeout, 재학습→gate→CURRENT→실제 로딩 및 rollback 검증 후 운영 활성화를 결정한다. R0 장문 품질 향상이나 운영 지연 gate 통과는 로컬 테스트로 주장하지 않는다.
