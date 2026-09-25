# API 예시·전달 파일 안내

현재 HTTP의 단일 기준은 [API 계약](../api-contract.md)이며, 설치·환경 설정은 [저장소 README](../../README.md)를 따른다. 여기의 날짜, version, URI, 점수와 ID는 설명용 값이다. 예시만으로 현재 배포 상태나 실제 학습 결과를 판단하지 않는다.

## 현행 요청과 응답

| 파일·경로 | 용도 |
| --- | --- |
| [inline 요청](escalation-feedback-inline-v1/escalate-request.example.json) | POST /escalate의 현행 feedback-inline-v1 body |
| [접수 응답](retraining-api-v1/escalate-accepted.example.json) | 202, run/version 식별자와 statusUrl |
| [실행 중](retraining-api-v1/retraining-running.example.json) | status·단계·집계·heartbeat schema |
| [성공](retraining-api-v1/retraining-succeeded.example.json) | 승격 결과·activation schema |
| [거부](retraining-api-v1/retraining-rejected.example.json) | gate 거부 terminal schema |
| [기술 실패](retraining-api-v1/retraining-failed.example.json) | 실패 terminal schema |
| [취소](retraining-api-v1/retraining-cancelled.example.json) | 취소 terminal schema |
| [현재 모델](retraining-api-v1/current-model.example.json) | desired/serving/previousKnownGood |
| [정책 조회](retraining-api-v1/version-policy-get.example.json) | 개발자 version policy response |
| [정책 예약](retraining-api-v1/version-policy-put.example.json) | mode·activation·expectedRevision·reason을 모두 포함한 PUT |
| [OpenAPI YAML](../backend-api-v1/openapi.yaml) / [JSON](../backend-api-v1/openapi.json) | Swagger·client 생성·contract test용 전달 snapshot |

실제 요청은 eventId/requestId, 질문, 최초·최종 model/label, routerModelVersion과 UTC 시각을 채운다. PII 제거·상향 성공·만족 조건을 실제로 확인하고 boolean true를 설정한다. Spring은 batchId와 원 event snapshot을 보존하고 재시도 때 내용을 바꾸지 않는다. JSON을 GCS에 직접 업로드하지 않는다.

상태 fixture는 schema 예시다. 현재 adapter가 모든 세부 status·failureCode·관측 보장을 실제 생성하는 것은 아니므로 [구현 범위](../api-contract.md#implementation)를 함께 확인한다. 보존한 OpenAPI는 v1 전달 snapshot이며 새 `/router/classify/v2`를 포함하지 않는다. v2 client·contract 생성은 live `/openapi.json`과 [v2 계약](../api-contract.md#routing-policy-v2)을 사용한다. OpenAPI의 host는 환경별 값으로 설정한다. pipelineRunId는 조회 URL이 아니며 statusUrl만 같은 FastAPI base URL에 resolve한다. Developer API는 Spring client 범위에서 제외한다.

## 보존한 과거 fixture

[escalation-feedback-v0.1](escalation-feedback-v0.1/)은 Git 이력 및 호환성 테스트용 자료다. **현재 POST /escalate 요청 예시로 사용하지 않는다.**

| 파일 | 보존 의미 |
| --- | --- |
| [escalate-request.example.json](escalation-feedback-v0.1/escalate-request.example.json) | Spring이 GCS URI와 eventCount를 전송하던 폐기 HTTP 계약 |
| [manifest.example.json](escalation-feedback-v0.1/manifest.example.json) | 내부 artifact와 이름이 같은 v0.1 구조 예시; sha256/URI는 실제 파일 검증값이 아님 |
| [event-row.example.json](escalation-feedback-v0.1/event-row.example.json) | snake_case Parquet 한 행의 논리적 JSON 표현; 현행 camelCase HTTP event와 다름 |

`feedback-export-v0.1`이라는 **내부 ML artifact schema**는 FastAPI가 생성하는 형태로 여전히 사용한다. 폐기된 것은 Spring의 GCS 업로드·URI 전달 HTTP 방식이다. 이 전환과 과거 문서 원문은 [deprecated settings](../../../LLM_Router/docs/deprecated-settings.md)에 압축해 기록했다. 기존 JSON/OpenAPI 파일은 보존하고 안내 문서는 이 파일로 통합한다.
