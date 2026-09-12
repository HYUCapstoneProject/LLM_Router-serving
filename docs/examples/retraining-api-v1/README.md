# Retraining API v1 JSON fixture

이 디렉터리의 JSON은 Spring/FastAPI contract test에서 같은 파일을 읽기 위한 목표 계약
fixture다. 현재 배포본의 응답이 아니라 구현 후 제공할 schema를 나타낸다.

| 파일 | 용도 |
| --- | --- |
| `escalate-accepted.example.json` | `POST /escalate` 202 응답 |
| `retraining-running.example.json` | status 실행 중 응답 |
| `retraining-succeeded.example.json` | 승격 및 FastAPI 반영 완료 응답 |
| `retraining-rejected.example.json` | deploy gate 거부 terminal 응답 |
| `retraining-failed.example.json` | 기술 실패 terminal 응답 |
| `retraining-cancelled.example.json` | 취소 terminal 응답 |
| `current-model.example.json` | 현재 목표/serving model 응답 |
| `version-policy-get.example.json` | 개발자 정책 조회 응답 |
| `version-policy-put.example.json` | 개발자 정책 변경 요청 |

정확한 필드 의미와 enum은 [`../../api-contract.md`](../../api-contract.md)를 따른다.
