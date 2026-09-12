# Backend API v1 전달 파일

Spring 팀은 다음 파일을 기준으로 구현한다.

- `openapi.yaml`: Swagger UI, OpenAPI Generator와 contract test용 계약
- `openapi.json`: YAML과 같은 내용을 담은 JSON 계약
- `../backend-integration-handoff.md`: 책임 경계, DTO, 저장값, 오류·재시도와 polling 설명
- `../examples/escalation-feedback-inline-v1/`: `POST /escalate` request fixture
- `../examples/retraining-api-v1/`: 재학습 상태·현재 model·개발자 policy response fixture

Spring은 OpenAPI의 `Routing`, `Health`, `Retraining`, `Model` tag를 사용한다.
`Developer` tag는 별도 개발자 credential용이므로 Spring client 생성 대상에서 제외한다.

OpenAPI server URL의 `host` 변수는 환경별 FastAPI 주소로 바꾼다.
`pipelineRunId`는 HTTP URL이 아니며 `statusUrl`만 FastAPI base URL에 resolve한다.
