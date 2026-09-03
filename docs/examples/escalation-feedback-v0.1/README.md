# Escalation feedback v0.1 예시 파일

이 디렉터리는 현재 외부 배포본과의 호환성 확인용 legacy fixture다. 새 Spring 구현은
[`../escalation-feedback-inline-v1/`](../escalation-feedback-inline-v1/)을 사용한다.

| 파일 | 용도 | 그대로 GCS에 업로드하는가? |
| --- | --- | --- |
| `escalate-request.example.json` | `POST /escalate` HTTP body 예시 | 아니오 |
| `manifest.example.json` | GCS에 게시할 `manifest.json` 예시 | 값을 교체한 뒤 예 |
| `event-row.example.json` | Spring DTO와 Parquet 한 행의 논리적 구조 예시 | 아니오 |

`event-row.example.json`을 배열이나 JSONL로 만들어 GCS에 올리면 안 된다. 실제 이벤트
실제 batch는 1건 이상이며 계약의 12개 컬럼과 타입을 사용해 `events.parquet`으로 직렬화한다.

사용 전에 세 파일의 `batchId`/`batch_id`, 날짜, URI를 실제 batch 값으로 바꾼다.
`manifest.example.json`의 `sha256`은 형식만 보여 주는 예시 값이므로, 완성된
`events.parquet` 파일 바이트에서 계산한 SHA-256으로 반드시 교체한다.

상세 계약은 [`../../escalation-feedback-batch-v0.1.md`](../../escalation-feedback-batch-v0.1.md)를
참고한다.
