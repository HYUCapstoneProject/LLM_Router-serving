# FastAPI 소스 코드 GCS 업로드 가이드

이 문서는 `LLM_Router-serving` 저장소의 소스 코드를 안전하게 압축하여 Google Cloud
Storage(GCS)에 올리는 방법을 설명한다. 처음 Google Cloud를 사용하는 고등학생도 따라 할
수 있도록 용어부터 확인한다.

## 0. 가장 먼저 알아둘 점

> **GCS에 코드를 올리는 것만으로 FastAPI 서버가 실행되지는 않는다.**

GCS는 파일을 보관하는 인터넷 저장소다. USB 메모리나 Google Drive와 비슷한 역할을
하지만, 서버와 프로그램이 사용하기 좋은 형태로 만들어져 있다.

이 프로젝트에서 서비스가 실행되기까지의 큰 흐름은 다음과 같다.

```text
소스 코드
  -> Docker 이미지 빌드
  -> Artifact Registry에 이미지 저장
  -> GCE, Cloud Run 등의 실행 환경에서 컨테이너 실행
  -> FastAPI 서버 동작
```

따라서 이 문서의 목표는 **소스 코드를 백업하거나 다른 Google Cloud 작업에서 사용할 수
있도록 GCS에 보관하는 것**이다. 실제 서버 배포는 별도의 배포 가이드가 필요하다.

### GCS와 Artifact Registry의 차이

| 저장소 | 보관하는 것 | 이 프로젝트의 예시 |
| --- | --- | --- |
| GCS | 일반 파일과 ML artifact | 소스 압축 파일, `manifest.json`, `events.parquet`, 모델 checkpoint |
| Artifact Registry | 실행 가능한 컨테이너 이미지 | `LLM_Router-serving` Docker 이미지 |

## 1. 이번 작업의 완성 모습

작업을 마치면 GCS에 다음과 같은 파일이 생긴다.

```text
gs://<버킷 이름>/releases/llm-router-serving/20260814-v1/
  llm-router-serving-20260814-v1.tar.gz
```

- `gs://`는 GCS 주소라는 뜻이다.
- 버킷(bucket)은 GCS의 가장 큰 저장 공간 단위다.
- object는 버킷 안에 저장한 파일이다.
- `releases/...`는 파일을 정리하기 위한 경로다. GCS 화면에서는 폴더처럼 보인다.

## 2. 준비물

다음 항목이 필요하다.

1. Google 계정
2. 사용할 Google Cloud 프로젝트
3. 프로젝트에서 GCS를 사용할 수 있는 권한
4. 현재 컴퓨터에 있는 `LLM_Router-serving` 저장소
5. 명령어 방식으로 진행한다면 Google Cloud CLI인 `gcloud`

프로젝트 ID는 Google Cloud Console 상단의 프로젝트 선택 메뉴에서 확인할 수 있다.
프로젝트 이름과 프로젝트 ID는 다를 수 있으므로 **프로젝트 ID**를 복사해야 한다.

예시:

```text
프로젝트 이름: HYUCapstone
프로젝트 ID: project-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

## 3. 절대로 올리면 안 되는 파일

다음 파일은 비밀번호나 불필요한 대용량 데이터를 포함할 수 있으므로 GCS에 올리지 않는다.

| 파일 또는 폴더 | 제외하는 이유 |
| --- | --- |
| `.env` | API token과 실제 환경 설정이 들어갈 수 있음 |
| 서비스 계정 JSON key | Google Cloud 계정 열쇠와 같음 |
| `.venv/` | 다른 컴퓨터에서 재사용하기 어려운 대용량 가상환경 |
| `.git/` | 전체 Git 내부 기록이며 소스 실행에 필요하지 않음 |
| `var/` | SQLite DB 등 실행 중 생성된 상태가 들어갈 수 있음 |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | 다시 만들 수 있는 임시 파일 |

`.env.example`은 실제 비밀값이 아니라 어떤 설정이 필요한지 보여주는 예시 파일이므로
압축 파일에 포함해도 된다.

> 서비스 계정 JSON key를 소스 폴더나 Docker 이미지에 넣지 않는다. 배포 환경에서는
> Google Cloud IAM과 실행 환경의 서비스 계정을 사용한다.

## 4. 권장 방법: Google Cloud CLI로 업로드하기

CLI는 글자로 명령을 입력하는 방식이다. 처음에는 낯설지만 작업 내용을 다시 실행하기 쉽고,
어떤 파일을 올렸는지 확인하기도 편하다.

### 4-1. `gcloud` 설치 확인

터미널에서 다음 명령을 실행한다.

```bash
gcloud --version
```

버전 정보가 출력되면 이미 설치된 것이다. `command not found`가 나오면
[Google Cloud CLI 공식 설치 문서](https://docs.cloud.google.com/sdk/docs/install-sdk)를
따라 설치한다.

### 4-2. Google 계정과 프로젝트 연결

다음 명령을 실행한다.

```bash
gcloud init
```

브라우저가 열리면 사용할 Google 계정으로 로그인하고, 목록에서 프로젝트를 선택한다.

설정 결과는 다음 명령으로 확인한다.

```bash
gcloud auth list
gcloud config get-value project
```

- `gcloud auth list`: 현재 로그인한 계정을 보여준다.
- `gcloud config get-value project`: 현재 선택된 프로젝트 ID를 보여준다.

잘못된 프로젝트가 선택되었다면 다음처럼 바꾼다. 아래 값은 실제 프로젝트 ID로 교체한다.

```bash
gcloud config set project "<프로젝트 ID>"
```

### 4-3. 사용할 값을 정하기

아래 세 값을 메모장에 먼저 적는다.

```text
PROJECT_ID=<Google Cloud 프로젝트 ID>
BUCKET_NAME=<전 세계에서 중복되지 않는 버킷 이름>
REGION=<GKE와 Vertex AI가 사용하는 리전>
```

이후 명령을 짧게 쓰기 위해 터미널 변수로 등록한다.

```bash
PROJECT_ID="<프로젝트 ID>"
BUCKET_NAME="<고유한 버킷 이름>"
REGION="us-west1"
RELEASE_ID="20260814-v1"
```

주의할 점:

- `<`와 `>`까지 입력하지 말고 안쪽 내용을 실제 값으로 교체한다.
- 버킷 이름에는 소문자, 숫자와 하이픈을 사용하는 것이 가장 간단하다.
- 버킷 이름은 전 세계 모든 GCS 사용자 사이에서 중복되면 안 된다.
- 버킷 이름은 외부에서 존재 여부를 확인할 수 있으므로 이름, 이메일, 학번 등의 개인정보를
  넣지 않는다.
- `us-west1`은 예시다. 실제로는 GKE, Vertex AI, 모델 artifact와 같은 리전을 사용한다.
  리전이 다르면 속도가 느려지거나 네트워크 비용이 생길 수 있다.
- `RELEASE_ID`는 업로드 버전을 구분하는 이름이다. 새 버전을 올릴 때마다 변경한다.

입력한 값은 다음 명령으로 확인할 수 있다.

```bash
echo "$PROJECT_ID"
echo "$BUCKET_NAME"
echo "$REGION"
echo "$RELEASE_ID"
```

### 4-4. 프로젝트를 다시 한번 확인하기

```bash
gcloud config set project "$PROJECT_ID"
gcloud config get-value project
```

두 번째 명령의 출력이 `PROJECT_ID`와 같아야 한다. 학교 프로젝트와 개인 프로젝트를
혼동하면 다른 프로젝트에 비용이 발생할 수 있으므로 이 단계를 건너뛰지 않는다.

### 4-5. GCS 버킷 만들기

버킷이 아직 없다면 다음 명령으로 만든다.

```bash
gcloud storage buckets create "gs://$BUCKET_NAME" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --uniform-bucket-level-access \
  --public-access-prevention
```

각 옵션의 의미는 다음과 같다.

| 옵션 | 의미 |
| --- | --- |
| `--project` | 버킷 비용과 권한을 관리할 Google Cloud 프로젝트 |
| `--location` | 실제 파일이 저장될 지역 |
| `--uniform-bucket-level-access` | 파일별 ACL 대신 IAM 한 방식으로 권한을 관리 |
| `--public-access-prevention` | 인터넷 전체 공개를 막음 |

`The requested bucket name is not available`과 비슷한 오류가 나오면 다른 사람이 이미 같은
이름을 사용 중이다. `BUCKET_NAME`을 바꾸고 다시 실행한다.

이미 팀 버킷이 있다면 새 버킷을 만들지 말고 관리자에게 정확한 버킷 이름과 업로드 경로를
확인한다.

버킷이 생성되었는지 확인한다.

```bash
gcloud storage buckets describe "gs://$BUCKET_NAME"
```

### 4-6. 저장소 폴더로 이동하기

```bash
cd /Users/yoocheolmin/workspace/school/LLM_Router-serving
pwd
```

`pwd` 결과가 다음과 같아야 한다.

```text
/Users/yoocheolmin/workspace/school/LLM_Router-serving
```

다음 파일들이 보이는지도 확인한다.

```bash
ls
```

최소한 `Dockerfile`, `README.md`, `pyproject.toml`, `src`가 보여야 한다. 그렇지 않으면 잘못된
폴더에 있으므로 압축을 시작하지 않는다.

### 4-7. 비밀값을 제외하고 소스 코드 압축하기

다음 명령은 업로드용 압축 파일을 `/tmp`에 만든다.

```bash
ARCHIVE_PATH="/tmp/llm-router-serving-${RELEASE_ID}.tar.gz"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.env' \
  --exclude='var' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='*.egg-info' \
  --exclude='build' \
  --exclude='dist' \
  -czf "$ARCHIVE_PATH" .
```

`/tmp`는 임시 파일을 두는 운영체제 폴더다. 프로젝트 폴더 안에 압축 파일을 만들면 그
압축 파일이 다시 자기 자신 안에 포함되는 실수가 생길 수 있으므로 `/tmp`를 사용한다.

압축 파일이 만들어졌는지 확인한다.

```bash
ls -lh "$ARCHIVE_PATH"
```

### 4-8. 업로드 전에 압축 내용 검사하기

먼저 압축 파일 안의 목록을 일부 확인한다.

```bash
tar -tzf "$ARCHIVE_PATH" | sed -n '1,80p'
```

다음으로 위험한 이름이 들어갔는지 검사한다.

```bash
tar -tzf "$ARCHIVE_PATH" | grep -E '(^|/)\.env$|\.pem$|\.key$|service-account|credentials.*\.json$'
```

- **아무것도 출력되지 않으면 정상**이다.
- 파일 이름이 출력되면 업로드하지 말고 해당 파일이 왜 포함되었는지 먼저 확인한다.
- `.env.example`은 위 검사 대상이 아니며 포함되어도 된다.

### 4-9. GCS에 업로드하기

버전별 경로를 만든 뒤 압축 파일을 올린다.

```bash
GCS_RELEASE_URI="gs://$BUCKET_NAME/releases/llm-router-serving/$RELEASE_ID"

gcloud storage cp "$ARCHIVE_PATH" "$GCS_RELEASE_URI/"
```

`Copying`과 완료 메시지가 출력되면 업로드가 끝난 것이다.

### 4-10. 업로드 결과 확인하기

```bash
gcloud storage ls -l "$GCS_RELEASE_URI/"
```

다음과 비슷한 파일 이름이 보여야 한다.

```text
llm-router-serving-20260814-v1.tar.gz
```

Google Cloud Console에서도 확인할 수 있다.

1. [Google Cloud Console](https://console.cloud.google.com/)을 연다.
2. 올바른 프로젝트가 선택되어 있는지 확인한다.
3. 왼쪽 메뉴에서 **Cloud Storage → Buckets**를 연다.
4. `$BUCKET_NAME`에 해당하는 버킷을 누른다.
5. `releases → llm-router-serving → 20260814-v1` 순서로 들어간다.

## 5. 다운로드까지 시험해 보기

업로드 목록만 보지 말고 다시 다운로드되는지도 확인하면 실수를 더 빨리 발견할 수 있다.

```bash
DOWNLOADED_PATH="/tmp/downloaded-llm-router-serving-${RELEASE_ID}.tar.gz"

gcloud storage cp \
  "$GCS_RELEASE_URI/llm-router-serving-${RELEASE_ID}.tar.gz" \
  "$DOWNLOADED_PATH"
```

원본과 다운로드 파일의 SHA-256을 확인한다.

```bash
shasum -a 256 "$ARCHIVE_PATH"
shasum -a 256 "$DOWNLOADED_PATH"
```

두 줄 앞부분의 긴 문자열이 완전히 같으면 업로드와 다운로드 과정에서 파일이 변하지 않은
것이다.

압축이 정상적으로 풀리는지도 별도의 임시 폴더에서 시험할 수 있다.

```bash
RESTORE_DIR="/tmp/llm-router-serving-restore-${RELEASE_ID}"
mkdir -p "$RESTORE_DIR"
tar -xzf "$DOWNLOADED_PATH" -C "$RESTORE_DIR"
ls "$RESTORE_DIR"
```

## 6. 웹 화면만 사용해서 업로드하는 방법

명령어 사용이 어려우면 버킷 생성과 업로드를 Google Cloud Console에서 할 수도 있다.
하지만 비밀 파일을 실수로 선택하지 않도록 **4-7에서 만든 압축 파일 하나만** 업로드하는
것이 안전하다.

### 6-1. 버킷 만들기

1. [Google Cloud Console](https://console.cloud.google.com/)에 로그인한다.
2. 화면 상단에서 올바른 프로젝트를 선택한다.
3. **Cloud Storage → Buckets**로 이동한다.
4. **Create** 또는 **버킷 만들기**를 누른다.
5. 전 세계에서 고유한 버킷 이름을 입력한다.
6. GKE 및 Vertex AI와 같은 리전을 선택한다.
7. Access control은 **Uniform**을 선택한다.
8. Public access prevention은 활성화한다.
9. 설정을 다시 확인하고 버킷을 만든다.

### 6-2. 압축 파일 올리기

1. 만든 버킷을 연다.
2. `releases/llm-router-serving/<RELEASE_ID>` 구조로 폴더를 만든다.
3. **Upload files**를 누른다.
4. `/tmp/llm-router-serving-<RELEASE_ID>.tar.gz`를 선택한다.
5. 업로드가 끝날 때까지 창을 닫지 않는다.

macOS Finder에서 `/tmp`를 열려면 **이동 → 폴더로 이동**을 누르고 `/tmp`를 입력한다.

## 7. 새 버전을 다시 올리는 방법

기존 파일을 덮어쓰지 말고 `RELEASE_ID`를 변경하여 새 경로에 올린다.

```bash
RELEASE_ID="20260815-v2"
```

그다음 4-7부터 다시 진행한다. 버전별 경로를 유지하면 문제가 발생했을 때 이전 소스를
찾기 쉽다.

권장 구조:

```text
gs://<버킷>/releases/llm-router-serving/20260814-v1/...
gs://<버킷>/releases/llm-router-serving/20260815-v2/...
```

가능하면 `RELEASE_ID`와 함께 Git commit ID도 기록한다. 예를 들어 릴리스 메모에 다음
명령의 결과를 적는다.

```bash
git rev-parse HEAD
```

## 8. 팀원에게 권한을 줄 때

버킷 전체를 인터넷에 공개하지 않는다. Google Cloud IAM으로 필요한 사람에게만 권한을
준다.

| 필요한 작업 | 권장 역할 |
| --- | --- |
| 파일 읽기만 필요 | Storage Object Viewer |
| 파일 업로드·수정·삭제 필요 | Storage Object User 또는 Storage Object Admin |
| 버킷 설정까지 관리 | Storage Admin |

항상 필요한 최소 권한만 부여한다. 단순 업로드를 위해 프로젝트 전체 Owner 권한을 주면
안 된다.

## 9. 자주 발생하는 오류

### `gcloud: command not found`

Google Cloud CLI가 설치되지 않았거나 터미널이 설치 사실을 아직 모르는 상태다. CLI를
설치한 뒤 터미널을 완전히 닫고 다시 연다.

### `You do not currently have an active account selected`

다음 명령으로 다시 로그인한다.

```bash
gcloud auth login
```

### `The requested bucket name is not available`

버킷 이름이 이미 사용 중이다. 개인정보를 넣지 않은 다른 고유한 이름으로 변경한다.

### `403 Forbidden` 또는 `permission denied`

현재 계정에 버킷 생성 또는 object 업로드 권한이 없다. 다음을 확인한다.

```bash
gcloud auth list
gcloud config get-value project
```

계정과 프로젝트가 맞다면 프로젝트 관리자에게 필요한 GCS IAM 역할을 요청한다.

### `BucketNotFoundException` 또는 `404`

`BUCKET_NAME` 철자가 맞는지 확인한다.

```bash
echo "$BUCKET_NAME"
gcloud storage buckets list --project="$PROJECT_ID"
```

### 압축 파일이 지나치게 크다

`.venv`, `.git`, 모델 checkpoint 또는 cache가 포함되었을 가능성이 높다. 업로드를 멈추고
4-7의 제외 목록을 다시 확인한다.

## 10. 이 프로젝트에서 GCS에 보관할 것과 보관하지 않을 것

| 항목 | GCS 사용 여부 | 이유 |
| --- | --- | --- |
| 소스 압축 파일 | 선택 사항 | 백업과 전달에 사용 가능 |
| Feedback `manifest.json` | 사용 | 배치 metadata와 무결성 정보 |
| Feedback `events.parquet` | 사용 | 실제 재학습 feedback 데이터 |
| 학습·평가 dataset과 config | 사용 | KFP component가 읽는 versioned artifact |
| 승인된 모델 package/checkpoint | 사용 | FastAPI가 적재할 ML artifact |
| Docker 이미지 | 사용하지 않음 | Artifact Registry에 저장 |
| `.env`, token, JSON key | 사용하지 않음 | Secret Manager와 IAM을 사용 |
| 실행 중 SQLite DB | 소스 archive에 포함하지 않음 | 배포 시 별도의 영속 저장소 필요 |

### 10-1. 새 버킷을 feedback과 모델 artifact에도 사용할 경우

이 문서에서 올린 소스 압축 파일은 백업용이므로 FastAPI가 자동으로 읽지 않는다. 같은
버킷을 `/escalate` feedback, dataset, config와 모델 artifact에도 사용할 계획이라면 실제
배포 환경의 `.env`에서 버킷 이름을 맞춰야 한다.

최소한 다음 값을 확인한다.

```dotenv
LLM_ROUTER_ALLOWED_GCS_BUCKETS=<실제 버킷 이름>
```

그리고 `.env.example`에 있는 다음 종류의 URI도 실제 버킷과 object 경로로 변경한다.

```text
LLM_ROUTER_BASE_TRAINING_DATASET_URI
LLM_ROUTER_FEEDBACK_PREPROCESSING_CONFIG_URI
LLM_ROUTER_TRAINING_CONFIG_URI
LLM_ROUTER_ROUTERARENA_DATASET_URI
LLM_ROUTER_EVALUATION_CONFIG_URI
LLM_ROUTER_PACKAGE_METADATA_URI
LLM_ROUTER_BASELINE_METRICS_URI
LLM_ROUTER_GATE_RULES_URI
```

`LLM_ROUTER_ALLOWED_GCS_BUCKETS`에 없는 버킷의 `manifestUri` 또는 `eventsUri`가
`POST /escalate`로 들어오면 FastAPI가 보안상 거부한다. `.env.example`에는 형식만 남기고,
실제 값과 secret은 Git이나 GCS 소스 압축 파일에 포함하지 않는다.

### 10-2. 비용 주의

GCS는 저장 용량, 요청 횟수와 리전 간 데이터 이동 등에 따라 비용이 발생할 수 있다.
불필요하게 `.venv`, Git 기록, 중복 모델을 올리지 말고 기존 프로젝트의 budget alert와
결제 상태를 확인한다. 이 문서의 명령을 실행하기 전에 팀에서 사용할 버킷과 리전을 먼저
합의하는 것이 가장 안전하다.

## 11. 실제 FastAPI 실행을 위한 다음 단계

GCS 업로드를 완료한 뒤 서버를 실행하려면 다음 작업이 추가로 필요하다.

1. 저장소의 `Dockerfile`로 컨테이너 이미지를 만든다.
2. 이미지를 Artifact Registry에 올린다.
3. GCE, Cloud Run 또는 합의된 실행 환경에서 이미지를 실행한다.
4. `.env`의 값을 Secret Manager와 배포 환경 변수로 안전하게 전달한다.
5. BGE와 승인된 ModernBERT 모델 경로를 연결한다.
6. 영속적인 `LLM_ROUTER_RETRAINING_DB_PATH`를 연결한다.
7. 실행 환경의 서비스 계정에 필요한 GCS와 Vertex AI 권한만 부여한다.
8. `/health`가 HTTP 200을 반환하는지 확인한 뒤 Spring Server와 연결한다.

Cloud Run에 소스에서 직접 배포하는 기능도 있지만, 내부적으로 Cloud Build가 컨테이너를
만들고 Artifact Registry에 저장한다. 즉 이 경우에도 GCS 자체가 FastAPI를 실행하는 것은
아니다. 또한 현재 서비스는 큰 ML 모델, 긴 모델 로딩 시간, 영속 SQLite와 내부 Vertex AI
연결을 사용하므로 실제 실행 환경은 자원과 네트워크 조건을 검토한 뒤 결정해야 한다.

## 12. 완료 체크리스트

- [ ] 올바른 Google Cloud 계정과 프로젝트를 선택했다.
- [ ] 버킷 리전이 GKE 및 Vertex AI 리전과 일치한다.
- [ ] 버킷의 public access prevention이 활성화되어 있다.
- [ ] `.env`, JSON key, `.venv`, `.git`, `var`를 압축에서 제외했다.
- [ ] 버전이 포함된 새 경로에 압축 파일을 업로드했다.
- [ ] `gcloud storage ls` 또는 Console에서 파일을 확인했다.
- [ ] 다시 다운로드하여 SHA-256이 일치하는지 확인했다.
- [ ] GCS 업로드와 실제 FastAPI 배포가 다른 작업임을 이해했다.

## 13. 공식 참고 문서

- [Google Cloud CLI 설치](https://docs.cloud.google.com/sdk/docs/install-sdk)
- [`gcloud` 초기 설정](https://docs.cloud.google.com/sdk/docs/initialize)
- [GCS 버킷 만들기](https://docs.cloud.google.com/storage/docs/creating-buckets)
- [파일 시스템에서 object 업로드](https://docs.cloud.google.com/storage/docs/uploading-objects)
- [`gcloud storage cp` 명령](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp)
- [Cloud Storage IAM 역할](https://docs.cloud.google.com/storage/docs/access-control/iam-roles)
- [Cloud Run 소스 코드 배포](https://docs.cloud.google.com/run/docs/deploying-source-code)
