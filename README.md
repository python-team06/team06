# Team06 GitHub Convention

## 1. 브랜치 운영 방식

프로젝트는 `main`, `dev`, 작업 브랜치로 구분하여 운영합니다.

```text
main
└── dev
    ├── feat/login
    ├── feat/data-analysis
    ├── fix/csv-error
    └── docs/readme
```

- `main`: 최종 제출 및 배포용 브랜치
- `dev`: 개발 내용을 통합하는 브랜치
- 작업 브랜치: 기능 개발 및 수정 작업을 진행하는 브랜치

작업 브랜치는 반드시 최신 `dev` 브랜치를 기준으로 생성합니다.

---

## 2. 작업 흐름

```text
dev 브랜치 최신화
→ dev 기준 작업 브랜치 생성
→ 기능 개발
→ 작업 브랜치에 push
→ 작업 브랜치에서 dev로 Pull Request
→ 코드 리뷰 후 dev에 merge
→ 개발 완료 후 dev에서 main으로 Pull Request
→ 최종 리뷰 후 main에 merge
```

`main`과 `dev` 브랜치에는 직접 작업하지 않습니다.

---

## 3. 브랜치 생성 방법

### dev 브랜치 이동 및 최신화

```bash
git switch dev
git pull origin dev
```

### dev 기준 작업 브랜치 생성

```bash
git switch -c feat/login
```

기존 명령어를 사용할 경우:

```bash
git checkout dev
git pull origin dev
git checkout -b feat/login
```

---

## 4. 브랜치 이름 규칙

브랜치 이름은 다음 형식을 사용합니다.

```text
브랜치카테고리/기능명
```

예시:

```text
feat/login
feat/data-analysis
fix/csv-save
refactor/data-processing
docs/readme
test/data-validation
```

### 브랜치 카테고리

| 카테고리 | 설명 |
|---|---|
| `feat` | 새로운 기능 추가 |
| `fix` | 오류 및 버그 수정 |
| `refactor` | 기능 변경 없이 코드 구조 개선 |
| `docs` | README 및 문서 수정 |
| `test` | 테스트 코드 추가 및 수정 |
| `chore` | 설정 파일, 패키지 등 기타 작업 |

브랜치 이름은 영문 소문자와 하이픈을 사용합니다.

```text
feat/data-analysis   ✅
feat/data_analysis   ❌
Feat/DataAnalysis    ❌
```

---

## 5. 커밋 메시지 규칙

커밋 메시지는 다음 형식을 사용합니다.

```text
카테고리: 작업 내용
```

예시:

```text
feat: CSV 데이터 병합 기능 추가
fix: 저장된 CSV 컬럼 개수 출력 오류 수정
refactor: 중복된 데이터 로딩 코드 제거
docs: GitHub 컨벤션 작성
test: 데이터 병합 테스트 추가
chore: gitignore 설정 추가
```

---

## 6. Pull Request 규칙

### 작업 브랜치 PR

작업 브랜치는 `dev` 브랜치로 Pull Request를 생성합니다.

```text
feat/login → dev
fix/csv-error → dev
```

### 최종 통합 PR

개발이 완료되면 `dev` 브랜치에서 `main` 브랜치로 Pull Request를 생성합니다.

```text
dev → main
```

### PR 제목

```text
[카테고리] 작업 내용
```

예시:

```text
[Feat] CSV 데이터 병합 기능 추가
[Fix] 컬럼 개수 출력 오류 수정
[Docs] README 및 GitHub 컨벤션 작성
```

### PR 내용 예시

```markdown
## 작업 내용

- CSV 파일 불러오기
- qname 기준 컬럼 연결
- 병합 결과 CSV 저장
- 생성된 CSV 컬럼 개수 출력

## 확인 사항

- [ ] 코드가 정상적으로 실행되는지 확인
- [ ] 불필요한 파일이 포함되지 않았는지 확인
- [ ] 대용량 CSV가 Git에 포함되지 않았는지 확인
```

---

## 7. 주의사항

- `main` 브랜치에 직접 push하지 않습니다.
- `dev` 브랜치에도 직접 기능 코드를 push하지 않습니다.
- 작업 시작 전에 반드시 `dev` 브랜치를 최신 상태로 업데이트합니다.
- 하나의 브랜치에서는 하나의 기능만 작업합니다.
- 대용량 데이터 파일은 GitHub에 올리지 않습니다.
- 개인 환경 파일과 생성 결과 파일은 `.gitignore`에 등록합니다.

```gitignore
.vrmachine/
results.csv
matched_results.csv
```
