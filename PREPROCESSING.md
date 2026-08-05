# SO Developer Survey 2024 — 전처리 파이프라인

Stack Overflow Developer Survey 2024 응답 원본(`results.csv`)을
통계·ML 분석이 바로 가능한 형태(parquet)로 변환한다.
(팀 GitHub 규칙은 `README.md` 참조)

## 구조

```
├── src/
│   ├── config.py                   경로·상수
│   └── preprocess.py               전처리 0~2단계
└── data/
    ├── results.csv.gz              응답 원본 압축본 (65,437행 × 114열)
    ├── schema.csv                  코드북 (87행 × 6열)
    ├── so2024_dictionary.csv       컬럼 -> qid/질문 원문 대응표 (silver 컬럼명 포함)
    ├── silver.parquet              [1단계] 구조 정리본 (65,437 × 115)
    ├── gold_wide.parquet           [2단계] ML용 멀티핫 행렬 (65,437 × 1,057)
    └── gold_long.parquet           [2단계] 통계용 세로 형태 (약 680만 행)
```

원본은 GitHub 파일당 100MB 제한 때문에 gzip(17MB)으로 올려두었다.
**pandas 가 `.gz` 를 그대로 읽으므로 압축을 풀 필요 없다** — `src/config.py` 가
`results.csv` 가 없으면 자동으로 `results.csv.gz` 를 쓴다.

## 실행

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.preprocess    # 약 15초, data/ 에 parquet 3개 생성
```

`-m` 으로 실행해야 `src.config` import 가 걸린다.

## 전처리 단계 (`src/preprocess.py`)

| 단계 | 내용 | 산출물 |
|---|---|---|
| 0 | `"NA"` 문자열만 결측으로 읽기 (`keep_default_na=False, na_values=["NA"]`) | — |
| 1 | `Check`/`CompTotal`/`Currency` 드롭 · 공백 컬럼명 10개 rename · `YearsCode*` sentinel 숫자화 · `Age`/`EdLevel`/`OrgSize`/`SOVisitFreq` 순서형 `*Num` 추가 | `silver.parquet` |
| 2 | 다중선택 57개 → 멀티핫 0/1 + `__answered` 플래그 (wide) / 응답자×선택지 세로 형태 (long) | `gold_wide.parquet`, `gold_long.parquet` |
| 3 | AISelect 예측용: 무응답 4,530명 제외 · 누수 컬럼 가족 185개 제거 · 층화 분할 8:2 (seed=42) — `python -m src.split_aiselect` | `ai_train.parquet`, `ai_test.parquet`, `ai_split_ids.csv`, `ai_split_manifest.json` |
| 4 | **선형 트랙** 변환: 윈저라이징(comp, train p1/p99) · log1p 13개 · zero-inflated `_any` 플래그 4개 · **중앙값 대체 + `_missing` 플래그 18개** — `python -m src.transform_linear` | `ai_train_linear.parquet`, `ai_test_linear.parquet`, `linear_transform_params.json` |
| 5 | **베이스라인 모델**: 로지스틱 회귀 (스케일링 + 원핫 min_frequency=30, 5-fold CV 후 test 1회) — `python -m src.train_logistic` | `model_logistic.joblib`, `logistic_metrics.json` |

설계 근거 (전수 검증 완료):
- `*Admired` 11개 = `Have ∩ Want` 와 100% 일치하는 파생 컬럼 → wide 에서 제외, long 에는 유지
- 그리드형 컬럼의 `""` = "답했지만 빈 버킷", `"NA"` = "질문 안 봄" → `__answered` 는 `"NA"` 만 미응답 처리

### 3단계: AISelect 예측용 분할 (`ai_train` / `ai_test`)

- 타깃: `target = 1 if AISelect == "Yes" else 0` (train/test 양성 비율 61.8% 동일 — 층화)
- **누수 제거**: AISelect 값에 따라 조건부로 노출되는 꼬리질문 15개
  (Yes 전용 11 + Yes·도입계획 공용 4) + 타깃과 동어반복인 `AISearchDev` 2개
  → wide 컬럼 가족 기준 185개 제거. 근거 실측치는 `src/split_aiselect.py` docstring 참조
- **팀 규칙** (자세한 내용은 `data/ai_split_manifest.json`):
  - `ai_test` 는 최종 평가 **1회만** 사용 — 모델·피처 선택은 train 내부 CV 로
  - target encoding·스케일링 등 fitted 변환은 **train 에서만 fit**
  - `ResponseId` 는 인덱스(식별자) — 피처로 사용 금지
  - silver/long 으로 작업할 때는 `ai_split_ids.csv` 로 같은 분할을 재현

### 4단계: 어느 트랙의 파일을 쓰나 (모델별)

| 모델 | 파일 | 이유 |
|---|---|---|
| LightGBM 등 트리 | `ai_train.parquet` / `ai_test.parquet` **그대로** | 트리는 단조변환에 불변, NaN 네이티브 처리 — 변환 불필요 |
| 로지스틱·statsmodels 등 선형 | `ai_train_linear.parquet` / `ai_test_linear.parquet` | 왜도 교정(log1p)·윈저라이징·중앙값 대체 완료 |

선형 파일의 규약:
- `<이름>_log` = log1p 변환본 (원본 수치 컬럼은 제거됨 — 중복 사용 방지)
- `<이름>_missing` = 원래 결측이었는지 (결측률이 클래스와 상관하므로 신호 보존)
- `<이름>_any` = zero-inflated 4개(JobSatPoints_4/5/10/11)의 "0 초과" 여부
- 윈저 경계·중앙값은 **train 에서만 fit** — 값은 `linear_transform_params.json` 참조
- **스케일링(StandardScaler)은 안 되어 있음** — 모델 파이프라인(CV 내부)에서 할 것
- category 컬럼의 결측은 그대로 NaN — 원핫 시 `dummy_na=True` 등으로 처리

## 어떤 파일을 쓰면 되나

| 용도 | 파일 | 예 |
|---|---|---|
| 모델 학습 (X 행렬) | `gold_wide.parquet` | `LanguageHaveWorkedWith__Python` 등 0/1 |
| 집계·차트 | `gold_long.parquet` | 아래 사용 예 참고 |
| 응답자 단위 작업·파생 컬럼 추가 | `silver.parquet` | 1행 = 응답자 1명 |
| "이 컬럼 무슨 질문?" | `so2024_dictionary.csv` | 컬럼명 → qid → 질문 원문 |

```python
import pandas as pd

# 사용 언어 Top 10
long = pd.read_parquet("data/gold_long.parquet")
long[long.question == "LanguageHaveWorkedWith"].value.value_counts().head(10)

# 컬럼이 어떤 질문이었는지 찾기
dic = pd.read_csv("data/so2024_dictionary.csv")
dic[dic.column_silver == "AIToolCurrentlyUsing"][["qid", "question"]]
```

## 데이터를 다루기 전에 알아둘 것

- 결측이 빈칸이 아니라 **`"NA"` 문자열**이다 (원본 기준). silver 이후로는 진짜 결측(NaN)으로 변환돼 있다.
- 다중선택 컬럼은 한 칸에 `"Python;SQL;Go"` 형태 — silver 까지는 그대로, gold 에서 분해된다.
- `ConvertedCompYearly`(USD 연봉)는 응답률 35.8% — 보수 분석은 사실상 1/3 표본이다.
- 원본 비압축 CSV(159MB)가 따로 필요하면:

```bash
BASE=https://media.githubusercontent.com/media/StackExchange/Survey/refs/heads/main/packages/archive/2024
curl -L "$BASE/results.csv" -o data/results.csv
```
