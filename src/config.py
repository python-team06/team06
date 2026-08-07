"""전처리 파이프라인 전역 설정.

경로·상수처럼 여러 곳에서 함께 쓰는 값을 한곳에 모아둔다.
스크립트마다 경로를 하드코딩하면 데이터 위치가 바뀔 때 전부 고쳐야 하므로,
이런 값은 설정 모듈로 분리한다.
"""

from pathlib import Path

# ---------- 경로 ----------
# __file__ -> src/config.py, parent -> src/, 한 번 더 parent -> 프로젝트 루트.
# 이렇게 하면 어느 디렉터리에서 실행하든 경로가 깨지지 않는다.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# 원본 응답 데이터 (Stack Overflow Developer Survey 2024, 65,437행 × 114열).
# 저장소에는 GitHub 파일당 100MB 제한 때문에 gzip 압축본(17MB)으로 올라가 있다.
# pandas 는 .gz 를 그대로 읽으므로 압축을 풀 필요가 없다 — 압축 안 푼 클론 환경에서는
# 자동으로 압축본을 쓴다.
RESULTS_CSV = DATA_DIR / "results.csv"
if not RESULTS_CSV.exists():
    RESULTS_CSV = DATA_DIR / "results.csv.gz"

# 코드북 원본 (87행 × 6열): 컬럼명이 실제로 어떤 질문이었는지
SCHEMA_CSV = DATA_DIR / "schema.csv"

# ---------- 전처리 산출물 ----------
# CSV 대신 parquet 을 쓰는 이유: dtype 이 보존되고(문자/숫자/카테고리),
# 용량이 수 분의 1로 줄며, 로딩이 수십 배 빠르다.
SILVER_PARQUET = DATA_DIR / "silver.parquet"          # 1단계: 구조 정리 (1행 = 응답자 1명)
GOLD_WIDE_PARQUET = DATA_DIR / "gold_wide.parquet"    # 2단계: ML용 멀티핫 행렬
GOLD_LONG_PARQUET = DATA_DIR / "gold_long.parquet"    # 2단계: 통계용 세로 형태
DICTIONARY_CSV = DATA_DIR / "so2024_dictionary.csv"   # 컬럼 -> qid/질문 원문 대응표

# ---------- AISelect 예측용 분할 산출물 (3단계) ----------
AI_TRAIN_PARQUET = DATA_DIR / "ai_train.parquet"      # 학습용 (80%)
AI_TEST_PARQUET = DATA_DIR / "ai_test.parquet"        # 최종 평가용 (20%, 1회만 사용)
AI_SPLIT_IDS_CSV = DATA_DIR / "ai_split_ids.csv"      # ResponseId -> train/test 소속
AI_MANIFEST_JSON = DATA_DIR / "ai_split_manifest.json"  # 분할 명세(제거 컬럼·규칙 포함)

# ---------- 선형 트랙 변환 산출물 (4단계) ----------
# 트리 모델(LightGBM)용은 ai_train/ai_test 그대로, 선형/로지스틱용은 아래 파일을 쓴다.
AI_TRAIN_LINEAR_PARQUET = DATA_DIR / "ai_train_linear.parquet"
AI_TEST_LINEAR_PARQUET = DATA_DIR / "ai_test_linear.parquet"
LINEAR_PARAMS_JSON = DATA_DIR / "linear_transform_params.json"  # train 에서 fit 한 값 기록

# ---------- 모델 산출물 (5단계) ----------
LOGISTIC_MODEL_JOBLIB = DATA_DIR / "model_logistic.joblib"   # 학습된 파이프라인
LOGISTIC_METRICS_JSON = DATA_DIR / "logistic_metrics.json"   # CV·test 성능 기록

# ---------- 데이터 규약 ----------
# 다중선택 문항은 한 칸에 "Python;SQL;Go" 처럼 이어 붙어 있다.
MULTI_SEP = ";"

# ---------- ML 평가 결과 저장 경로 ----------
# 모델 학습 이후 생성되는 모델 파일, 성능 지표, 그래프,
# HTML 보고서, 실행 로그를 프로젝트 루트의 output 폴더 아래에 저장한다.
#
# BASE_DIR는 위쪽에서 다음과 같이 프로젝트 루트로 설정되어 있다.
#
# BASE_DIR = Path(__file__).resolve().parent.parent
#
# 따라서 아래 경로는 실행 위치와 관계없이 항상
# 프로젝트 루트/output 을 가리킨다.
OUTPUT_DIR = BASE_DIR / "output"


# 학습이 완료된 모델 파일을 저장하는 폴더다.
#
# 예:
# output/model/catboost_pipeline.joblib
MODEL_DIR = OUTPUT_DIR / "model"


# accuracy, precision, recall, F1-score, ROC-AUC 같은
# 모델 평가 결과를 JSON 또는 CSV 파일로 저장하는 폴더다.
#
# 예:
# output/metrics/metrics.json
# output/metrics/classification_report.csv
# output/metrics/predictions.csv
METRICS_DIR = OUTPUT_DIR / "metrics"


# Confusion Matrix, ROC Curve처럼
# 모델의 성능을 시각적으로 확인하기 위한 이미지를 저장하는 폴더다.
#
# 예:
# output/figures/confusion_matrix.png
# output/figures/roc_curve.png
FIGURE_DIR = OUTPUT_DIR / "figures"


# 모델 평가 결과를 정리한 HTML 보고서를 저장하는 폴더다.
#
# 예:
# output/report/model_evaluation_report.html
REPORT_DIR = OUTPUT_DIR / "report"


# 파이프라인 실행 과정에서 발생하는 로그를 저장하는 폴더다.
#
# 로그에는 모델 학습 시작·완료, 평가 완료,
# 모델 저장 경로, 오류 발생 내용 등이 기록된다.
#
# 예:
# output/logs/ml_pipeline.log
LOG_DIR = OUTPUT_DIR / "logs"


# 프로그램 실행 시 필요한 출력 폴더가 존재하지 않으면
# 자동으로 생성한다.
#
# parents=True:
# 상위 폴더인 output이 없더라도 함께 생성한다.
#
# exist_ok=True:
# 이미 폴더가 존재하더라도 오류를 발생시키지 않는다.
for directory in (
    OUTPUT_DIR,
    MODEL_DIR,
    METRICS_DIR,
    FIGURE_DIR,
    REPORT_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------- 분류 대상 라벨 ----------
# 모델이 사용하는 숫자형 target 값을
# 사람이 이해할 수 있는 한글 이름으로 변환하기 위한 설정이다.
#
# 이 값은 다음 위치에서 사용된다.
#
# 1. Classification Report의 클래스 이름
# 2. Confusion Matrix의 축 라벨
# 3. 예측 결과 CSV의 실제값·예측값 한글 표시
# 4. HTML 평가 보고서의 분류 기준 설명
#
# 실제 target의 의미가 바뀌면 이 부분도 함께 수정해야 한다.
TARGET_LABELS = {
    0: "현재 AI 사용자",
    1: "향후 AI 사용 예정자",
}