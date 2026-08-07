"""
프로젝트 전체 실행 진입점.

실행 순서
--------
1. 데이터 전처리
2. AISelect 학습/테스트 데이터 분할
3. 데이터 시각화
4. ML Pipeline 실행

프로젝트 루트에서 실행:

    python main.py
"""

import time

from src import preprocess
from src import split_aiselect
from src import visualize_plotly
from src import visualize_seaborn

from pipeline.main_pipeline import main as run_main_pipeline


def main() -> None:
    """전체 데이터 분석 및 ML 파이프라인을 순서대로 실행한다."""

    # 전체 실행 시간 측정
    start_time = time.perf_counter()

    print("=" * 70)
    print("Stack Overflow AI Usage Analysis")
    print("전체 파이프라인 실행 시작")
    print("=" * 70)

    # =========================================================
    # 1. 데이터 전처리
    # =========================================================
    print("\n")
    print("=" * 70)
    print("[1단계] 데이터 전처리")
    print("=" * 70)

    # results.csv
    #     ↓
    # silver.parquet
    # gold_wide.parquet
    # gold_long.parquet
    preprocess.main()

    print("\n[완료] 데이터 전처리")


    # =========================================================
    # 2. AISelect 학습 / 테스트 데이터 분할
    # =========================================================
    print("\n")
    print("=" * 70)
    print("[2단계] Train / Test 데이터 분할")
    print("=" * 70)

    # gold_wide.parquet
    #     ↓
    # AISelect 무응답 제거
    # 누수 컬럼 제거
    #     ↓
    # ai_train.parquet
    # ai_test.parquet
    split_aiselect.main()

    print("\n[완료] Train / Test 데이터 분할")


    # =========================================================
    # 3. 데이터 시각화
    # =========================================================
    print("\n")
    print("=" * 70)
    print("[3단계] 데이터 시각화")
    print("=" * 70)

    # ---------------------------------------------------------
    # 3-1. Plotly 시각화
    # ---------------------------------------------------------
    print("\n[3-1] Plotly 시각화 시작")

    visualize_plotly.main()

    print("[완료] Plotly 시각화")


    # ---------------------------------------------------------
    # 3-2. Seaborn 시각화
    # ---------------------------------------------------------
    print("\n[3-2] Seaborn 시각화 시작")

    visualize_seaborn.main()

    print("[완료] Seaborn 시각화")


    # =========================================================
    # 4. ML Pipeline
    # =========================================================
    print("\n")
    print("=" * 70)
    print("[4단계] ML Pipeline 실행")
    print("=" * 70)

    # 모델 학습
    # 모델 평가
    # 평가 결과 저장
    # HTML 보고서 생성 등의 작업은
    # pipeline/main_pipeline.py에서 수행한다.
    run_main_pipeline()

    print("\n[완료] ML Pipeline")


    # =========================================================
    # 전체 실행 완료
    # =========================================================
    elapsed_time = time.perf_counter() - start_time

    print("\n")
    print("=" * 70)
    print("전체 파이프라인 실행 완료")
    print(f"총 실행 시간: {elapsed_time:.2f}초")
    print("=" * 70)


if __name__ == "__main__":
    main()