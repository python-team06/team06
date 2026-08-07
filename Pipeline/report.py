"""모델 평가 결과를 Markdown 보고서로 생성하는 모듈."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import REPORT_DIR, TARGET_LABELS

from .logger_config import get_logger

logger = get_logger()


def dataframe_to_markdown_table(report_df: pd.DataFrame) -> str:
    """Classification Report 데이터프레임을 Markdown 표로 변환한다.

    pandas의 to_markdown()은 tabulate 패키지를 필요로 할 수 있으므로,
    별도 패키지 없이 동작하도록 Markdown 표를 직접 생성한다.

    Parameters
    ----------
    report_df:
        클래스별 precision, recall, f1-score, support가 담긴
        Classification Report 데이터프레임.

    Returns
    -------
    str
        Markdown 문법으로 작성된 표 문자열.
    """

    # 원래 인덱스에는 클래스 이름, accuracy, macro avg 등이 들어 있으므로
    # 인덱스를 일반 컬럼으로 바꿔 Markdown 표의 첫 번째 열로 사용한다.
    table_df = report_df.reset_index()

    # reset_index()로 생성된 첫 번째 컬럼 이름을 의미가 분명하도록 변경한다.
    table_df = table_df.rename(
        columns={
            table_df.columns[0]: "class",
        }
    )

    # 숫자형 값은 보고서에서 보기 쉽도록 소수점 넷째 자리까지 표시한다.
    formatted_rows = []

    for _, row in table_df.iterrows():
        formatted_row = []

        for column in table_df.columns:
            value = row[column]

            if isinstance(value, float):
                formatted_row.append(f"{value:.4f}")
            else:
                formatted_row.append(str(value))

        formatted_rows.append(formatted_row)

    # Markdown 표의 헤더를 만든다.
    header = "| " + " | ".join(table_df.columns) + " |"

    # 각 컬럼 아래에 구분선을 추가한다.
    separator = "| " + " | ".join(
        ["---"] * len(table_df.columns)
    ) + " |"

    # 각 행을 Markdown 표 문법으로 변환한다.
    body = "\n".join(
        "| " + " | ".join(row) + " |"
        for row in formatted_rows
    )

    return "\n".join(
        [
            header,
            separator,
            body,
        ]
    )


def create_markdown_report(
    metrics: dict,
    report_df: pd.DataFrame,
) -> Path:
    """평가 지표와 Classification Report를 Markdown 파일로 저장한다.

    Parameters
    ----------
    metrics:
        Accuracy, Precision, Recall, F1-score, ROC-AUC,
        평가 데이터 수와 평가 시각 등을 담은 딕셔너리.

    report_df:
        클래스별 평가 결과가 정리된
        Classification Report 데이터프레임.

    Returns
    -------
    Path
        생성된 Markdown 보고서 파일 경로.
    """

    # Classification Report를 Markdown 표 문자열로 변환한다.
    classification_report_table = dataframe_to_markdown_table(
        report_df
    )

    # metrics 딕셔너리에서 주요 성능 지표를 꺼내
    # Markdown 표의 각 행으로 구성한다.
    metric_rows = []

    for name, value in metrics.items():
        # test_size와 evaluated_at은 별도 항목으로 표시하므로
        # 주요 평가 지표 표에서는 제외한다.
        if name in {
            "test_size",
            "evaluated_at",
        }:
            continue

        metric_rows.append(
            f"| {name} | {value:.4f} |"
        )

    metric_table = "\n".join(
        [
            "| Metric | Value |",
            "| --- | ---: |",
            *metric_rows,
        ]
    )

    # Markdown 보고서 전체 내용을 하나의 문자열로 만든다.
    #
    # 이미지 경로는 report.md가 output/report 안에 저장되는 것을 기준으로
    # output/figures 폴더를 상대경로로 참조한다.
    markdown = f"""# AI 사용 집단 분류 모델 평가 보고서

- 생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 모델 평가 시각: {metrics["evaluated_at"]}
- 평가 데이터 수: {metrics["test_size"]}건

## 분류 기준

| Target | Label |
| ---: | --- |
| 0 | {TARGET_LABELS[0]} |
| 1 | {TARGET_LABELS[1]} |

## 1. 주요 평가 지표

{metric_table}

## 2. Classification Report

{classification_report_table}

## 3. Confusion Matrix

![Confusion Matrix](../figures/confusion_matrix.png)

## 4. ROC Curve

![ROC Curve](../figures/roc_curve.png)

## 5. 결과 해석

- **Accuracy**는 전체 테스트 데이터 중 모델이 올바르게 분류한 비율이다.
- **Precision**은 모델이 1로 예측한 데이터 중 실제로 1인 데이터의 비율이다.
- **Recall**은 실제 1인 데이터 중 모델이 올바르게 찾아낸 비율이다.
- **F1-score**는 Precision과 Recall의 균형을 함께 평가하는 지표다.
- **ROC-AUC**는 서로 다른 두 집단을 전반적으로 구분하는 모델의 능력을 나타낸다.

## 6. 생성된 결과물

- 모델 파일: `output/model/catboost_pipeline.joblib`
- 평가 지표: `output/metrics/metrics.json`
- 분류 보고서: `output/metrics/classification_report.csv`
- 예측 결과: `output/metrics/predictions.csv`
- 혼동행렬: `output/figures/confusion_matrix.png`
- ROC 곡선: `output/figures/roc_curve.png`
"""

    # 최종 Markdown 파일 저장 경로를 설정한다.
    output_path = REPORT_DIR / "report.md"

    # 한글이 깨지지 않도록 UTF-8 인코딩으로 저장한다.
    output_path.write_text(
        markdown,
        encoding="utf-8",
    )

    logger.info(
        "Markdown 보고서 저장 완료: %s",
        output_path,
    )

    return output_path