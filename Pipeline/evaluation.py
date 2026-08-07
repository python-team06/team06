"""학습된 이진 분류 모델의 성능 평가 및 결과 저장 함수 모음."""

from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURE_DIR, METRICS_DIR, TARGET_LABELS

from .logger_config import get_logger


# 프로젝트 전체에서 동일한 로거를 사용한다.
logger = get_logger()


# 운영체제마다 설치된 한글 글꼴이 다르므로 후보를 순서대로 확인한다.
# 사용 가능한 첫 번째 한글 글꼴을 matplotlib 기본 글꼴로 지정한다.
_KOREAN_FONT_CANDIDATES = [
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "NanumGothic",
    "Malgun Gothic",
    "AppleGothic",
]

_available_fonts = {font.name for font in fm.fontManager.ttflist}

for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break

# 한글 글꼴 사용 시 마이너스 기호가 네모로 깨지는 현상을 방지한다.
plt.rcParams["axes.unicode_minus"] = False


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    dict,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """학습된 이진 분류 모델을 테스트 데이터로 평가한다.

    Parameters
    ----------
    model:
        ``predict``와 ``predict_proba``를 지원하는 학습 완료 모델 또는 Pipeline.
    X_test:
        모델에 입력할 테스트 특성 데이터.
    y_test:
        테스트 데이터의 실제 정답.

    Returns
    -------
    tuple
        다음 네 값을 순서대로 반환한다.

        1. 주요 평가 지표를 담은 딕셔너리
        2. Classification Report 데이터프레임
        3. 모델의 예측 클래스 배열
        4. 클래스 1에 대한 예측 확률 배열
    """

    logger.info("모델 평가 시작")

    # 모델이 반환한 예측 결과가 (n, 1) 형태일 수 있으므로
    # numpy 배열로 변환한 뒤 1차원 형태로 정리한다.
    predictions = np.asarray(
        model.predict(X_test)
    ).reshape(-1)

    # 이진 분류에서 두 번째 열은 클래스 1일 확률이다.
    # ROC Curve와 ROC-AUC 계산에 사용한다.
    probabilities = model.predict_proba(X_test)[:, 1]

    # 핵심 성능 지표를 하나의 딕셔너리로 정리한다.
    # JSON 저장을 위해 numpy 타입이 아닌 기본 Python float/int로 변환한다.
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "test_size": int(len(y_test)),
        "evaluated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    # 클래스별 precision, recall, f1-score, support를 계산한다.
    # output_dict=True로 지정하면 결과를 표 형태로 가공할 수 있다.
    report = classification_report(
        y_test,
        predictions,
        target_names=[
            TARGET_LABELS[0],
            TARGET_LABELS[1],
        ],
        output_dict=True,
        zero_division=0,
    )

    # CSV와 HTML 보고서에서 활용하기 위해 DataFrame으로 변환한다.
    report_df = pd.DataFrame(report).T

    logger.info("모델 평가 완료")

    return (
        metrics,
        report_df,
        predictions,
        probabilities,
    )


def save_confusion_matrix(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> Path:
    """실제값과 예측값으로 혼동행렬 이미지를 생성해 저장한다."""

    # 실제 클래스와 예측 클래스 조합별 개수를 계산한다.
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # 숫자 0, 1 대신 사람이 읽기 쉬운 클래스명을 축에 표시한다.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            TARGET_LABELS[0],
            TARGET_LABELS[1],
        ],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    output_path = FIGURE_DIR / "confusion_matrix.png"

    # 보고서에서도 선명하게 보이도록 150dpi로 저장한다.
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    # 반복 실행 시 matplotlib 객체가 메모리에 남지 않도록 닫는다.
    plt.close(fig)

    logger.info("혼동행렬 이미지 저장 완료: %s", output_path)
    return output_path


def save_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> Path:
    """예측 확률을 이용해 ROC Curve 이미지를 생성해 저장한다."""

    # 여러 확률 임계값에서 FPR과 TPR을 계산한다.
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )

    # ROC 곡선 아래 면적을 계산한다.
    auc_score = roc_auc_score(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    # 모델의 ROC 곡선을 그린다.
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}",
    )

    # 무작위 분류기의 기준선을 함께 표시한다.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()

    output_path = FIGURE_DIR / "roc_curve.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("ROC Curve 이미지 저장 완료: %s", output_path)
    return output_path


def save_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """테스트 데이터의 실제값, 예측값, 예측 확률을 CSV로 저장한다."""

    # 원본 테스트 데이터와 인덱스를 맞추기 위해 같은 인덱스로 결과표를 만든다.
    prediction_result = pd.DataFrame(
        index=test_df.index
    )

    # 실제 target 값을 정수형으로 저장한다.
    prediction_result["actual_target"] = (
        test_df["target"]
        .astype(int)
        .to_numpy()
    )

    # 모델이 예측한 클래스 값을 정수형으로 저장한다.
    prediction_result["predicted_target"] = (
        predictions.astype(int)
    )

    # 클래스 1일 예측 확률을 저장한다.
    prediction_result["predicted_probability"] = (
        probabilities
    )

    # 사람이 결과를 쉽게 읽을 수 있도록 숫자 라벨을 한글 이름으로 변환한다.
    prediction_result["actual_label"] = (
        prediction_result["actual_target"]
        .map(TARGET_LABELS)
    )

    prediction_result["predicted_label"] = (
        prediction_result["predicted_target"]
        .map(TARGET_LABELS)
    )

    # 실제값과 예측값이 같은지 여부를 True/False로 표시한다.
    prediction_result["correct"] = (
        prediction_result["actual_target"]
        == prediction_result["predicted_target"]
    )

    output_path = METRICS_DIR / "predictions.csv"

    prediction_result.to_csv(
        output_path,
        index=True,
        encoding="utf-8-sig",
    )

    logger.info("예측 결과 CSV 저장 완료: %s", output_path)
    return prediction_result