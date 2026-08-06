from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    FIGURE_DIR,
    METRICS_DIR,
    TARGET_LABELS,
)
from logger_config import get_logger


logger = get_logger()

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
    """학습된 이진 분류 모델을 평가한다."""

    logger.info("모델 평가 시작")

    predictions = np.asarray(
        model.predict(X_test)
    ).reshape(-1)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    report_df = pd.DataFrame(report).T

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
    """혼동행렬 이미지를 저장한다."""

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

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

    output_path = (
        FIGURE_DIR / "confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def save_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> Path:
    """ROC Curve 이미지를 저장한다."""

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(y_test, probabilities)
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}",
    )

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

    return output_path


def save_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """실제값과 예측값을 CSV로 저장한다."""

    prediction_result = pd.DataFrame(
        index=test_df.index
    )

    prediction_result["actual_target"] = (
        test_df["target"]
        .astype(int)
        .to_numpy()
    )

    prediction_result["predicted_target"] = (
        predictions.astype(int)
    )

    prediction_result["predicted_probability"] = (
        probabilities
    )

    prediction_result["actual_label"] = (
        prediction_result["actual_target"]
        .map(TARGET_LABELS)
    )

    prediction_result["predicted_label"] = (
        prediction_result["predicted_target"]
        .map(TARGET_LABELS)
    )

    prediction_result["correct"] = (
        prediction_result["actual_target"]
        == prediction_result["predicted_target"]
    )

    output_path = (
        METRICS_DIR / "predictions.csv"
    )

    prediction_result.to_csv(
        output_path,
        index=True,
        encoding="utf-8-sig",
    )

    return prediction_result