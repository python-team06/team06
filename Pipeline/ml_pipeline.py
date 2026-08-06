import json

import joblib
import numpy as np
import pandas as pd

from ml_config import METRICS_DIR, MODEL_DIR
from evaluation import (
    evaluate_classifier,
    save_confusion_matrix,
    save_predictions,
    save_roc_curve,
)
from logger_config import get_logger
from report import create_html_report

logger = get_logger()


def finalize_ml_pipeline(
    model_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> dict:
    """이미 학습된 모델을 평가하고 결과물과 HTML 보고서를 생성한다."""
    logger.info("=" * 60)
    logger.info("학습 완료 모델 평가·저장 자동화 시작")

    try:
        metrics, report_df, predictions, probabilities = evaluate_classifier(
            model_pipeline,
            X_test,
            y_test,
        )

        model_path = MODEL_DIR / "catboost_pipeline.joblib"
        joblib.dump(model_pipeline, model_path)

        loaded_pipeline = joblib.load(model_path)
        reload_predictions = np.asarray(
            loaded_pipeline.predict(X_test)
        ).reshape(-1)

        if not np.array_equal(predictions, reload_predictions):
            raise ValueError("저장 전후 예측 결과가 다릅니다.")

        metrics_path = METRICS_DIR / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=4)

        report_path = METRICS_DIR / "classification_report.csv"
        report_df.to_csv(report_path, encoding="utf-8-sig")

        confusion_matrix_path = save_confusion_matrix(y_test, predictions)
        roc_curve_path = save_roc_curve(y_test, probabilities)
        prediction_result = save_predictions(
            test_df,
            predictions,
            probabilities,
        )
        html_report_path = create_html_report(metrics, report_df)

        result = {
            "status": "success",
            "metrics": metrics,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "classification_report_path": str(report_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "roc_curve_path": str(roc_curve_path),
            "html_report_path": str(html_report_path),
            "prediction_count": int(len(prediction_result)),
        }

        logger.info("평가·저장 자동화 완료")
        logger.info("=" * 60)
        return result

    except Exception:
        logger.exception("ML Pipeline 후처리 중 오류 발생")
        raise


def run_ml_pipeline(
    model_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> dict:
    """학습부터 평가·저장·HTML 생성까지 한 번에 수행한다."""
    logger.info("모델 학습 시작")
    model_pipeline.fit(X_train, y_train)
    logger.info("모델 학습 완료")

    return finalize_ml_pipeline(
        model_pipeline=model_pipeline,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
    )