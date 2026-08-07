"""모델 학습 이후 평가, 검증, 저장, 보고서 생성을 자동화하는 모듈."""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import METRICS_DIR, MODEL_DIR

from .evaluation import (
    evaluate_classifier,
    save_confusion_matrix,
    save_predictions,
    save_roc_curve,
)
from .logger_config import get_logger
from .report import create_markdown_report


logger = get_logger()


def finalize_ml_pipeline(
    model_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> dict:
    """학습 완료 모델을 평가하고 모든 결과물을 저장한다.

    이 함수는 모델 학습 자체는 수행하지 않는다. 이미 ``fit``이 완료된 모델을
    전달받아 다음 작업을 순서대로 처리한다.

    1. 테스트 데이터 성능 평가
    2. 학습 모델 저장
    3. 모델 재로딩 후 예측 결과 일치 여부 검증
    4. 평가 지표 및 Classification Report 저장
    5. 혼동행렬, ROC Curve, 예측 결과 저장
    6. MD 보고서 생성
    7. 전체 결과 경로를 딕셔너리로 반환
    """

    logger.info("=" * 60)
    logger.info("학습 완료 모델 평가·저장 자동화 시작")

    try:
        # 모델 성능 지표, 상세 보고서, 예측값, 예측 확률을 한 번에 계산한다.
        metrics, report_df, predictions, probabilities = evaluate_classifier(
            model_pipeline,
            X_test,
            y_test,
        )

        # 학습이 끝난 전체 Pipeline을 joblib 파일로 저장한다.
        # 전처리 단계와 모델이 함께 저장되므로 추론 시 같은 흐름을 재사용할 수 있다.
        model_path = MODEL_DIR / "catboost_pipeline.joblib"
        joblib.dump(model_pipeline, model_path)
        logger.info("모델 저장 완료: %s", model_path)

        # 저장한 모델을 다시 불러와 정상적으로 복원되는지 확인한다.
        loaded_pipeline = joblib.load(model_path)

        # 재로딩한 모델로 동일한 테스트 데이터를 예측한다.
        reload_predictions = np.asarray(
            loaded_pipeline.predict(X_test)
        ).reshape(-1)

        # 저장 전 모델과 저장 후 모델의 예측 결과가 다르면
        # 모델 저장 또는 복원 과정에 문제가 있는 것으로 판단한다.
        if not np.array_equal(predictions, reload_predictions):
            raise ValueError("저장 전후 예측 결과가 다릅니다.")

        logger.info("모델 재로딩 검증 완료")

        # 주요 평가 지표를 JSON으로 저장한다.
        metrics_path = METRICS_DIR / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=4,
            )

        logger.info("평가 지표 JSON 저장 완료: %s", metrics_path)

        # 클래스별 평가 결과를 CSV로 저장한다.
        report_path = METRICS_DIR / "classification_report.csv"
        report_df.to_csv(
            report_path,
            encoding="utf-8-sig",
        )

        logger.info("Classification Report 저장 완료: %s", report_path)

        # 시각화 및 예측 결과 파일을 생성한다.
        confusion_matrix_path = save_confusion_matrix(
            y_test,
            predictions,
        )

        roc_curve_path = save_roc_curve(
            y_test,
            probabilities,
        )

        prediction_result = save_predictions(
            test_df,
            predictions,
            probabilities,
        )

        # 앞서 만든 지표와 표를 이용해 최종 md 보고서를 생성한다.
        markdown_report_path = create_markdown_report(
            metrics,
            report_df,
        )

        # 메인 실행 파일에서 결과를 한 번에 확인할 수 있도록
        # 상태, 성능, 저장 경로를 하나의 딕셔너리로 정리한다.
        result = {
            "status": "success",
            "metrics": metrics,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "classification_report_path": str(report_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "roc_curve_path": str(roc_curve_path),
            "markdown_report_path": str(markdown_report_path),
            "prediction_count": int(len(prediction_result)),
        }

        logger.info("평가·저장 자동화 완료")
        logger.info("=" * 60)

        return result

    except Exception:
        # logger.exception은 오류 메시지와 전체 traceback을 함께 기록한다.
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
    """모델 학습부터 평가·저장·md 보고서 생성까지 한 번에 수행한다.

    모델을 아직 학습하지 않은 상태에서 호출할 때 사용하는 진입 함수다.
    먼저 ``fit``을 수행한 뒤 ``finalize_ml_pipeline``에 평가 및 저장 작업을 맡긴다.
    """

    logger.info("모델 학습 시작")

    # 전달받은 Pipeline 또는 모델을 학습 데이터로 학습한다.
    model_pipeline.fit(
        X_train,
        y_train,
    )

    logger.info("모델 학습 완료")

    # 학습된 모델의 평가, 저장, 시각화, 보고서 생성을 수행한다.
    return finalize_ml_pipeline(
        model_pipeline=model_pipeline,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
    )