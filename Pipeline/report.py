from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

from config import REPORT_DIR, TARGET_LABELS
from logger_config import get_logger

logger = get_logger()

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 사용 집단 분류 모델 평가 보고서</title>
    <style>
        body {
            font-family: Arial, "Apple SD Gothic Neo", sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #222;
        }
        h1, h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .metric-card {
            width: 170px;
            padding: 16px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 16px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th { background: #f2f2f2; }
        img {
            max-width: 700px;
            width: 100%;
        }
        .description {
            padding: 15px;
            background: #f7f7f7;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h1>AI 사용 집단 분류 모델 평가 보고서</h1>
    <p>생성 시각: {{ generated_at }}</p>

    <div class="description">
        <strong>분류 기준</strong><br>
        0: {{ target_zero }}<br>
        1: {{ target_one }}
    </div>

    <h2>1. 주요 평가 지표</h2>
    <div class="metric-container">
        {% for name, value in metrics.items() %}
            {% if name not in ["test_size", "evaluated_at"] %}
                <div class="metric-card">
                    <div>{{ name }}</div>
                    <div class="metric-value">{{ "%.4f"|format(value) }}</div>
                </div>
            {% endif %}
        {% endfor %}
    </div>
    <p>평가 데이터 수: {{ metrics.test_size }}건</p>

    <h2>2. Classification Report</h2>
    {{ classification_report | safe }}

    <h2>3. Confusion Matrix</h2>
    <img src="../figures/confusion_matrix.png" alt="Confusion Matrix">

    <h2>4. ROC Curve</h2>
    <img src="../figures/roc_curve.png" alt="ROC Curve">

    <h2>5. 결과 해석</h2>
    <div class="description">
        <p>Accuracy는 전체 테스트 데이터 중 올바르게 분류한 비율이다.</p>
        <p>Precision은 모델이 1로 예측한 대상 중 실제 1인 비율이다.</p>
        <p>Recall은 실제 1인 대상 중 모델이 찾아낸 비율이다.</p>
        <p>F1-score는 Precision과 Recall을 함께 고려한 지표다.</p>
        <p>ROC-AUC는 두 집단을 구분하는 전반적인 능력을 나타낸다.</p>
    </div>
</body>
</html>
"""


def create_html_report(metrics: dict, report_df: pd.DataFrame) -> Path:
    """평가 결과를 HTML 보고서로 생성한다."""
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        target_zero=TARGET_LABELS[0],
        target_one=TARGET_LABELS[1],
        metrics=metrics,
        classification_report=report_df.to_html(
            float_format=lambda value: f"{value:.4f}",
        ),
    )

    output_path = REPORT_DIR / "model_evaluation_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML 보고서 저장 완료: %s", output_path)
    return output_path