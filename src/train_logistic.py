"""AISelect 예측 베이스라인: 로지스틱 회귀 (5단계).

입력은 선형 트랙 데이터(ai_train_linear / ai_test_linear)다 — 왜도 교정과
결측 대체가 끝나 있어, 여기서는 인코딩·스케일링과 학습만 한다.

평가 규율 (ai_split_manifest 의 규칙 그대로):
    - 성능 추정은 train 내부 5-fold 층화 CV 로 한다
    - test 는 마지막에 딱 1회 평가한다 — threshold·피처·하이퍼파라미터를
      test 성능을 보고 고치기 시작하면 그 순간부터 test 가 오염된다

파이프라인 구성:
    수치형 18개        StandardScaler (계수 크기를 비교 가능하게)
    범주형(category)   "Missing" 대체 후 원핫. min_frequency=30 으로 희소 수준
                       (Country 꼬리 등)은 infrequent 버킷에 접는다 — 선형 모델에서
                       표본 적은 수준의 계수는 잡음이기 때문 (고카디널리티 대응)
    0/1 더미·플래그    그대로 통과

실행:
    python -m src.train_logistic
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        AI_TRAIN_LINEAR_PARQUET,
        LOGISTIC_METRICS_JSON,
        LOGISTIC_MODEL_JOBLIB,
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        AI_TRAIN_LINEAR_PARQUET,
        LOGISTIC_METRICS_JSON,
        LOGISTIC_MODEL_JOBLIB,
    )

SEED = 42
SCORING = ["roc_auc", "accuracy", "f1", "balanced_accuracy"]


def load(path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    y = df.pop("target")
    # category -> object: SimpleImputer 가 새 값("Missing")을 채울 수 있게
    for c in df.select_dtypes(include="category").columns:
        df[c] = df[c].astype(object)
    return df, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cont_cols = X.select_dtypes(include="float64").columns.tolist()   # 수치형 18개
    cat_cols = X.select_dtypes(include="object").columns.tolist()     # 단일선택 범주형
    # 나머지(uint8 멀티핫, bool 플래그)는 remainder 로 그대로 통과

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), cont_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30)),
            ]), cat_cols),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=3000, random_state=SEED)),
    ])


def top_coefficients(model: Pipeline, k: int = 15) -> tuple[list, list]:
    """표준화된 계수 상·하위 k개. 이름의 파이프라인 접두어는 떼어낸다."""
    names = model[:-1].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    coefs = model[-1].coef_[0]
    order = np.argsort(coefs)
    top_pos = [(names[i], round(float(coefs[i]), 3)) for i in order[::-1][:k]]
    top_neg = [(names[i], round(float(coefs[i]), 3)) for i in order[:k]]
    return top_pos, top_neg


def main() -> None:
    t0 = time.perf_counter()
    x_tr, y_tr = load(AI_TRAIN_LINEAR_PARQUET)
    x_te, y_te = load(AI_TEST_LINEAR_PARQUET)
    base_rate = float(y_tr.mean())
    print(f"train {x_tr.shape[0]:,} × {x_tr.shape[1]:,} | 다수 클래스 기준선 {base_rate:.1%}")

    model = build_pipeline(x_tr)

    # ---- 1) train 내부 5-fold 층화 CV — 성능 추정의 기준 ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(model, x_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=1)
    cv_result = {m: {"mean": round(float(scores[f"test_{m}"].mean()), 4),
                     "std": round(float(scores[f"test_{m}"].std()), 4)}
                 for m in SCORING}
    print("\n[5-fold CV (train)]")
    for m, v in cv_result.items():
        print(f"  {m:<18} {v['mean']:.4f} ± {v['std']:.4f}")

    # ---- 2) 전체 train 으로 학습 -> test 1회 평가 ----
    model.fit(x_tr, y_tr)
    prob = model.predict_proba(x_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    test_result = {
        "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
        "accuracy": round(float((pred == y_te).mean()), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    print("\n[test 1회 평가]")
    for m in SCORING:
        print(f"  {m:<18} {test_result[m]:.4f}")
    tn, fp, fn, tp = np.array(test_result["confusion_matrix"]).ravel()
    print(f"  confusion          TN {tn:,} / FP {fp:,} / FN {fn:,} / TP {tp:,}")

    # ---- 3) 해석: 표준화 계수 상·하위 ----
    top_pos, top_neg = top_coefficients(model)
    print("\n[AI 사용(1) 쪽으로 미는 피처 Top 15]")
    for n, c in top_pos:
        print(f"  {c:+.3f}  {n}")
    print("\n[미사용(0) 쪽으로 미는 피처 Top 15]")
    for n, c in top_neg:
        print(f"  {c:+.3f}  {n}")

    # ---- 4) 저장 ----
    n_features = len(model[:-1].get_feature_names_out())
    joblib.dump(model, LOGISTIC_MODEL_JOBLIB)
    LOGISTIC_METRICS_JSON.write_text(json.dumps({
        "model": "LogisticRegression(C=1.0, L2), 원핫 min_frequency=30",
        "n_train": len(x_tr), "n_test": len(x_te),
        "n_features_encoded": n_features,
        "baseline_majority_accuracy": round(base_rate, 4),
        "cv_5fold_train": cv_result,
        "test_single_shot": test_result,
        "top_positive": top_pos, "top_negative": top_neg,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n인코딩 후 피처 {n_features:,}개 | {time.perf_counter() - t0:.0f}초")
    print(f"저장: {LOGISTIC_MODEL_JOBLIB.name}, {LOGISTIC_METRICS_JSON.name}")


if __name__ == "__main__":
    main()
