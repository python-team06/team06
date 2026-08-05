"""선형 모델 트랙용 변환 (4단계): 왜도 교정 + 결측 대체.

트리 트랙(LightGBM 등)은 ai_train / ai_test 를 그대로 쓴다 — 트리는 단조변환에
불변이고 결측(NaN)을 네이티브로 다루므로 이 변환이 필요 없다.

선형/로지스틱(statsmodels·sklearn) 트랙을 위해 수치형 18개에 다음을 적용한다:

    1) 윈저라이징   ConvertedCompYearly 를 train 의 p1/p99 로 절단.
                    ($1 같은 하위 쓰레기값이 log 를 좌왜(-2.2)로 뒤집는 것을 방지)
    2) log1p        우측 왜도 |skew|>1 인 13개 -> <이름>_log 로 교체.
                    (JobSat 은 좌왜라 log 가 악화시키므로 제외 — 실측 -1.0 -> -2.6)
    3) 0-초과 플래그 zero-inflated 4개(0 비율 59~68%)에 <이름>_any 추가.
                    log1p 만으로는 왜도 0.9~1.3 이 남아, "썼는가/얼마나"를 분리한다.
    4) 결측 대체    train 중앙값으로 채우고 <이름>_missing 플래그를 추가.
                    결측률이 클래스와 일관되게 상관(사용자 쪽이 ~1.5%p 낮음)하므로
                    플래그가 그 신호를 보존한다. 평균 대신 중앙값인 이유: 왜도가 큰
                    분포에서 평균은 꼬리에 끌려간다(연봉 mean 86.9k vs median 65k).

누수 방지: 윈저 경계와 중앙값은 fitted 값이다 — 전부 train 에서만 계산해서
test 에 그대로 적용하고, data/linear_transform_params.json 에 기록한다.
스케일링(StandardScaler)은 여기서 하지 않는다 — 모델 파이프라인(CV 내부)의 몫.

실행:
    python -m src.transform_linear
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        AI_TEST_PARQUET,
        AI_TRAIN_LINEAR_PARQUET,
        AI_TRAIN_PARQUET,
        LINEAR_PARAMS_JSON,
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        AI_TEST_PARQUET,
        AI_TRAIN_LINEAR_PARQUET,
        AI_TRAIN_PARQUET,
        LINEAR_PARAMS_JSON,
    )

# 왜도 진단(analyze 단계 실측)에 근거한 컬럼 분류
WINSOR_COL = "ConvertedCompYearly"                     # skew +50.9, min $1 -> 절단 필요
LOG_COLS = [                                           # 우측 왜도 |skew|>1 (13개)
    "ConvertedCompYearly", "YearsCode", "YearsCodePro", "WorkExp",
    "JobSatPoints_1", "JobSatPoints_4", "JobSatPoints_5", "JobSatPoints_6",
    "JobSatPoints_7", "JobSatPoints_8", "JobSatPoints_9", "JobSatPoints_10",
    "JobSatPoints_11",
]
ANY_COLS = [                                           # zero-inflated (0 이 59~68%)
    "JobSatPoints_4", "JobSatPoints_5", "JobSatPoints_10", "JobSatPoints_11",
]
RAW_COLS = [                                           # 변환 없이 대체만 (대칭·순서형·좌왜)
    "JobSat", "AgeNum", "EdLevelNum", "OrgSizeNum", "SOVisitFreqNum",
]
NUM_COLS = LOG_COLS + RAW_COLS                         # 수치형 18개 전부


def fit(train: pd.DataFrame) -> dict:
    """fitted 값(윈저 경계·중앙값)을 train 에서만 계산한다."""
    lo, hi = train[WINSOR_COL].quantile([0.01, 0.99])
    medians = {}
    for c in NUM_COLS:
        s = train[c]
        if c == WINSOR_COL:
            s = s.clip(lo, hi)
        if c in LOG_COLS:
            s = np.log1p(s)
        medians[_out_name(c)] = float(s.median())
    return {"winsor": {"col": WINSOR_COL, "p01": float(lo), "p99": float(hi)},
            "medians": medians}


def _out_name(col: str) -> str:
    return f"{col}_log" if col in LOG_COLS else col


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """train 에서 fit 한 params 로 임의의 분할(train/test)을 변환한다."""
    out = df.copy()
    added = []

    for c in NUM_COLS:
        raw = out[c]
        # 플래그는 대체 '전'의 원본에서 계산한다
        out[f"{c}_missing"] = raw.isna().astype("uint8")
        if c in ANY_COLS:
            out[f"{c}_any"] = (raw > 0).fillna(False).astype("uint8")
            added.append(f"{c}_any")
        added.append(f"{c}_missing")

        s = raw
        if c == WINSOR_COL:
            s = s.clip(params["winsor"]["p01"], params["winsor"]["p99"])
        if c in LOG_COLS:
            s = np.log1p(s)
        name = _out_name(c)
        out[name] = s.fillna(params["medians"][name])
        if name != c:
            out = out.drop(columns=c)                  # 원본과 변환본 중복 방지

    # 검증: 수치 블록에 결측이 남아 있으면 안 된다
    check = [_out_name(c) for c in NUM_COLS]
    assert not out[check].isna().any().any(), "대체 후에도 결측이 남았다"
    assert len(out) == len(df)
    return out


def main() -> None:
    train = pd.read_parquet(AI_TRAIN_PARQUET)
    test = pd.read_parquet(AI_TEST_PARQUET)

    params = fit(train)
    train_t = transform(train, params)
    test_t = transform(test, params)                   # 같은 params — 누수 없음

    train_t.to_parquet(AI_TRAIN_LINEAR_PARQUET)
    test_t.to_parquet(AI_TEST_LINEAR_PARQUET)
    LINEAR_PARAMS_JSON.write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    w = params["winsor"]
    print(f"윈저 경계(train p1/p99): ${w['p01']:,.0f} ~ ${w['p99']:,.0f}")
    print(f"train  {train_t.shape[0]:,}행 × {train_t.shape[1]:,}열"
          f" (피처 +{train_t.shape[1] - train.shape[1]})  -> {AI_TRAIN_LINEAR_PARQUET.name}")
    print(f"test   {test_t.shape[0]:,}행 × {test_t.shape[1]:,}열"
          f"  -> {AI_TEST_LINEAR_PARQUET.name}")
    print(f"params -> {LINEAR_PARAMS_JSON.name}")

    # 왜도는 '관측값만'으로 진단한다 — 중앙값 대체가 만든 스파이크(최대 61.5%)를
    # 포함하면 통계가 왜곡된다. 대체 행은 _missing 플래그로 걸러낸다.
    print("\n[변환 후 왜도 (train, 관측값만) — 변환 전 대비]")
    before = {"ConvertedCompYearly": 50.86, "YearsCode": 1.19, "YearsCodePro": 1.34,
              "WorkExp": 1.25, "JobSatPoints_1": 1.72, "JobSatPoints_4": 3.40,
              "JobSatPoints_5": 2.78, "JobSatPoints_6": 1.39, "JobSatPoints_7": 1.52,
              "JobSatPoints_8": 1.67, "JobSatPoints_9": 2.00, "JobSatPoints_10": 2.64,
              "JobSatPoints_11": 2.78, "JobSat": -1.01, "AgeNum": 0.86,
              "EdLevelNum": -0.78, "OrgSizeNum": 0.23, "SOVisitFreqNum": 0.02}
    for c in NUM_COLS:
        name = _out_name(c)
        observed = train_t.loc[train_t[f"{c}_missing"] == 0, name]
        print(f"  {name:<26} {before[c]:+6.2f} -> {observed.skew():+.2f}")


if __name__ == "__main__":
    main()
