"""Stack Overflow Survey 2024 전처리 파이프라인 (0~2단계).

results.csv(원본)를 통계·ML 분석이 바로 가능한 형태로 변환한다.

    0단계  로딩 규약     "NA" 문자열만 결측으로 읽는다 (다른 값은 건드리지 않는다)
    1단계  구조 정리     드롭 / 컬럼명 공백 제거 / sentinel 숫자화 / 순서형 인코딩
                         -> data/silver.parquet    (원본과 같은 1행 = 응답자 1명)
    2단계  다중선택 처리  ';' 로 묶인 57개 컬럼을 두 가지 형태로 푼다
                         -> data/gold_wide.parquet (ML용: 0/1 멀티핫 + __answered 플래그)
                         -> data/gold_long.parquet (통계용: 응답자 × 선택지 1행)

설계 근거 — 전부 실데이터 전수 검증을 거친 사실이다:
    - *Admired 11개 컬럼은 *HaveWorkedWith ∩ *WantToWorkWith 와 정확히 일치하는
      파생 컬럼이다(11개 트리오 모두 100% 일치 확인). 피처 행렬(wide)에서는
      순수 중복이므로 제외하고, 집계용(long)에는 남긴다.
    - 그리드형 문항 컬럼(OpSys*/AITool*/AINext*)에는 결측("NA")과 빈 문자열("")이
      둘 다 존재한다. "NA" = 질문 자체를 안 봄, "" = 답했지만 그 버킷은 비워둠.
      따라서 __answered 플래그는 "NA"만 미응답으로 센다.

실행:
    python -m src.preprocess
"""

import sys
import time
from pathlib import Path

import pandas as pd

try:
    from src.config import (
        GOLD_LONG_PARQUET,
        GOLD_WIDE_PARQUET,
        MULTI_SEP,
        RESULTS_CSV,
        SILVER_PARQUET,
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        GOLD_LONG_PARQUET,
        GOLD_WIDE_PARQUET,
        MULTI_SEP,
        RESULTS_CSV,
        SILVER_PARQUET,
    )

# ---------- 1단계 규칙 ----------

# 정보량이 없거나(상수), 더 나은 대체 컬럼이 있는 컬럼
DROP_COLS = [
    "Check",      # 어텐션 체크. 공개본은 전원 "Apples" -> 상수, 정보량 0
    "CompTotal",  # 응답자 각자의 통화 단위라 그대로는 비교 불가. USD 환산본만 쓴다
    "Currency",   # CompTotal 과 세트로만 의미가 있으므로 함께 제거
]

# 숫자 컬럼에 섞인 텍스트 보기(sentinel)의 수치 치환
SENTINELS = {"Less than 1 year": 0.5, "More than 50 years": 51.0}
SENTINEL_COLS = ["YearsCode", "YearsCodePro"]

# 순서형 범주 -> 숫자 매핑. 원본 라벨 컬럼은 차트용으로 남기고 <이름>Num 을 추가한다.
# (원핫과 달리 '많다/적다'의 순서 정보가 선형 모델에 살아남는다)
# 값 문자열은 실데이터에서 그대로 복사했다 — EdLevel/OrgSize 는 곧은따옴표(')가 아니라
# 굽은따옴표(’)를 쓰므로 손으로 다시 치면 안 맞는다.
# None 은 "순서를 매길 수 없는 응답" 표시로, NaN 이 된다.
ORDINALS: dict[str, dict[str, float | None]] = {
    "Age": {  # 구간 중앙값 (Under 18 은 SO 응답 연령 분포를 고려해 16)
        "Under 18 years old": 16.0,
        "18-24 years old": 21.0,
        "25-34 years old": 29.5,
        "35-44 years old": 39.5,
        "45-54 years old": 49.5,
        "55-64 years old": 59.5,
        "65 years or older": 70.0,
        "Prefer not to say": None,
    },
    "EdLevel": {  # 학력 단계 1~7
        "Primary/elementary school": 1.0,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2.0,
        "Some college/university study without earning a degree": 3.0,
        "Associate degree (A.A., A.S., etc.)": 4.0,
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 5.0,
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 6.0,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7.0,
        "Something else": None,
    },
    "OrgSize": {  # 조직 규모 1~9
        "Just me - I am a freelancer, sole proprietor, etc.": 1.0,
        "2 to 9 employees": 2.0,
        "10 to 19 employees": 3.0,
        "20 to 99 employees": 4.0,
        "100 to 499 employees": 5.0,
        "500 to 999 employees": 6.0,
        "1,000 to 4,999 employees": 7.0,
        "5,000 to 9,999 employees": 8.0,
        "10,000 or more employees": 9.0,
        "I don’t know": None,
    },
    "SOVisitFreq": {  # 방문 빈도 1(드묾)~5(하루 여러 번)
        "Less than once per month or monthly": 1.0,
        "A few times per month or weekly": 2.0,
        "A few times per week": 3.0,
        "Daily or almost daily": 4.0,
        "Multiple times per day": 5.0,
    },
}

# 2단계에서 피처 행렬(wide)에 넣지 않는 파생 컬럼의 접미사
ADMIRED_SUFFIX = "Admired"


# ---------- 0단계: 로딩 ----------

def load_raw() -> pd.DataFrame:
    """이 데이터의 "NA" 문자열만 결측으로 읽는다.

    keep_default_na=False 가 없으면 pandas 가 "None", "N/A" 같은 값까지
    제멋대로 결측 처리한다. 이 파일의 결측 표기는 "NA" 하나뿐이므로
    정확히 그것만 지정한다. dtype 은 추론에 맡긴다 — 그러면 ResponseId,
    JobSatPoints_* 같은 순수 숫자 컬럼은 알아서 숫자로 들어온다.
    """
    return pd.read_csv(RESULTS_CSV, keep_default_na=False, na_values=["NA"])


# ---------- 1단계: 구조 정리 ----------

def _no_space(name: str) -> str:
    """공백 포함 컬럼명을 공백 없는 이름으로 바꾼다.

    'AIToolInterested in Using' -> 'AIToolInterestedInUsing'
    공백이 있으면 df.컬럼명 접근이 안 되고 쿼리 엔진에서도 사고가 난다.
    """
    words = name.split(" ")
    return words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])


def to_silver(raw: pd.DataFrame) -> pd.DataFrame:
    """구조 정리. 행 수는 절대 바뀌지 않는다(1행 = 응답자 1명)."""
    df = raw.drop(columns=DROP_COLS)
    df = df.rename(columns={c: _no_space(c) for c in df.columns if " " in c})

    # sentinel 치환 후 숫자로. 예상 밖의 텍스트가 있으면 조용히 NaN 이 되는 대신
    # 여기서 바로 터뜨린다 — 데이터가 바뀌었는데 모르고 지나가는 사고 방지.
    for c in SENTINEL_COLS:
        as_num = pd.to_numeric(df[c].replace(SENTINELS), errors="coerce")
        lost = df[c].notna() & as_num.isna()
        if lost.any():
            raise ValueError(f"{c}: 숫자로 못 바꾼 값 {df.loc[lost, c].unique()[:5]}")
        df[c] = as_num

    # 순서형 인코딩. 매핑에 없는 값이 나타나면 (설문 개정 등) 역시 바로 터뜨린다.
    for col, mapping in ORDINALS.items():
        observed = set(df[col].dropna().unique())
        unknown = observed - mapping.keys()
        if unknown:
            raise ValueError(f"{col}: 매핑에 없는 값 {sorted(unknown)[:3]}")
        df[col + "Num"] = df[col].map(mapping)

    return df


# ---------- 2단계: 다중선택 처리 ----------

def find_multiselect(df: pd.DataFrame) -> list[str]:
    """';' 가 실제로 등장하는 문자열 컬럼 = 다중선택 문항.

    na=False 가 중요하다: 결측을 True 로 오판하는 것을 막는다.
    """
    return [
        c for c in df.columns
        if df[c].dtype == object
        and df[c].str.contains(MULTI_SEP, regex=False, na=False).any()
    ]


def to_gold_wide(silver: pd.DataFrame, multi: list[str]) -> pd.DataFrame:
    """ML용 wide: 다중선택을 0/1 멀티핫으로 편다.

    - 컬럼마다 <이름>__answered 플래그를 함께 만든다. 멀티핫만 있으면
      '그 항목을 안 골랐다'(0)와 '질문에 답하지 않았다'(전부 0)가 구분되지 않는다.
    - *Admired 는 뺀다: Have ∩ Want 와 전수 일치하는 파생 컬럼이라 순수 중복이다.
    - 남은 문자열 컬럼은 category 로 바꾼다: parquet 이 작아지고
      LightGBM 등이 인코딩 없이 바로 받는다.
    """
    feature_multi = [c for c in multi if not c.endswith(ADMIRED_SUFFIX)]

    parts = [silver.drop(columns=multi)]
    for c in feature_multi:
        parts.append(silver[c].notna().rename(f"{c}__answered"))
        # ""(답은 했지만 빈 버킷)는 NA 로 바꿔 get_dummies 가 전부-0 행으로 만들게 한다.
        cell = silver[c].replace("", pd.NA)
        parts.append(
            cell.str.get_dummies(sep=MULTI_SEP).add_prefix(f"{c}__").astype("uint8")
        )

    wide = pd.concat(parts, axis=1)
    for c in wide.select_dtypes(include="object").columns:
        wide[c] = wide[c].astype("category")
    return wide


def to_gold_long(silver: pd.DataFrame, multi: list[str]) -> pd.DataFrame:
    """통계용 long: (응답자, 문항, 선택지) 1행짜리 세로 형태.

    "언어별 사용자 수" 같은 집계는 멀티핫 1,000여 컬럼보다 이 형태가 편하다.
    *Admired 도 여기엔 남긴다 — '가장 사랑받는 기술' 차트의 원천이다.
    """
    frames = []
    for c in multi:
        s = silver[["ResponseId", c]].dropna(subset=[c])
        s = s.assign(value=s[c].str.split(MULTI_SEP)).explode("value")
        s = s[s["value"] != ""]  # 빈 버킷("")은 행을 만들지 않는다
        frames.append(
            pd.DataFrame(
                {"ResponseId": s["ResponseId"], "question": c, "value": s["value"]}
            )
        )
    long = pd.concat(frames, ignore_index=True)
    long["question"] = long["question"].astype("category")
    long["value"] = long["value"].astype("category")
    return long


# ---------- 실행 ----------

def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> None:
    t0 = time.perf_counter()

    raw = load_raw()
    print(f"[0단계] 로드           {raw.shape[0]:,}행 × {raw.shape[1]}열")

    silver = to_silver(raw)
    assert len(silver) == len(raw), "행 수가 변하면 버그다"
    assert not any(" " in c for c in silver.columns), "공백 컬럼명이 남아 있다"
    silver.to_parquet(SILVER_PARQUET, index=False)
    print(f"[1단계] silver         {silver.shape[0]:,}행 × {silver.shape[1]}열"
          f"  -> {SILVER_PARQUET.name} ({_mb(SILVER_PARQUET)})")

    multi = find_multiselect(silver)
    print(f"[2단계] 다중선택 {len(multi)}개 검출"
          f" (wide 에서는 *Admired {sum(c.endswith(ADMIRED_SUFFIX) for c in multi)}개 제외)")

    wide = to_gold_wide(silver, multi)
    assert len(wide) == len(raw)
    wide.to_parquet(GOLD_WIDE_PARQUET, index=False)
    print(f"        gold_wide      {wide.shape[0]:,}행 × {wide.shape[1]:,}열"
          f"  -> {GOLD_WIDE_PARQUET.name} ({_mb(GOLD_WIDE_PARQUET)})")

    long = to_gold_long(silver, multi)
    long.to_parquet(GOLD_LONG_PARQUET, index=False)
    print(f"        gold_long      {long.shape[0]:,}행 × {long.shape[1]}열"
          f"  -> {GOLD_LONG_PARQUET.name} ({_mb(GOLD_LONG_PARQUET)})")

    print(f"\n완료: {time.perf_counter() - t0:.1f}초")


if __name__ == "__main__":
    main()
