"""AISelect(AI 도구 사용 여부) 예측 모델용 train/test 분할 (3단계).

gold_wide 를 입력으로 다음을 수행한다:

    1) 타깃 정의    AISelect 무응답 제외, target = 1 if AISelect == "Yes" else 0
    2) 누수 제거    AISelect 값에 따라 조건부로 노출되는 꼬리질문을
                    컬럼 가족(원본 / __answered / __더미) 단위로 제거
    3) 층화 분할    train : test = 0.8 : 0.2, seed=42, stratify=target
    4) 배포 산출물  ai_train / ai_test parquet + ResponseId 목록 + 매니페스트

누수 제거 근거 — 전 그룹 응답률을 실측한 결과다:

    - Yes 전용 꼬리질문 11개 (AIBen, AIAcc, AIComplex, AITool* 3, AINext* 5):
      AISelect=Yes 그룹만 90~99% 응답, 나머지 그룹은 응답률 0.0%
    - Yes+도입계획 공용 꼬리질문 4개 (AISent, AIThreat, AIEthics, AIChallenges):
      "No, and I don't plan to"/무응답 그룹에서 응답률 0.0%
      -> 두 부류 모두 '응답했다는 사실' 자체가 타깃을 노출한다
    - 준누수 2개 (AISearchDevHaveWorkedWith, AISearchDevWantToWorkWith):
      구조적 분기는 아니지만(전 그룹 노출) 내용이 타깃의 동어반복
      (AI 도구 사용 경험: 그룹별 응답률 5.7 -> 33.9 -> 61.9 -> 90.2%)

실행:
    python -m src.split_aiselect
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        AI_MANIFEST_JSON,
        AI_SPLIT_IDS_CSV,
        AI_TEST_PARQUET,
        AI_TRAIN_PARQUET,
        GOLD_WIDE_PARQUET,
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        AI_MANIFEST_JSON,
        AI_SPLIT_IDS_CSV,
        AI_TEST_PARQUET,
        AI_TRAIN_PARQUET,
        GOLD_WIDE_PARQUET,
    )

TEST_SIZE = 0.2
SEED = 42

# 제거할 컬럼 가족의 '기준 이름'. wide 에서는 기준 이름 그대로(단일선택),
# <기준>__answered, <기준>__<선택지> (다중선택) 형태로 존재하므로 접두어로 확장한다.
LEAK_BASES = [
    # Yes 전용 꼬리질문 (11)
    "AIBen", "AIAcc", "AIComplex",
    "AIToolCurrentlyUsing", "AIToolInterestedInUsing", "AIToolNotInterestedInUsing",
    "AINextMuchMoreIntegrated", "AINextMoreIntegrated", "AINextNoChange",
    "AINextLessIntegrated", "AINextMuchLessIntegrated",
    # Yes + "No, but I plan to soon" 공용 꼬리질문 (4)
    "AISent", "AIThreat", "AIEthics", "AIChallenges",
    # 준누수: 타깃과 동어반복 (2)
    "AISearchDevHaveWorkedWith", "AISearchDevWantToWorkWith",
]

TARGET = "AISelect"


def expand_leak_columns(columns: pd.Index) -> list[str]:
    """기준 이름 17개를 wide 의 실제 컬럼 가족으로 확장한다."""
    dropped = [
        c for c in columns
        if any(c == base or c.startswith(base + "__") for base in LEAK_BASES)
    ]
    # 기준 이름이 하나도 안 걸리면 컬럼명이 바뀌었다는 뜻이므로 바로 터뜨린다.
    for base in LEAK_BASES:
        if not any(c == base or c.startswith(base + "__") for c in dropped):
            raise ValueError(f"누수 기준 '{base}' 에 해당하는 컬럼이 없다 — 컬럼명 변경 확인 필요")
    return dropped


def main() -> None:
    wide = pd.read_parquet(GOLD_WIDE_PARQUET)
    n_total = len(wide)

    # ---- 1) 타깃 정의: 무응답 제외 후 이진화 ----
    labeled = wide[wide[TARGET].notna()].copy()
    n_dropped_na = n_total - len(labeled)
    labeled["target"] = (labeled[TARGET] == "Yes").astype("int8")

    # ---- 2) 누수 컬럼 가족 + 타깃 원본 제거, ResponseId 는 인덱스로 ----
    leak_cols = expand_leak_columns(labeled.columns)
    df = (
        labeled.drop(columns=leak_cols + [TARGET])
        .set_index("ResponseId")  # 식별자: 피처에 섞이지 않도록 인덱스로 뺀다
    )

    # ---- 3) 층화 분할 ----
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["target"]
    )

    # ---- 검증: 여기서 어긋나면 배포하면 안 된다 ----
    assert len(train) + len(test) == len(df)
    assert not train.index.intersection(test.index).size, "train/test 에 같은 응답자가 있다"
    remaining = [
        c for c in df.columns if any(c == b or c.startswith(b + "__") for b in LEAK_BASES)
    ]
    assert not remaining, f"누수 컬럼이 남아 있다: {remaining[:3]}"

    # ---- 4) 산출물 저장 ----
    train.to_parquet(AI_TRAIN_PARQUET)  # index=True 기본값: ResponseId 보존
    test.to_parquet(AI_TEST_PARQUET)

    ids = pd.concat([
        pd.Series("train", index=train.index, name="split"),
        pd.Series("test", index=test.index, name="split"),
    ])
    ids.sort_index().to_csv(AI_SPLIT_IDS_CSV)

    manifest = {
        "생성일": date.today().isoformat(),
        "입력": GOLD_WIDE_PARQUET.name,
        "타깃": 'target = 1 if AISelect == "Yes" else 0',
        "무응답_제외": n_dropped_na,
        "분할": {"test_size": TEST_SIZE, "random_state": SEED, "stratify": "target"},
        "행수": {"train": len(train), "test": len(test)},
        "양성비율": {
            "train": round(float(train["target"].mean()), 4),
            "test": round(float(test["target"].mean()), 4),
        },
        "제거_기준_17개": LEAK_BASES,
        "제거_컬럼_전체": leak_cols,
        "규칙": [
            "test 는 최종 평가 1회만 사용한다 (모델·피처 선택은 train 내부 CV 로)",
            "target encoding, 스케일링, 통계 기반 대체 등 fitted 변환은 train 에서만 fit 한다",
            "ResponseId 는 인덱스(식별자)다 — 피처로 사용 금지",
            "3클래스 분석이 필요하면 ai_split_ids.csv 를 silver 와 조인해 같은 분할을 재현한다",
        ],
    }
    AI_MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 요약 출력 ----
    print(f"입력            {n_total:,}행 (무응답 {n_dropped_na:,}명 제외 -> {len(df):,}행)")
    print(f"누수 제거        기준 {len(LEAK_BASES)}개 -> 실제 컬럼 {len(leak_cols)}개 (+ {TARGET})")
    print(f"피처 수          {df.shape[1] - 1:,}개 (+ target)")
    print(f"train           {len(train):,}행, 양성 {train['target'].mean():.1%}"
          f"  -> {AI_TRAIN_PARQUET.name}")
    print(f"test            {len(test):,}행, 양성 {test['target'].mean():.1%}"
          f"  -> {AI_TEST_PARQUET.name}")
    print(f"기타            {AI_SPLIT_IDS_CSV.name}, {AI_MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
