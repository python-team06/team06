"""로지스틱 베이스라인 결과 시각화 (5단계).

세 가지를 그린다:
    5_1  계수 상·하위 12개 — 피처를 축약 컬럼명이 아니라 so2024_dictionary 의
         **질문 원문**으로 표기한다 (예: "TechDoc__AI-powered..." 대신
         "What is the source of the technical documentation ...? ➜ AI-powered ...")
    5_2  ROC 곡선(test) + confusion matrix(test)

피처명 복원 규칙 — 파이프라인 접두어로 출처를 판별한다:
    num__X          수치형. X_log / XNum 등은 원 컬럼의 질문 + 변환 표기
    cat__X_V        단일선택 원핫. 컬럼 X(최장 접두어 매칭)의 질문 + 선택지 V
    remainder__B__O 멀티핫 더미. 문항 B 의 질문 + 선택지 O
    remainder__X_missing / _any   결측·0초과 플래그

실행:
    python -m src.visualize_logistic
"""

import re
import sys
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

try:
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        DATA_DIR,
        DICTIONARY_CSV,
        LOGISTIC_MODEL_JOBLIB,
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        AI_TEST_LINEAR_PARQUET,
        DATA_DIR,
        DICTIONARY_CSV,
        LOGISTIC_MODEL_JOBLIB,
    )

TOP_K = 12
OUT_COEF = DATA_DIR / "5_1_logistic_coef_fullname.png"
OUT_PERF = DATA_DIR / "5_2_logistic_performance.png"

# 차트 크롬 (dataviz 검증 팔레트: 파랑=사용(1), 주황=미사용(0) — 분포 차트와 동일)
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
C_YES, C_NO = "#2a78d6", "#eb6834"


def load_question_map() -> dict[str, str]:
    """column_silver -> 질문 원문 (HTML 태그 제거, 첫 물음표까지만)."""
    dic = pd.read_csv(DICTIONARY_CSV, dtype=str, keep_default_na=False)
    qmap = {}
    for r in dic.itertuples(index=False):
        if not r.column_silver or r.column_silver == "(드롭됨)":
            continue
        q = re.sub(r"<[^>]+>", " ", r.question)          # HTML 제거
        q = re.sub(r"\s+", " ", q).replace("*", "").strip()
        if "?" in q:
            q = q[: q.index("?") + 1]                     # "Select all that apply" 꼬리 제거
        qmap[r.column_silver] = q
    return qmap


def make_labeler(qmap: dict[str, str], cat_cols: list[str]):
    """모델 피처명 -> 질문 원문 라벨 함수를 만든다."""
    manual = {
        "ConvertedCompYearly": "연간 보수 (USD 환산)",
        "ResponseId": "응답자 ID",
    }
    by_len = sorted(cat_cols, key=len, reverse=True)      # 최장 접두어 매칭용

    def q(col: str) -> str:
        if col in manual:
            return manual[col]
        if col in qmap:
            return qmap[col]
        if col.endswith("Num") and col[:-3] in qmap:      # AgeNum -> Age
            return qmap[col[:-3]] + " (순서형)"
        return col

    def clean_value(v: str) -> str:
        if v == "Missing":
            return "(무응답)"
        if v == "infrequent_sklearn":
            return "(희소 수준 묶음)"
        return v

    def two_line(question: str, sub: str) -> str:
        """1행 = 질문(잘릴 수 있음), 2행 = 선택지/변환 표기(항상 온전히 보임).

        선택지가 피처를 구분하는 핵심 정보라서, 질문이 아니라 질문 쪽을 자른다.
        """
        head = textwrap.shorten(question, width=66, placeholder="…")
        return f"{head}\n{textwrap.shorten(sub, width=58, placeholder='…')}"

    def label(feat: str) -> str:
        group, name = feat.split("__", 1)
        if group == "num":                                # 수치형
            if name.endswith("_log"):
                return two_line(q(name[:-4]), "— log 척도")
            return two_line(q(name), "— 수치")
        if group == "cat":                                # 단일선택 원핫: {col}_{value}
            col = next((c for c in by_len if name.startswith(c + "_")), None)
            if col is None:
                return name
            return two_line(q(col), f"➜ {clean_value(name[len(col) + 1:])}")
        # remainder: 멀티핫 더미 / 플래그
        if "__" in name:
            base, opt = name.split("__", 1)
            if opt == "answered":
                return two_line(q(base), "— 응답 여부")
            return two_line(q(base), f"➜ {opt}")
        if name.endswith("_missing"):
            return two_line(q(name[: -len('_missing')]), "— 결측 여부")
        if name.endswith("_any"):
            return two_line(q(name[: -len('_any')]), "— 0점 초과 여부")
        return q(name)

    return label


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "font.sans-serif": ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                            "Helvetica", "Arial"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
    })


def plot_coefficients(model, labeler) -> None:
    names = model[:-1].get_feature_names_out()
    coefs = model[-1].coef_[0]
    order = np.argsort(coefs)
    idx = list(order[::-1][:TOP_K]) + list(order[:TOP_K][::-1])   # +상위 -> -상위
    vals = coefs[idx]
    labels = [labeler(names[i]) for i in idx]
    colors = [C_YES if v > 0 else C_NO for v in vals]

    fig, ax = plt.subplots(figsize=(13.5, 12.5), dpi=150)
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, height=0.62, color=colors)
    for y, v in zip(ypos, vals):
        ax.text(v + (0.03 if v > 0 else -0.03), y, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=BASE, linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("표준화 계수 (0 기준 오른쪽 = AI 사용(1) 방향)", fontsize=9)
    ax.set_title("로지스틱 회귀 계수 상·하위 12 — 질문 원문 표기\n", loc="left",
                 fontsize=13, fontweight="bold", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_YES),
               plt.Rectangle((0, 0), 1, 1, color=C_NO)]
    ax.legend(handles, ["AI 사용(1) 방향", "미사용(0) 방향"], frameon=False,
              loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_COEF, facecolor=SURFACE, bbox_inches="tight")
    print(f"저장: {OUT_COEF.name}")


def plot_performance(model) -> None:
    test = pd.read_parquet(AI_TEST_LINEAR_PARQUET)
    y = test.pop("target")
    for c in test.select_dtypes(include="category").columns:
        test[c] = test[c].astype(object)
    prob = model.predict_proba(test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ROC
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    ax1.plot(fpr, tpr, color=C_YES, linewidth=2, label=f"로지스틱 (AUC {auc:.3f})")
    ax1.plot([0, 1], [0, 1], color=BASE, linewidth=1, linestyle="--",
             label="무작위 기준 (0.5)")
    ax1.set_xlabel("위양성률 (FPR)", fontsize=9)
    ax1.set_ylabel("재현율 (TPR)", fontsize=9)
    ax1.set_title("ROC 곡선 — test 1회 평가", loc="left", fontsize=11,
                  fontweight="bold", color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.grid(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    ax2.imshow(cm, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#1c5cab"]), alpha=0.9)
    ticks = ["미사용(0)", "사용(1)"]
    ax2.set_xticks([0, 1], [f"예측 {t}" for t in ticks], fontsize=9)
    ax2.set_yticks([0, 1], [f"실제 {t}" for t in ticks], fontsize=9)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            color = "#ffffff" if cm[i, j] > cm.max() * 0.5 else INK
            ax2.text(j, i, f"{cm[i, j]:,}\n(행의 {share:.0%})",
                     ha="center", va="center", fontsize=11, color=color)
    ax2.set_title("Confusion matrix — threshold 0.5", loc="left", fontsize=11,
                  fontweight="bold", color=INK)

    fig.tight_layout()
    fig.savefig(OUT_PERF, facecolor=SURFACE, bbox_inches="tight")
    print(f"저장: {OUT_PERF.name}")


def main() -> None:
    setup_style()
    model = joblib.load(LOGISTIC_MODEL_JOBLIB)
    qmap = load_question_map()
    cat_cols = (pd.read_parquet(AI_TEST_LINEAR_PARQUET)
                .select_dtypes(include="category").columns.tolist())
    labeler = make_labeler(qmap, cat_cols)
    plot_coefficients(model, labeler)
    plot_performance(model)


if __name__ == "__main__":
    main()
