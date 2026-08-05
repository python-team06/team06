"""
EDA 시각화 (seaborn)

gold_long.parquet / gold_wide.parquet / silver.parquet 를 불러와
seaborn으로 시각화한다.

실행:
    python3 -m src.visualize
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

from src.config import GOLD_LONG_PARQUET, GOLD_WIDE_PARQUET, SILVER_PARQUET

sns.set_theme(style="whitegrid")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

def load_data():
    silver = pd.read_parquet(SILVER_PARQUET)
    gold_wide = pd.read_parquet(GOLD_WIDE_PARQUET)
    gold_long = pd.read_parquet(GOLD_LONG_PARQUET)
    return silver, gold_wide, gold_long


# ========== 1. gold_long (다중선택 57개 응답 분석) ==========


# 특정 질문에 대해 응답 빈도가 가장 높은 top N개의 항목을 가로바 차트(sns.barplot)로 그린다
def plot_top_n(gold_long: pd.DataFrame, question: str, top_n: int = 15):
    """1-1. 주요 언어/도구 top N 빈도"""
    sub = gold_long[gold_long["question"] == question]
    counts = sub["value"].astype(str).value_counts().head(top_n)  # astype(str) 추가
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"{question} (Top {top_n})")
    ax.set_xlabel("응답자 수")
    fig.tight_layout()
    return fig


def plot_have_vs_want(gold_long: pd.DataFrame, base: str, top_n: int = 15):
    """1-2. 현재(HaveWorkedWith) vs 선호(WantToWorkWith) 비교."""
    have_q, want_q = f"{base}HaveWorkedWith", f"{base}WantToWorkWith"
    have = gold_long[gold_long["question"] == have_q]["value"].astype(str).value_counts()
    want = gold_long[gold_long["question"] == want_q]["value"].astype(str).value_counts()

    top_items = have.head(top_n).index
    df = pd.DataFrame({"Have": have.reindex(top_items), "Want": want.reindex(top_items)})
    df.index.name = base
    df = df.reset_index()
    df = df.melt(id_vars=base, var_name="type", value_name="count")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=df, y=base, x="count", hue="type", ax=ax)
    ax.set_title(f"{base}: 현재 사용 vs 선호 (Top {top_n})")
    fig.tight_layout()
    return fig

# gold_long에는 없는 직군 정보(DevType)를 silver 데이터셋에서 ResponseId 기준 Mapping/Join으로 가져온다
# pd.crosstab()을 통해 직군별 특정 기술 사용 비율을 행기준 정규화한 후 sns.heatmap으로 시각화
def plot_tech_by_devtype(silver: pd.DataFrame, gold_long: pd.DataFrame,
                          question: str = "LanguageHaveWorkedWith",
                          top_devtype: int = 8, top_tech: int = 10):
    """1-3. 직군(DevType)별 주요 기술 스택 교차 분석.

    DevType은 gold_long에 없으므로(단일선택 취급) silver에서 ResponseId로 join.
    """
    top_dt = silver["DevType"].value_counts().head(top_devtype).index
    dt_map = silver.set_index("ResponseId")["DevType"]

    sub = gold_long[gold_long["question"] == question].copy()
    sub["DevType"] = sub["ResponseId"].map(dt_map)
    sub = sub[sub["DevType"].isin(top_dt)]

    top_tech_items = sub["value"].value_counts().head(top_tech).index
    sub = sub[sub["value"].isin(top_tech_items)]

    cross = pd.crosstab(sub["DevType"], sub["value"], normalize="index")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"직군별 {question} 사용 비율 (행 기준 정규화)")
    fig.tight_layout()
    return fig


# ========== 2. silver (수치형 19개) ==========

# 오른쪽으로 긴 꼬리를 갖는 연봉 데이터에는 log_scale=True를 적용해 로그 스케일 분포 확인
# 경력 변수 (YearCode, YearsCodePro 등)에는 sns.histplot으로 히스토그램 작성
def plot_salary_experience_dist(silver: pd.DataFrame):
    """2-1. 연봉/경력 분포. 오른쪽 꼬리 긴 분포 예상 -> log_scale."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(silver["ConvertedCompYearly"].dropna(), bins=50, log_scale=True, ax=axes[0])
    axes[0].set_title("ConvertedCompYearly (log)")

    sns.histplot(silver["YearsCode"].dropna(), bins=30, ax=axes[1])
    axes[1].set_title("YearsCode")

    sns.histplot(silver["YearsCodePro"].dropna(), bins=30, ax=axes[2])
    axes[2].set_title("YearsCodePro")
    fig.tight_layout()
    return fig


# 연봉 극단값을 제거하여 트렌드가 일그러지는 것 방지
# 경력/나이에 따른 연봉 변화를 산점도, 추세선으로 시각화
def plot_exp_age_vs_salary(silver: pd.DataFrame, cap_quantile: float = 0.99):
    """2-2. 경력/나이에 따른 연봉 변화. 극단 이상치는 상위 분위수로 컷."""
    cap = silver["ConvertedCompYearly"].quantile(cap_quantile)
    df = silver[silver["ConvertedCompYearly"] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=df, x="YearsCodePro", y="ConvertedCompYearly",
                scatter_kws={"alpha": 0.1, "s": 10}, line_kws={"color": "red"}, ax=axes[0])
    axes[0].set_title("경력(YearsCodePro) vs 연봉")

    sns.regplot(data=df, x="AgeNum", y="ConvertedCompYearly",
                scatter_kws={"alpha": 0.1, "s": 10}, line_kws={"color": "red"}, ax=axes[1])
    axes[1].set_title("나이(AgeNum) vs 연봉")
    fig.tight_layout()
    return fig


# 수치형 변수 간의 상관관계 계산한 뒤 전체적인 상관관계 시각화
def plot_numeric_corr(silver: pd.DataFrame):
    """2-3. 수치형 변수 간 상관관계 히트맵."""
    numeric_cols = silver.select_dtypes(include=["number"]).columns.drop("ResponseId")
    corr = silver[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("수치형 변수 상관관계")
    fig.tight_layout()
    return fig


# ========== 3. silver (단일 범주형 39개) ==========

# Cardinality(고유값 개수)의 크기에 맞춰 다르게 시각화 

# ORDINALS 순서(원본 라벨 기준). config.py의 ORDINALS와 동기화 필요.
ORDINAL_ORDER = {
    "Age": ["Under 18 years old", "18-24 years old", "25-34 years old", "35-44 years old",
            "45-54 years old", "55-64 years old", "65 years or older", "Prefer not to say"],
    "EdLevel": None,  # 라벨이 길어 Num 기준 정렬 권장
    "OrgSize": None,
    "SOVisitFreq": [
        "Less than once per month or monthly", "A few times per month or weekly",
        "A few times per week", "Daily or almost daily", "Multiple times per day",
    ],
}


# 고유값이 적은 항목들을 서브플롯 그리드로 한번에 처리
# 서열 정보(ORDINAL_ORDER)를 반영하거나 빈도순으로 정렬
def plot_low_cardinality(silver: pd.DataFrame, cols: list[str], wrap_width: int = 30):
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        order = ORDINAL_ORDER.get(col) or silver[col].value_counts().index
        sns.countplot(data=silver, y=col, order=order, ax=ax)
        # 긴 라벨 줄바꿈
        wrapped = ["\n".join(textwrap.wrap(str(label), wrap_width)) for label in order]
        ax.set_yticks(range(len(wrapped)))
        ax.set_yticklabels(wrapped, fontsize=8)
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# 국가와 같이 항목 수가 많은 변수는 상위 20개만 슬라이싱하여 가로 바 차트로 시각화
def plot_high_cardinality(silver: pd.DataFrame, col: str, top_n: int = 20):
    """3-2. cardinality 큰 변수(Country, DevType 등) top N."""
    counts = silver[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"{col} (Top {top_n})")
    fig.tight_layout()
    return fig


# ========== 4. gold_wide (3개 이상 변수 동시 비교) ==========

# One-Hot 또는 Wide 포맷 데이터를 활용해 3개 이상의 변수 관계를 복합 시각화
def plot_catplot_multi(silver: pd.DataFrame, gold_wide: pd.DataFrame,
                        tech_col: str, x: str, hue: str):
    """4-1. catplot으로 3개 변수(기술 보유여부 x 범주1 x 범주2) 동시 비교.

    tech_col 예: "LanguageHaveWorkedWith__Python" (gold_wide 멀티핫 컬럼명)
    """
    df = gold_wide[["ResponseId", tech_col]].merge(
        silver[["ResponseId", x, hue]], on="ResponseId", how="left"
    )
    df = df.dropna(subset=[x, hue])

    g = sns.catplot(data=df, x=x, y=tech_col, hue=hue, kind="bar",
                     estimator="mean", height=5, aspect=1.8)
    g.set_axis_labels(x, f"{tech_col} 사용 비율")
    g.figure.suptitle(f"{tech_col} 사용률: {x} x {hue}", y=1.02)
    return g


# ========== 실행 ==========

def main():
    silver, gold_wide, gold_long = load_data()

    # 1. 다중선택
    plot_top_n(gold_long, "LanguageHaveWorkedWith").savefig(
        "data/1_1_top_languages.png", dpi=150, bbox_inches="tight")
    plot_have_vs_want(gold_long, "Language").savefig(
        "data/1_2_language_have_vs_want.png", dpi=150, bbox_inches="tight")
    plot_tech_by_devtype(silver, gold_long).savefig(
        "data/1_3_lang_by_devtype.png", dpi=150, bbox_inches="tight")

    # 2. 수치형
    plot_salary_experience_dist(silver).savefig(
        "data/2_1_salary_exp_dist.png", dpi=150, bbox_inches="tight")
    plot_exp_age_vs_salary(silver).savefig(
        "data/2_2_exp_age_vs_salary.png", dpi=150, bbox_inches="tight")
    plot_numeric_corr(silver).savefig(
        "data/2_3_numeric_corr.png", dpi=150, bbox_inches="tight")

     # 3. 단일 범주형 — 라벨 짧은 것만 그리드로
    low_card_cols = ["RemoteWork", "AIThreat", "TBranch"]
    plot_low_cardinality(silver, low_card_cols).savefig(
        "data/3_1_low_cardinality.png", dpi=150, bbox_inches="tight")

    # 라벨이 긴 것들은 개별 시각화
    plot_high_cardinality(silver, "ICorPM", top_n=5).savefig(
        "data/3_1b_icorpm.png", dpi=150, bbox_inches="tight")
    plot_high_cardinality(silver, "BuildvsBuy", top_n=5).savefig(
        "data/3_1c_buildvsbuy.png", dpi=150, bbox_inches="tight")

    plot_high_cardinality(silver, "Country").savefig(
        "data/3_2_country_top20.png", dpi=150, bbox_inches="tight")

    # 4. gold_wide 복합 시각화
    plot_catplot_multi(
        silver, gold_wide,
        tech_col="LanguageHaveWorkedWith__Python",
        x="RemoteWork",
        hue="EdLevel",
    ).savefig("data/4_1_python_by_remote_edlevel.png", dpi=150, bbox_inches="tight")

    print("완료")


if __name__ == "__main__":
    main()