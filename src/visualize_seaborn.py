"""
EDA 시각화 (seaborn)

data/ai_train.parquet 을 불러와 seaborn으로 시각화한다.
(gold_long/gold_wide/silver 3분할 대신 target 포함 단일 wide 테이블 사용)

실행:
    python3 visualize_ai_train.py
    (프로젝트 루트에서 실행)
"""
import base64
import io
import textwrap
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # src/ 의 상위 = team06/
DATA_PATH = PROJECT_ROOT / "data" / "ai_train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data"  # HTML 4개가 저장될 폴더

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 카테고리별로 섹션(html 조각)을 누적
_sections = {
    "target": [],
    "numeric": [],
    "categorical": [],
    "multivariate": [],
}

# 카테고리 -> 저장될 파일명, 리포트 제목
_REPORT_META = {
    "target": ("report_target.html", "Target Distribution Report"),
    "numeric": ("report_numeric.html", "Numeric Features Report"),
    "categorical": ("report_categorical.html", "Categorical Features Report"),
    "multivariate": ("report_multivariate.html", "Multivariate Analysis Report"),
}


def load_data():
    return pd.read_parquet(DATA_PATH)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def add_section(title: str, fig, category: str):
    """fig(또는 catplot의 FacetGrid)를 base64 이미지로 변환해 해당 category 리포트에 추가."""
    if category not in _sections:
        raise ValueError(f"알 수 없는 category: {category} (가능: {list(_sections.keys())})")
    target_fig = fig.figure if hasattr(fig, "figure") else fig
    src = fig_to_base64(target_fig)
    _sections[category].append(
        f'<section><h2>{title}</h2><img src="{src}" alt="{title}"></section>'
    )


def _render_html(title: str, sections_html: list[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  section {{ margin: 40px 0; }}
  h2 {{ color: #333; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{"".join(sections_html)}
</body>
</html>"""


def save_report(open_in_browser: bool = True):
    """카테고리별로 모아둔 섹션을 각각 별도 HTML 파일로 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for category, (filename, title) in _REPORT_META.items():
        sections_html = _sections[category]
        if not sections_html:
            print(f"[건너뜀] '{category}' — 저장된 섹션 없음")
            continue

        html = _render_html(title, sections_html)
        path = OUTPUT_DIR / filename
        path.write_text(html, encoding="utf-8")
        print(f"저장됨: {path.resolve()}")
        saved_paths.append(path)

    if open_in_browser and saved_paths:
        # 첫 번째 리포트만 자동으로 열기 (전체 다 열고 싶으면 반복문으로 변경)
        webbrowser.open(f"file://{saved_paths[0].resolve()}")


# ========== 0. 컬럼 타입 자동 분류 ==========

def get_onehot_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    marker = f"{prefix}__"
    return [c for c in df.columns if c.startswith(marker)]


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["category", "object"]).columns.tolist()


def get_numeric_cols(df: pd.DataFrame, exclude: tuple = ("target",)) -> list[str]:
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in cols if c not in exclude]


# ========== 1. 원핫(다중선택) 컬럼 분석 ==========
# -> 컬럼 개수가 3개 이상(응답 항목들)을 동시에 비교하는 성격이라 multivariate로 분류

def plot_top_n(df: pd.DataFrame, prefix: str, top_n: int = 15):
    cols = get_onehot_cols(df, prefix)
    counts = df[cols].sum().sort_values(ascending=False).head(top_n)
    counts.index = [c.replace(f"{prefix}__", "") for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"{prefix} (Top {top_n})")
    ax.set_xlabel("응답자 수")
    fig.tight_layout()
    return fig


def plot_have_vs_want(df: pd.DataFrame, base: str, top_n: int = 15):
    have_cols = get_onehot_cols(df, f"{base}HaveWorkedWith")
    want_cols = get_onehot_cols(df, f"{base}WantToWorkWith")

    have = df[have_cols].sum()
    have.index = [c.split("__", 1)[1] for c in have.index]
    want = df[want_cols].sum()
    want.index = [c.split("__", 1)[1] for c in want.index]

    top_items = have.sort_values(ascending=False).head(top_n).index
    plot_df = pd.DataFrame({"Have": have.reindex(top_items), "Want": want.reindex(top_items)})
    plot_df.index.name = base
    plot_df = plot_df.reset_index().melt(id_vars=base, var_name="type", value_name="count")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=plot_df, y=base, x="count", hue="type", ax=ax)
    ax.set_title(f"{base}: 현재 사용 vs 선호 (Top {top_n})")
    fig.tight_layout()
    return fig


def plot_tech_by_group(df: pd.DataFrame, group_col: str, tech_prefix: str,
                        top_group: int = 8, top_tech: int = 10):
    top_groups = df[group_col].value_counts().head(top_group).index
    tech_cols = get_onehot_cols(df, tech_prefix)

    top_tech_cols = df[tech_cols].sum().sort_values(ascending=False).head(top_tech).index

    sub = df[df[group_col].isin(top_groups)]
    cross = sub.groupby(group_col, observed=True)[top_tech_cols].mean()
    cross.columns = [c.split("__", 1)[1] for c in cross.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"{group_col}별 {tech_prefix} 사용 비율")
    fig.tight_layout()
    return fig


# ========== 2. 수치형 변수 ==========

def plot_salary_experience_dist(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["ConvertedCompYearly"].dropna(), bins=50, log_scale=True, ax=axes[0])
    axes[0].set_title("ConvertedCompYearly (log)")

    sns.histplot(df["YearsCode"].dropna(), bins=30, ax=axes[1])
    axes[1].set_title("YearsCode")

    sns.histplot(df["YearsCodePro"].dropna(), bins=30, ax=axes[2])
    axes[2].set_title("YearsCodePro")
    fig.tight_layout()
    return fig


def plot_exp_age_vs_salary(df: pd.DataFrame, cap_quantile: float = 0.99):
    cap = df["ConvertedCompYearly"].quantile(cap_quantile)
    sub = df[df["ConvertedCompYearly"] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(data=sub, x="YearsCodePro", y="ConvertedCompYearly",
                scatter_kws={"alpha": 0.1, "s": 10}, line_kws={"color": "red"}, ax=axes[0])
    axes[0].set_title("경력(YearsCodePro) vs 연봉")

    sns.regplot(data=sub, x="AgeNum", y="ConvertedCompYearly",
                scatter_kws={"alpha": 0.1, "s": 10}, line_kws={"color": "red"}, ax=axes[1])
    axes[1].set_title("나이(AgeNum) vs 연봉")
    fig.tight_layout()
    return fig


def plot_numeric_corr(df: pd.DataFrame):
    numeric_cols = get_numeric_cols(df)
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("수치형 변수 상관관계")
    fig.tight_layout()
    return fig


# ========== 3. 단일 범주형 ==========

ORDINAL_ORDER = {
    "Age": ["Under 18 years old", "18-24 years old", "25-34 years old", "35-44 years old",
            "45-54 years old", "55-64 years old", "65 years or older", "Prefer not to say"],
    "EdLevel": None,
    "OrgSize": None,
    "SOVisitFreq": [
        "Less than once per month or monthly", "A few times per month or weekly",
        "A few times per week", "Daily or almost daily", "Multiple times per day",
    ],
}


def plot_low_cardinality(df: pd.DataFrame, cols: list[str], wrap_width: int = 30):
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        order = ORDINAL_ORDER.get(col) or df[col].value_counts().index
        sns.countplot(data=df, y=col, order=order, ax=ax)
        wrapped = ["\n".join(textwrap.wrap(str(label), wrap_width)) for label in order]
        ax.set_yticks(range(len(wrapped)))
        ax.set_yticklabels(wrapped, fontsize=8)
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_high_cardinality(df: pd.DataFrame, col: str, top_n: int = 20):
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"{col} (Top {top_n})")
    fig.tight_layout()
    return fig


# ========== 4. 3개 이상 변수 동시 비교 ==========

def plot_catplot_multi(df: pd.DataFrame, tech_col: str, x: str, hue: str):
    sub = df.dropna(subset=[x, hue])

    g = sns.catplot(data=sub, x=x, y=tech_col, hue=hue, kind="bar",
                     estimator="mean", height=5, aspect=1.8)
    g.set_axis_labels(x, f"{tech_col} 사용 비율")
    g.figure.suptitle(f"{tech_col} 사용률: {x} x {hue}", y=1.02)
    return g


# 타겟 분포 (원본에는 없었으나 요청한 4개 카테고리 중 하나라 추가)
def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="target", ax=ax)
    total = len(df)
    for container in ax.containers:
        labels = [f"{int(bar.get_height()):,}\n({bar.get_height()/total*100:.1f}%)" for bar in container]
        ax.bar_label(container, labels=labels, padding=8)
    ax.set_title("Distribution of Target")
    fig.tight_layout()
    return fig


# ========== 실행 ==========

def section_safe(title: str, category: str, fn, *args, **kwargs):
    """필요한 컬럼이 없으면 에러 대신 경고만 남기고 건너뜀."""
    try:
        add_section(title, fn(*args, **kwargs), category)
    except KeyError as e:
        print(f"[건너뜀] '{title}' — 컬럼 없음: {e}")


def main():
    df = load_data()
    print("사용 가능한 컬럼:", df.columns.tolist())

    # 0. 타겟 분포
    section_safe("Target Distribution", "target", plot_target_distribution, df)

    # 1. 다중선택(원핫) -> multivariate
    section_safe("Top Languages", "multivariate", plot_top_n, df, "LanguageHaveWorkedWith")
    section_safe("Language: 현재 사용 vs 선호", "multivariate", plot_have_vs_want, df, "Language")
    section_safe(
        "MainBranch별 Language 사용 비율", "multivariate",
        plot_tech_by_group, df, group_col="MainBranch", tech_prefix="LanguageHaveWorkedWith",
    )

    # 2. 수치형
    section_safe("연봉/경력 분포", "numeric", plot_salary_experience_dist, df)
    section_safe("경력/나이 vs 연봉", "numeric", plot_exp_age_vs_salary, df)
    section_safe("수치형 변수 상관관계", "multivariate", plot_numeric_corr, df)  # 변수 여러 개 동시 비교라 multivariate

    # 3. 단일 범주형
    low_card_cols = [c for c in ["RemoteWork", "AIThreat", "TBranch"] if c in df.columns]
    if low_card_cols:
        section_safe("저카디널리티 범주형 분포", "categorical", plot_low_cardinality, df, low_card_cols)
    else:
        print("[건너뜀] 저카디널리티 범주형 — low_card_cols를 실제 컬럼명으로 교체하세요")

    section_safe("Country Top 20", "categorical", plot_high_cardinality, df, "Country")

    # 4. 3개 이상 변수 복합 시각화
    section_safe(
        "Python 사용률: RemoteWork x EdLevel", "multivariate",
        plot_catplot_multi, df, tech_col="LanguageHaveWorkedWith__Python", x="RemoteWork", hue="EdLevel",
    )

    save_report()
    print("완료")


if __name__ == "__main__":
    main()