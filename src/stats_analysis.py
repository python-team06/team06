"""
통합 분석 스크립트
[1] 기술통계 (수치형: scipy.stats.describe / 범주형: 최빈값+빈도)
[2] 시각화 (상관계수 히트맵 3종: JobSatPoints만 / JobSatPoints 제외 / 전체 수치형)
[3] 통계분석 (AISelect(target) vs 범주형: 카이제곱, vs 수치형: ANOVA)
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib
import platform
import seaborn as sns

import sys
from pathlib import Path

try:
    from src.config import (
        SILVER_PARQUET
    )
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        SILVER_PARQUET
    )


# ---------------------------------------------------------
# 한글 폰트 설정
# ---------------------------------------------------------
if platform.system() == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
else:
    matplotlib.rc("font", family="NanumGothic")
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================
# 0. 공통 설정 & 데이터 불러오기 & 전처리 (한 번만 수행)
# =========================================================
FILE_PATH = SILVER_PARQUET
ENCODING = "utf-8"
ALPHA = 0.05
TARGET = "AISelect"

df = pd.read_parquet(FILE_PATH)


def clean_years(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "Less than 1 year":
        return 0.5
    if x == "More than 50 years":
        return 51
    try:
        return float(x)
    except ValueError:
        return np.nan


for col in ["YearsCode", "YearsCodePro"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_years)

if "CompTotal" in df.columns:
    df["CompTotal"] = pd.to_numeric(df["CompTotal"], errors="coerce")
    lower, upper = df["CompTotal"].quantile([0.01, 0.99])
    df["CompTotal_trimmed"] = df["CompTotal"].where(
        (df["CompTotal"] >= lower) & (df["CompTotal"] <= upper)
    )

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"수치형 변수 ({len(numeric_cols)}개): {numeric_cols}")
print(f"범주형 변수 ({len(categorical_cols)}개): {categorical_cols}\n")


# #########################################################
# [1] 기술통계
# #########################################################
print("#" * 70)
print("# [1] 기술통계")
print("#" * 70)

# ---------------------------------------------------------
# [1-1] 수치형 변수 기술통계 (scipy.stats.describe)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("[1-1] 수치형 변수 기술통계 (scipy.stats.describe)")
print("=" * 70)

numeric_desc_results = []
for col in numeric_cols:
    data = df[col].dropna().values.astype(np.float64)
    if len(data) < 2:
        continue

    # 극단적 이상치로 인한 overflow 방지 (상하위 0.5% 클리핑 후 skew/kurtosis 계산)
    lo, hi = np.percentile(data, [0.5, 99.5])
    data_clipped = np.clip(data, lo, hi)
    desc = stats.describe(data_clipped)

    numeric_desc_results.append({
        "변수": col,
        "n(결측제외)": desc.nobs,
        "결측치": df[col].isna().sum(),
        "최소값(원본)": round(data.min(), 2),
        "최대값(원본)": round(data.max(), 2),
        "평균": round(desc.mean, 2),
        "분산": round(desc.variance, 2),
        "표준편차": round(np.sqrt(desc.variance), 2),
        "중앙값": round(np.median(data), 2),
        "왜도(skewness)": round(desc.skewness, 3),
        "첨도(kurtosis)": round(desc.kurtosis, 3),
    })

numeric_desc_df = pd.DataFrame(numeric_desc_results)
print(numeric_desc_df.to_string(index=False))
numeric_desc_df.to_csv("1_descriptive_numeric.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 1_descriptive_numeric.csv")

# ---------------------------------------------------------
# [1-2] 범주형 변수 기술통계 (최빈값 + 빈도)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("[1-2] 범주형 변수 기술통계 (최빈값 + 빈도표)")
print("=" * 70)

categorical_desc_results = []
for col in categorical_cols:
    data = df[col].dropna()
    if len(data) == 0:
        continue

    # scipy.stats.mode는 SciPy 1.11+부터 문자열 미지원 -> np.unique로 최빈값 계산
    values, counts = np.unique(data.astype(str), return_counts=True)
    max_idx = np.argmax(counts)
    mode_value = values[max_idx]
    mode_count = counts[max_idx]

    n_valid = len(data)
    categorical_desc_results.append({
        "변수": col,
        "n(결측제외)": n_valid,
        "결측치": df[col].isna().sum(),
        "고유값개수": data.nunique(),
        "최빈값(mode)": mode_value,
        "최빈값빈도": mode_count,
        "최빈값비율(%)": round(mode_count / n_valid * 100, 1),
    })

categorical_desc_df = pd.DataFrame(categorical_desc_results)
print(categorical_desc_df.to_string(index=False))
categorical_desc_df.to_csv("1_descriptive_categorical.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 1_descriptive_categorical.csv")


# #########################################################
# [2] 시각화 (상관계수 히트맵 3종)
# #########################################################
print("\n\n" + "#" * 70)
print("# [2] 시각화 - 상관계수 히트맵")
print("#" * 70)

jobsat_cols = [c for c in numeric_cols if c.startswith("JobSatPoints_")] + \
              (["JobSat"] if "JobSat" in numeric_cols else [])
non_jobsat_cols = [c for c in numeric_cols if c not in jobsat_cols]


def analyze_correlation(cols, title, filename_prefix):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    corr_matrix = df[cols].corr(method="pearson")
    print(corr_matrix.round(3))
    corr_matrix.to_csv(f"2_{filename_prefix}_matrix.csv", encoding="utf-8-sig")

    plt.figure(figsize=(max(6, len(cols) * 0.9), max(5, len(cols) * 0.7)))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="white")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"2_{filename_prefix}_heatmap.png", dpi=150)
    plt.show()
    print(f"저장 완료: 2_{filename_prefix}_matrix.csv / 2_{filename_prefix}_heatmap.png")

    results = []
    for col1, col2 in combinations(cols, 2):
        valid = df[[col1, col2]].dropna()
        if len(valid) < 3:
            continue
        r, p = stats.pearsonr(valid[col1], valid[col2])
        strength = "강함" if abs(r) >= 0.5 else "중간" if abs(r) >= 0.3 else "약함"
        results.append({
            "변수1": col1, "변수2": col2, "n": len(valid),
            "r": round(r, 3), "p_value": p, "강도": strength,
            "유의함(p<0.05)": p < ALPHA
        })
    result_df = pd.DataFrame(results).sort_values("r", key=abs, ascending=False)
    result_df.to_csv(f"2_{filename_prefix}_with_pvalue.csv", index=False, encoding="utf-8-sig")
    print(f"\n상위 상관관계:\n{result_df.head(10).to_string(index=False)}")

    return corr_matrix, result_df


if jobsat_cols:
    analyze_correlation(jobsat_cols,
                         "[2-1] JobSatPoints & JobSat 상관계수 히트맵",
                         "jobsatpoints")

if non_jobsat_cols:
    analyze_correlation(non_jobsat_cols,
                         "[2-2] 수치형 변수 상관계수 히트맵 (JobSatPoints 제외)",
                         "no_jobsatpoints")

analyze_correlation(numeric_cols,
                     "[2-3] 전체 수치형 변수 상관계수 히트맵",
                     "all_numeric")


# #########################################################
# [3] 통계분석 (AISelect(target) vs 범주형/수치형)
# #########################################################
print("\n\n" + "#" * 70)
print("# [3] 통계분석 - AISelect(target)와의 관계")
print("#" * 70)

# 예측변수로 안전한(조건부 스킵 아닌) 범주형/수치형 변수
STAT_CATEGORICAL_VARS = [
    "Age", "EdLevel", "RemoteWork", "OrgSize", "MainBranch",
    "SOAccount", "Industry", "DevType"
]
STAT_NUMERIC_VARS = ["YearsCode", "YearsCodePro", "WorkExp", "CompTotal_trimmed"]

print(f"\nAISelect 분포:\n{df[TARGET].value_counts()}\n")

# ---------------------------------------------------------
# [3-1] 범주형 변수 vs AISelect : 카이제곱 검정 + Cramér's V
# ---------------------------------------------------------
print("=" * 70)
print("[3-1] 범주형 변수 vs AISelect : 카이제곱 검정")
print("=" * 70)

chi_results = []
for col in STAT_CATEGORICAL_VARS:
    if col not in df.columns:
        continue
    sub = df[[col, TARGET]].dropna()
    ct = pd.crosstab(sub[col], sub[TARGET])

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    strength = "강함" if v >= 0.3 else "중간" if v >= 0.1 else "약함"

    chi_results.append({
        "변수": col, "n": n, "chi2": round(chi2, 2), "dof": dof,
        "p_value": p, "CramersV": round(v, 3), "강도": strength,
        "유의함(p<0.05)": p < ALPHA
    })

chi_df = pd.DataFrame(chi_results).sort_values("CramersV", ascending=False)
print(chi_df.to_string(index=False))
chi_df.to_csv("3_aiselect_chisquare_results.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 3_aiselect_chisquare_results.csv")


# ---------------------------------------------------------
# [3-2] 수치형 변수 vs AISelect : 일원분산분석(ANOVA) + eta-squared
# ---------------------------------------------------------
def eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else np.nan


print("\n" + "=" * 70)
print("[3-2] 수치형 변수 vs AISelect : 일원분산분석(ANOVA)")
print("=" * 70)

anova_results = []
for col in STAT_NUMERIC_VARS:
    if col not in df.columns:
        continue
    sub = df[[col, TARGET]].dropna()

    groups_dict = {k: v[col].values for k, v in sub.groupby(TARGET, observed=True)}
    groups = list(groups_dict.values())
    group_names = list(groups_dict.keys())
    if len(groups) < 2:
        continue

    f_stat, p_value = stats.f_oneway(*groups)
    eta2 = eta_squared(groups)
    strength = "큼" if eta2 >= 0.14 else "중간" if eta2 >= 0.06 else "작음"

    print(f"\n--- {col} ---")
    for name, g in zip(group_names, groups):
        print(f"  {name}: n={len(g)}, 평균={g.mean():.2f}, 표준편차={g.std():.2f}")
    print(f"  F={f_stat:.3f}, p={p_value:.4f} -> "
          f"{'통계적으로 유의함 ✅' if p_value < ALPHA else '유의하지 않음'}")
    print(f"  eta-squared={eta2:.4f} (효과크기: {strength})")

    anova_results.append({
        "변수": col, "F_statistic": round(f_stat, 3), "p_value": p_value,
        "eta_squared": round(eta2, 4), "효과크기": strength,
        "유의함(p<0.05)": p_value < ALPHA
    })

anova_df = pd.DataFrame(anova_results).sort_values("eta_squared", ascending=False)
print("\n--- 요약 ---")
print(anova_df.to_string(index=False))
anova_df.to_csv("3_aiselect_anova_results.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 3_aiselect_anova_results.csv")

# ---------------------------------------------------------
# [3-3] 통계분석 결과 시각화
# ---------------------------------------------------------
if len(chi_df) > 0:
    top_cat = chi_df.iloc[0]["변수"]
    sub = df[[top_cat, TARGET]].dropna()
    ct_pct = pd.crosstab(sub[top_cat], sub[TARGET], normalize="index") * 100
    ct_pct.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab10")
    plt.title(f"{top_cat} 별 AISelect 비율 (Cramér's V 최상위)")
    plt.ylabel("비율 (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="AISelect", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("3_aiselect_top_categorical.png", dpi=150)
    print("\n그래프 저장: 3_aiselect_top_categorical.png")

valid_numeric_vars = [c for c in STAT_NUMERIC_VARS if c in df.columns]
if valid_numeric_vars:
    fig, axes = plt.subplots(1, len(valid_numeric_vars),
                              figsize=(5 * len(valid_numeric_vars), 5))
    if len(valid_numeric_vars) == 1:
        axes = [axes]
    for ax, col in zip(axes, valid_numeric_vars):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("3_aiselect_numeric_boxplots.png", dpi=150)
    print("그래프 저장: 3_aiselect_numeric_boxplots.png")

print("""
--- 해석 가이드 ---
Cramér's V  : 0.1 미만=거의 무의미, 0.1~0.3=약함/중간, 0.3+=강함
eta-squared : 0.01=작음, 0.06=중간, 0.14+=큼
표본이 커서 p-value는 거의 항상 유의하게 나오므로
반드시 CramersV / eta_squared로 실질적 의미를 판단하세요.
""")

print("\n" + "#" * 70)
print("# 전체 분석 완료")
print("#" * 70)