"""전처리 보고서용: 수치형 원자료 분포 진단 (0단계 그림).

변환(4단계) '이전'의 수치형 분포가 어떤 문제를 갖고 있는지 보여준다 —
윈저라이징·log1p·중앙값 대체의 근거 자료다. 대표 8개 패널:

    1  ConvertedCompYearly        극단 우측 왜도 (+50.9, 최대 $16.2M)
    2  같은 변수의 log10          log 를 씌우면 쓸 만해지지만 $1~ 하위 쓰레기가
                                  좌측 꼬리로 남는다 -> 윈저라이징 병행 근거
    3  YearsCode                  sentinel 치환 흔적 (0.5 / 51 스파이크)
    4  YearsCodePro / 5 WorkExp   우측 왜도 (경력형 공통)
    6  JobSat                     좌왜 -> log 제외 근거
    7  JobSatPoints_6 / 8 _4      배점형: zero-inflated (0 비율 27% / 68%)

실행:
    python -m src.visualize_raw_dist
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import DATA_DIR, SILVER_PARQUET
except ModuleNotFoundError:
    # 이 파일을 직접 실행하면(IDE 의 F5 등) src 를 패키지로 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import DATA_DIR, SILVER_PARQUET

OUT = DATA_DIR / "0_1_raw_numeric_distributions.png"

SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

STAT_COLS = ["ConvertedCompYearly", "YearsCode", "YearsCodePro", "WorkExp", "JobSat",
             "JobSatPoints_1", "JobSatPoints_4", "JobSatPoints_5", "JobSatPoints_6",
             "JobSatPoints_7", "JobSatPoints_8", "JobSatPoints_9", "JobSatPoints_10",
             "JobSatPoints_11"]


def stat_line(s: pd.Series, n_total: int) -> str:
    v = s.dropna()
    return (f"결측 {100 * (1 - len(v) / n_total):.0f}% · 중앙값 {v.median():,.0f} · "
            f"왜도 {v.skew():+.1f}")


def print_stats(df: pd.DataFrame) -> None:
    """보고서 표에 옮겨 적을 수치를 콘솔에 찍는다."""
    n = len(df)
    print(f"{'변수':<22}{'결측%':>7}{'중앙값':>10}{'평균':>12}{'왜도':>8}"
          f"{'최소':>8}{'최대':>12}")
    for c in STAT_COLS:
        v = df[c].dropna()
        print(f"{c:<22}{100 * (1 - len(v) / n):>7.1f}{v.median():>10,.1f}"
              f"{v.mean():>12,.1f}{v.skew():>8.2f}{v.min():>8,.1f}{v.max():>12,.0f}")


def main() -> None:
    df = pd.read_parquet(SILVER_PARQUET, columns=STAT_COLS)
    n = len(df)
    print_stats(df)

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "font.sans-serif": ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                            "Helvetica", "Arial"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), dpi=150)
    ax = axes.ravel()

    def draw(a, series, bins, title, note, xlabel=""):
        a.hist(series.dropna(), bins=bins, color=BLUE)
        a.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=10)
        a.text(0, 1.005, note, transform=a.transAxes, fontsize=7.8, color=MUTED,
               va="bottom")
        a.set_yticks([])
        a.set_xlabel(xlabel, fontsize=8)
        a.tick_params(labelsize=8)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)

    comp = df["ConvertedCompYearly"]
    p995 = comp.quantile(0.995)
    draw(ax[0], comp.clip(upper=p995), np.linspace(0, p995, 40),
         "① 연 보수 USD — 극단 우측 왜도",
         f"{stat_line(comp, n)} · 최대 \\${comp.max() / 1e6:.1f}M (표시범위 밖)",
         "표시는 상위 0.5% 절단")
    draw(ax[1], np.log10(comp.dropna()), 40,
         "② 같은 변수 log10 — 변환 근거",
         "종형에 가까워지나 \\$1~\\$200대 쓰레기 응답이 좌측 꼬리로 남음 → 윈저라이징 병행",
         "log10(USD)")
    yc = df["YearsCode"]
    draw(ax[2], yc, np.arange(0, 53, 1),
         "③ 코딩 연수 — sentinel 흔적",
         f"{stat_line(yc, n)} · 0.5=“Less than 1 year”(569건), "
         "51=“More than 50 years”(254건)", "년")
    draw(ax[3], df["YearsCodePro"], np.arange(0, 53, 1),
         "④ 직업 코딩 연수 — 우측 왜도",
         stat_line(df["YearsCodePro"], n), "년")
    draw(ax[4], df["WorkExp"], np.arange(0, 51, 1),
         "⑤ 근무 경력 — 우측 왜도",
         stat_line(df["WorkExp"], n), "년")
    draw(ax[5], df["JobSat"], np.arange(-0.5, 11.5, 1),
         "⑥ 직무 만족(0~10) — 좌왜",
         stat_line(df["JobSat"], n) + " · log 제외 대상", "점")
    js6, js4 = df["JobSatPoints_6"], df["JobSatPoints_4"]

    def zero_pct(s: pd.Series) -> str:
        return f"0점 비율 {100 * (s.dropna() == 0).mean():.0f}%"

    draw(ax[6], js6, np.arange(0, 102, 2.5),
         "⑦ 배점: 코드 품질 개선 — zero-inflated",
         f"{stat_line(js6, n)} · {zero_pct(js6)}", "배점(100점 분배)")
    draw(ax[7], js4, np.arange(0, 102, 2.5),
         "⑧ 배점: 오픈소스 기여 — 극단 zero-inflated",
         f"{stat_line(js4, n)} · {zero_pct(js4)}", "배점(100점 분배)")

    fig.suptitle("수치형 원자료 분포 진단 — 변환(4단계) 이전 · n=65,437",
                 x=0.008, y=0.995, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.008, 0.955,
             "JobSatPoints 11개 문항 중 대표 2개만 표시 "
             "(나머지도 왜도 +1.4~+3.4, 0점 비율 27~68%로 동일 양상)",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
