import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.config import (
        GOLD_LONG_PARQUET,
        GOLD_WIDE_PARQUET,
        SILVER_PARQUET,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        GOLD_LONG_PARQUET,
        GOLD_WIDE_PARQUET,
        SILVER_PARQUET,
    )


# 전처리 완료된 데이터셋 로드
def load_data():
    silver = pd.read_parquet(SILVER_PARQUET)
    gold_wide = pd.read_parquet(GOLD_WIDE_PARQUET)
    gold_long = pd.read_parquet(GOLD_LONG_PARQUET)
    return silver, gold_wide, gold_long


# ------------------------------------------------------------------
# 1. [분포 비교] 국가별 AI 사용률 (Choropleth Map)
# ------------------------------------------------------------------
def plot_country_ai_usage(silver: pd.DataFrame):
    """
    목적
    ----
    국가(Country)가 AI 사용 여부(AISelect)에 영향을 주는지 탐색한다.

    Plotly 장점
    ----------
    - Hover로 국가별 AI 사용률과 응답자 수 확인
    - Zoom/Pan 가능
    - 정적 지도보다 국가 비교가 쉬움
    """

    # AISelect를 이진 타깃으로 변환 (Yes = 1, 나머지 = 0)
    df = silver[["Country", "AISelect"]].dropna().copy()
    df["target"] = (df["AISelect"] == "Yes").astype(int)

    # 국가별 AI 사용률, 응답자 수
    country_stats = (
        df.groupby("Country")
        .agg(
            ai_use_rate=("target", lambda x: x.mean() * 100),
            n=("target", "count"),
        )
        .reset_index()
    )

    # 응답자가 너무 적은 국가는 사용률이 왜곡될 수 있으므로 제외
    country_stats = country_stats[country_stats["n"] >= 30]

    # Choropleth(세계 지도) 생성, Hover에서 국가별 사용률과 응답자 수 확인 가능
    fig = px.choropleth(
        country_stats,
        locations="Country",
        locationmode="country names",
        color="ai_use_rate",
        hover_name="Country",
        hover_data={"ai_use_rate": ":.1f", "n": True},
        color_continuous_scale="Blues",
        labels={"ai_use_rate": "AI 사용률(%)", "n": "응답자 수"},
        title="<b>1. 국가별 AI 사용률</b>",
    )

    fig.update_layout(template="plotly_white", geo=dict(showframe=False))

    return fig


# ------------------------------------------------------------------
# 2. [그룹 비교] 개발 직무 및 경력 구간 별 AI 사용률
# ------------------------------------------------------------------
def plot_devtype_ai_usage(silver: pd.DataFrame):
    """
    목적
    ----
    개발 직무 및 경력 구간에 따라 AI 사용률이 달라지는지 확인한다.

    Plotly 장점
    ----------
    - 상단 드롭다운 메뉴로 전체 / 주니어 / 미드 / 시니어 경력별 직무 AI 사용률 즉시 전환
    - Hover로 정확한 사용률(%)과 표본 수(N) 확인
    """
    # 경력(YearsCodePro) 컬럼 추가 추출
    df = silver[["DevType", "AISelect", "YearsCodePro"]].dropna().copy()
    df["target"] = (df["AISelect"] == "Yes").astype(int)

    # DevType 다중 응답 분리
    df["DevType"] = df["DevType"].str.split(";")
    df = df.explode("DevType")
    df["DevType"] = df["DevType"].str.strip()

    # 상위 10개 주요 직무만 추출
    top_devs = df["DevType"].value_counts().head(10).index
    df = df[df["DevType"].isin(top_devs)].copy()

    # 경력 구간 생성
    exp_bins = [-1, 3, 10, 100]
    exp_labels = ["주니어 (0-3년)", "미드레벨 (4-10년)", "시니어 (11년 이상)"]
    df["ExpGroup"] = pd.cut(df["YearsCodePro"], bins=exp_bins, labels=exp_labels)

    dropdown_options = ["전체 경력"] + exp_labels

    # 드롭다운 옵션별 집계 데이터 사전 생성
    data_by_exp = {}
    for opt in dropdown_options:
        sub_df = df if opt == "전체 경력" else df[df["ExpGroup"] == opt]

        res = (
            sub_df.groupby("DevType")
            .agg(
                ai_rate=("target", lambda x: x.mean() * 100 if len(x) > 0 else 0),
                n=("target", "count"),
            )
            .reindex(top_devs)  # Y축 직무 순서 고정
            .reset_index()
            .sort_values("ai_rate", ascending=True)  # AI 사용률 순으로 정렬
        )
        data_by_exp[opt] = res

    # Plotly Figure 생성
    fig = go.Figure()

    # 최초 화면은 전체 경력 기준으로 표시
    default_opt = "전체 경력"
    default_df = data_by_exp[default_opt]

    fig.add_trace(
        go.Bar(
            x=default_df["ai_rate"],
            y=default_df["DevType"],
            orientation="h",
            text=default_df["ai_rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>AI 사용률: %{x:.1f}%<br>응답자 수: %{customdata:,}명<extra></extra>",
            customdata=default_df["n"],
            marker_color="#1f77b4",
        )
    )

    # 드롭다운 버튼 목록 생성
    buttons = []
    for opt in dropdown_options:
        opt_df = data_by_exp[opt]
        buttons.append(
            dict(
                method="update",
                label=opt,
                args=[
                    {
                        "x": [opt_df["ai_rate"]],
                        "y": [opt_df["DevType"]],
                        "text": [opt_df["ai_rate"].apply(lambda x: f"{x:.1f}%")],
                        "customdata": [opt_df["n"]],
                    },
                    {"title": f"<b>2. 개발 직무별 AI 사용률 ({opt})</b>"},
                ],
            )
        )

    # 드롭다운 메뉴 및 레이아웃 설정
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        title=f"<b>2. 개발 직무별 AI 사용률 ({default_opt})</b>",
        xaxis=dict(title="AI 사용률 (%)", range=[0, 105]),
        yaxis=dict(title="개발 직무"),
        template="plotly_white",
        height=600,
    )

    return fig


# ------------------------------------------------------------------
# 3. [그룹 비교] 개발 경력 구간별 AI 사용률
# ------------------------------------------------------------------
def plot_experience_ai_usage(silver: pd.DataFrame):
    """
    목적
    ----
    개발 경력이 증가할수록 AI 사용률이 달라지는지 확인한다.

    Plotly 장점
    ----------
    - Hover로 AI 사용률과 표본 수 확인
    - 추세(Line)를 직관적으로 확인
    """

    # 사용할 컬럼 선택
    df = silver[["YearsCodePro", "AISelect"]].dropna().copy()
    df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"], errors="coerce")
    df = df.dropna()
    df["target"] = (df["AISelect"] == "Yes").astype(int)

    # 경력을 구간으로
    bins = [0, 2, 5, 10, 15, 20, 30, 60]
    labels = ["0~2년", "3~5년", "6~10년", "11~15년", "16~20년", "21~30년", "30년+"]

    df["CareerGroup"] = pd.cut(
        df["YearsCodePro"], bins=bins, labels=labels, include_lowest=True
    )

    # 경력 구 간별 AI 사용률과 응답자 수 계산
    result = (
        df.groupby("CareerGroup", observed=True)
        .agg(ai_rate=("target", lambda x: x.mean() * 100), n=("target", "count"))
        .reset_index()
    )

    # 그래프
    fig = px.line(
        result,
        x="CareerGroup",
        y="ai_rate",
        markers=True,
        hover_data={"ai_rate": ":.1f", "n": True},
        labels={"CareerGroup": "개발 경력", "ai_rate": "AI 사용률(%)"},
        title="<b>3. 개발 경력 구간별 AI 사용률</b>",
    )

    fig.update_layout(template="plotly_white")

    return fig


def main():

    silver, gold_wide, gold_long = load_data()

    fig1 = plot_country_ai_usage(silver)
    fig1.show()
    fig1.write_html("data/plot1_country_ai_usage.html")
    print("저장 완료 : data/plot1_country_ai_usage.html")

    fig2 = plot_devtype_ai_usage(silver)
    fig2.show()
    fig2.write_html("data/plot2_devtype_ai_usage.html")
    print("저장 완료 : data/plot2_devtype_ai_usage.html")

    fig3 = plot_experience_ai_usage(silver)
    fig3.show()
    fig3.write_html("data/plot3_experience_ai_usage.html")
    print("저장 완료 : data/plot3_experience_ai_usage.html")


if __name__ == "__main__":
    main()
