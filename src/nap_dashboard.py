from __future__ import annotations

import logging
import sys
import textwrap
import types
from html import escape
from pathlib import Path

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
if "xarray" not in sys.modules:
    xarray_stub = types.ModuleType("xarray")

    class _XArrayDataArray:
        pass

    xarray_stub.DataArray = _XArrayDataArray
    sys.modules["xarray"] = xarray_stub

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .nap_adapter import (
        DEFAULT_NAP_WORKBOOK,
        default_cache_path,
        default_catalog_path,
        default_explanations_path,
        load_nap_timeseries,
        workbook_status,
    )
    from .nap_analytics import (
        MARKET_GROUPS,
        available_curve_groups,
        build_curve_history,
        build_forward_curve,
        build_market_snapshot,
        build_spread_series,
        catalog_with_labels,
        display_lookup,
        forward_curve_spreads,
        long_to_wide,
        relationship_package,
    )
    from .nap_formula_registry import evaluate_registry_formula, load_formula_registry
    from .risk_engine import (
        drawdown_series,
        ewma_volatility,
        log_returns,
        max_drawdown,
        realized_volatility,
        risk_contribution,
        var_es_over_window,
        zscore_of_value,
    )
    from .seasonal_engine import calendar_heatmap_frame, monthly_box_frame, remove_feb29, seasonal_matrix, seasonal_percentile_band
except ImportError:
    from src.nap_adapter import (
        DEFAULT_NAP_WORKBOOK,
        default_cache_path,
        default_catalog_path,
        default_explanations_path,
        load_nap_timeseries,
        workbook_status,
    )
    from src.nap_analytics import (
        MARKET_GROUPS,
        available_curve_groups,
        build_curve_history,
        build_forward_curve,
        build_market_snapshot,
        build_spread_series,
        catalog_with_labels,
        display_lookup,
        forward_curve_spreads,
        long_to_wide,
        relationship_package,
    )
    from src.nap_formula_registry import evaluate_registry_formula, load_formula_registry
    from src.risk_engine import (
        drawdown_series,
        ewma_volatility,
        log_returns,
        max_drawdown,
        realized_volatility,
        risk_contribution,
        var_es_over_window,
        zscore_of_value,
    )
    from src.seasonal_engine import calendar_heatmap_frame, monthly_box_frame, remove_feb29, seasonal_matrix, seasonal_percentile_band


if getattr(sys, "frozen", False):
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
else:
    BASE_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

LIGHT_TEMPLATE = "plotly_white"
DARK_TEMPLATE = "plotly_dark"
ACCENT = "#2f7d8c"
AMBER = "#d29b47"
NEGATIVE = "#bf5b5b"
POSITIVE = "#4b9b72"


PAGE_OPTIONS = {
    "行情地图": "market",
    "序列详情": "detail",
    "季节性": "seasonality",
    "关系实验室": "relationship",
    "远期曲线": "curve",
    "波动 / 风险": "risk",
    "运费 / 套利": "freight",
    "价格词典": "glossary",
}

SECTOR_CN = {
    "Crude": "原油",
    "Gasoline": "汽油",
    "Naphtha": "石脑油",
    "Diesel": "柴油",
    "Jet/Heating Oil": "航煤 / 取暖油",
    "Propane/LPG": "丙烷 / LPG",
    "Fuel Oil": "燃料油",
    "Freight": "运费",
    "Cracks": "裂解价差",
    "Margins": "利润",
    "LNG": "LNG",
}

STRUCTURE_CN = {
    "backwardation": "Backwardation",
    "contango": "Contango",
    "flat": "平水",
    "flat/spot": "现货 / 无曲线",
}

SNAPSHOT_CN = {
    "Current": "当前",
    "7D ago": "1周前",
    "30D ago": "1个月前",
    "90D ago": "3个月前",
}


def _plot_template() -> str:
    return DARK_TEMPLATE if st.session_state.get("nap_theme", "浅色") in {"深色", "Dark"} else LIGHT_TEMPLATE


def _inject_theme(theme: str) -> None:
    dark = theme in {"深色", "Dark"}
    bg = "#0f1417" if dark else "#f6f3ee"
    panel = "#171f24" if dark else "#ffffff"
    panel_soft = "#11191e" if dark else "#f1ece4"
    text = "#e9eef0" if dark else "#17242b"
    muted = "#93a4ad" if dark else "#66757d"
    line = "rgba(148, 163, 172, 0.22)" if dark else "rgba(44, 62, 70, 0.14)"
    sidebar = "#10171b" if dark else "#eee8df"
    st.markdown(
        f"""
        <style>
        :root {{
            --nap-bg: {bg};
            --nap-panel: {panel};
            --nap-panel-soft: {panel_soft};
            --nap-text: {text};
            --nap-muted: {muted};
            --nap-line: {line};
            --nap-accent: {ACCENT};
            --nap-amber: {AMBER};
            --nap-pos: {POSITIVE};
            --nap-neg: {NEGATIVE};
        }}
        .stApp {{
            background: var(--nap-bg);
            color: var(--nap-text);
            font-family: "Aptos", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }}
        .block-container {{
            padding-top: 0.9rem;
            padding-bottom: 2.5rem;
            max-width: 1640px;
        }}
        [data-testid="stSidebar"] {{
            background: var(--nap-sidebar, {sidebar});
            border-right: 1px solid var(--nap-line);
        }}
        [data-testid="stSidebar"] * {{ color: var(--nap-text); }}
        [data-testid="stSidebar"] [data-baseweb="radio"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--nap-muted);
        }}
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
        }}
        .nap-topbar {{
            display: grid;
            grid-template-columns: minmax(18rem, 1.6fr) repeat(4, minmax(7rem, 0.5fr));
            gap: 0.75rem;
            align-items: stretch;
            margin-bottom: 0.9rem;
        }}
        .nap-brand {{
            padding: 0.9rem 1rem;
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
        }}
        .nap-brand-kicker {{
            color: var(--nap-accent);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 750;
        }}
        .nap-brand-title {{
            margin-top: 0.25rem;
            color: var(--nap-text);
            font-size: 1.35rem;
            font-weight: 780;
            line-height: 1.1;
        }}
        .nap-status {{
            padding: 0.78rem 0.85rem;
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
        }}
        .nap-status-label {{
            color: var(--nap-muted);
            font-size: 0.72rem;
            line-height: 1.1;
        }}
        .nap-status-value {{
            color: var(--nap-text);
            font-size: 1rem;
            font-weight: 760;
            margin-top: 0.25rem;
            word-break: break-word;
        }}
        .nap-section-title {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.9rem 0 0.6rem 0;
        }}
        .nap-section-title span:first-child {{
            color: var(--nap-text);
            font-size: 1.05rem;
            font-weight: 780;
        }}
        .nap-section-title span:last-child {{
            color: var(--nap-muted);
            font-size: 0.82rem;
        }}
        .nap-card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.75rem;
        }}
        .nap-card {{
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            padding: 0.72rem 0.78rem;
            min-height: 142px;
        }}
        .nap-card-name {{
            color: var(--nap-text);
            font-size: 0.92rem;
            font-weight: 760;
            line-height: 1.22;
            min-height: 2.2rem;
        }}
        .nap-card-meta {{
            color: var(--nap-muted);
            font-size: 0.72rem;
            margin-top: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .nap-card-latest {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.5rem;
            margin-top: 0.58rem;
        }}
        .nap-price {{
            color: var(--nap-text);
            font-size: 1.28rem;
            font-weight: 820;
        }}
        .nap-unit {{
            color: var(--nap-muted);
            font-size: 0.72rem;
        }}
        .nap-microgrid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.38rem;
            margin-top: 0.58rem;
        }}
        .nap-micro {{
            border-top: 1px solid var(--nap-line);
            padding-top: 0.32rem;
            min-width: 0;
        }}
        .nap-micro label {{
            color: var(--nap-muted);
            font-size: 0.62rem;
            display: block;
        }}
        .nap-micro strong {{
            color: var(--nap-text);
            font-size: 0.75rem;
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .nap-pill {{
            display: inline-flex;
            border: 1px solid var(--nap-line);
            border-radius: 999px;
            padding: 0.15rem 0.45rem;
            color: var(--nap-muted);
            font-size: 0.66rem;
            margin-top: 0.45rem;
        }}
        .nap-positive {{ color: var(--nap-pos) !important; }}
        .nap-negative {{ color: var(--nap-neg) !important; }}
        .nap-note {{
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            color: var(--nap-muted);
            line-height: 1.55;
            font-size: 0.9rem;
        }}
        .nap-note strong {{ color: var(--nap-text); }}
        div[data-testid="stPlotlyChart"] {{
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            background: var(--nap-panel);
            padding: 0.25rem 0.3rem;
        }}
        .stDataFrame {{
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            overflow: hidden;
        }}
        div[data-testid="stMetric"] {{
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }}
        @media (max-width: 980px) {{
            .nap-topbar {{ grid-template-columns: 1fr 1fr; }}
            .nap-brand {{ grid-column: 1 / -1; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt(value: object, digits: int = 2, pct: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pct:
        return f"{number * 100:.{digits}f}%"
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _signed_class(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "nap-positive" if float(value) >= 0 else "nap-negative"


def _download_csv(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=True).encode("utf-8-sig"), file_name=filename, mime="text/csv", key=key)


def _safe_dataframe(df: pd.DataFrame, *, hide_index: bool = False, use_container_width: bool = True) -> None:
    try:
        st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)
    except Exception as exc:
        if "pyarrow" not in repr(exc).lower() and "multiarray" not in repr(exc).lower():
            raise
        view = df.reset_index(drop=True) if hide_index else df
        st.markdown(view.to_html(index=not hide_index, escape=True), unsafe_allow_html=True)


def _clean_option(value: object, fallback: str = "未分类") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _sector_label(value: object) -> str:
    text = _clean_option(value)
    return SECTOR_CN.get(text, text)


def _sector_order(options: list[str]) -> list[str]:
    order = {sector: idx for idx, sector in enumerate(MARKET_GROUPS)}
    return sorted(options, key=lambda item: (order.get(item, 999), _sector_label(item)))


def _option_values(frame: pd.DataFrame, column: str, *, include_all: bool = True) -> list[str]:
    if column not in frame.columns or frame.empty:
        return ["全部"] if include_all else []
    values = sorted({_clean_option(value) for value in frame[column].dropna() if _clean_option(value) != "未分类"})
    return (["全部"] + values) if include_all else values


def _filter_by_option(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if value == "全部" or column not in frame.columns:
        return frame
    return frame[frame[column].fillna("").astype(str).str.strip() == value]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _lookup_explanation(explanations: dict, series_id: str, meta: pd.Series | dict) -> dict:
    series_map = explanations.get("series", {})
    if series_id in series_map:
        return series_map[series_id]
    product = str(meta.get("product", ""))
    sector = str(meta.get("sector", ""))
    defaults = explanations.get("defaults", {})
    return defaults.get(product) or defaults.get(sector) or defaults.get("generic", {})


@st.cache_data(show_spinner=False)
def _cached_load(workbook_path: str, cache_path: str, catalog_path: str, refresh_token: int) -> pd.DataFrame:
    return load_nap_timeseries(workbook_path, cache_path=cache_path, catalog_path=catalog_path, refresh=refresh_token > 0)


def _sidebar() -> dict[str, object]:
    st.sidebar.markdown("### Reuters NAP 交易看板")
    page = st.sidebar.radio(
        "导航",
        list(PAGE_OPTIONS),
        label_visibility="collapsed",
    )
    theme = st.sidebar.radio("主题", ["浅色", "深色"], horizontal=True)
    st.session_state["nap_theme"] = theme
    workbook_path = st.sidebar.text_input("Nap.xlsx 路径", value=str(DEFAULT_NAP_WORKBOOK))
    cache_path = st.sidebar.text_input("缓存路径", value=str(default_cache_path()))
    catalog_path = st.sidebar.text_input("Catalog 路径", value=str(default_catalog_path()))
    if "nap_refresh_token" not in st.session_state:
        st.session_state["nap_refresh_token"] = 0
    if st.sidebar.button("刷新数据缓存", use_container_width=True):
        st.session_state["nap_refresh_token"] += 1
        _cached_load.clear()
    st.sidebar.caption("优先缓存为 parquet；本机 parquet 引擎不可用时自动使用本地 pickle 兜底。")
    return {
        "page": PAGE_OPTIONS[page],
        "theme": theme,
        "workbook_path": workbook_path,
        "cache_path": cache_path,
        "catalog_path": catalog_path,
        "refresh_token": st.session_state["nap_refresh_token"],
    }


def _render_topbar(df: pd.DataFrame, workbook_path: str) -> None:
    status = workbook_status(df, workbook_path)
    latest = status["latest_trade_date"]
    mtime = status["workbook_mtime"]
    mtime_text = f"{mtime:%Y-%m-%d %H:%M}" if pd.notna(mtime) else "-"
    latest_text = f"{latest:%Y-%m-%d}" if pd.notna(latest) else "-"
    st.markdown(
        f"""
        <div class="nap-topbar">
          <div class="nap-brand">
            <div class="nap-brand-kicker">REUTERS EXCEL RESEARCH TERMINAL</div>
            <div class="nap-brand-title">NAP 多品种交易研究看板</div>
          </div>
          <div class="nap-status"><div class="nap-status-label">数据更新时间</div><div class="nap-status-value">{mtime_text}</div></div>
          <div class="nap-status"><div class="nap-status-label">最新交易日</div><div class="nap-status-value">{latest_text}</div></div>
          <div class="nap-status"><div class="nap-status-label">序列数</div><div class="nap-status-value">{status["series_count"]:,}</div></div>
          <div class="nap-status"><div class="nap-status-label">数据行数</div><div class="nap-status-value">{status["row_count"]:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _series_selector(
    catalog: pd.DataFrame,
    key: str,
    default_query: str = "",
    label: str = "选择序列",
    allow_all_sector: bool = False,
) -> str | None:
    if catalog.empty:
        st.info("没有可用序列。")
        return None

    frame = catalog.copy()
    sector_values = _sector_order(_option_values(frame, "sector", include_all=False))
    if allow_all_sector:
        sector_labels = ["全部"] + [_sector_label(value) for value in sector_values]
        sector_map = {"全部": "全部"} | {_sector_label(value): value for value in sector_values}
    else:
        sector_labels = [_sector_label(value) for value in sector_values]
        sector_map = {_sector_label(value): value for value in sector_values}
    if not sector_labels:
        st.info("没有可用板块。")
        return None

    default_index = 0
    if default_query:
        q = default_query.lower()
        matched = frame[
            (
                frame["display_name"].astype(str)
                + " "
                + frame["ric"].astype(str)
                + " "
                + frame["sector"].astype(str)
                + " "
                + frame["product"].astype(str)
                + " "
                + frame["region"].astype(str)
            ).str.lower().str.contains(q, regex=False, na=False)
        ]
        if not matched.empty:
            wanted_sector = _sector_label(matched.iloc[0].get("sector"))
            if wanted_sector in sector_labels:
                default_index = sector_labels.index(wanted_sector)

    selected_sector_label = st.selectbox("板块", sector_labels, index=default_index, key=f"{key}_sector")
    selected_sector = sector_map[selected_sector_label]
    if selected_sector != "全部":
        frame = _filter_by_option(frame, "sector", selected_sector)

    product = st.selectbox("品种", _option_values(frame, "product"), key=f"{key}_product")
    frame = _filter_by_option(frame, "product", product)

    region = st.selectbox("地区", _option_values(frame, "region"), key=f"{key}_region")
    frame = _filter_by_option(frame, "region", region)

    contract_options = _option_values(frame, "contract_month")
    if len(contract_options) > 2:
        contract = st.selectbox("合约", contract_options, key=f"{key}_contract")
        frame = _filter_by_option(frame, "contract_month", contract)

    if frame.empty:
        st.info("当前分类下没有可用序列。")
        return None

    label_map = dict(zip(frame["label"], frame["series_id"]))
    labels = list(label_map)
    selected_index = 0
    if default_query:
        q = default_query.lower()
        for idx, option in enumerate(labels):
            if q in option.lower():
                selected_index = idx
                break
    selected_label = st.selectbox(label, labels, index=selected_index, key=f"{key}_select")
    return label_map[selected_label]


def _series_from_wide(wide: pd.DataFrame, series_id: str) -> pd.Series:
    if series_id not in wide.columns:
        return pd.Series(dtype=float)
    return wide[series_id].dropna()


def _apply_fig_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    title_text = title
    if title_text is None:
        title_text = getattr(getattr(fig.layout, "title", None), "text", None)
    if title_text is None or str(title_text).strip().lower() == "undefined":
        title_text = ""
    fig.update_layout(
        template=_plot_template(),
        title={"text": str(title_text)},
        margin=dict(l=36, r=24, t=54 if title_text else 28, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif"),
    )
    return fig


def _render_market_card(row: pd.Series) -> str:
    chg = row.get("chg_1d")
    chg5 = row.get("chg_5d")
    chg20 = row.get("chg_20d")
    display = escape(str(row.get("display_name", "")))
    meta = escape(f"{row.get('product', '')} / {row.get('region', '')} / {row.get('ric', '')}")
    unit = escape(str(row.get("unit", "")))
    structure = escape(STRUCTURE_CN.get(str(row.get("structure", "-")), str(row.get("structure", "-"))))
    sector = escape(_sector_label(row.get("sector", "")))
    return textwrap.dedent(f"""
    <div class="nap-card">
      <div class="nap-card-name">{display}</div>
      <div class="nap-card-meta">{meta}</div>
      <div class="nap-card-latest"><span class="nap-price">{_fmt(row.get("latest"))}</span><span class="nap-unit">{unit}</span></div>
      <div class="nap-microgrid">
        <div class="nap-micro"><label>1D</label><strong class="{_signed_class(chg)}">{_fmt(chg)}</strong></div>
        <div class="nap-micro"><label>5D</label><strong class="{_signed_class(chg5)}">{_fmt(chg5)}</strong></div>
        <div class="nap-micro"><label>20D</label><strong class="{_signed_class(chg20)}">{_fmt(chg20)}</strong></div>
        <div class="nap-micro"><label>Z60</label><strong>{_fmt(row.get("z_60d"))}</strong></div>
        <div class="nap-micro"><label>Pct250</label><strong>{_fmt(row.get("pct_250d"), 0)}</strong></div>
        <div class="nap-micro"><label>Vol20</label><strong>{_fmt(row.get("vol_20d"), 1, pct=True)}</strong></div>
        <div class="nap-micro"><label>结构</label><strong>{structure}</strong></div>
        <div class="nap-micro"><label>日期</label><strong>{row.get("latest_date").strftime("%m-%d") if pd.notna(row.get("latest_date")) else "-"}</strong></div>
      </div>
      <span class="nap-pill">{sector}</span>
    </div>
    """).strip()


def render_market_map(df: pd.DataFrame) -> None:
    snapshot = build_market_snapshot(df)
    if snapshot.empty:
        st.warning("没有加载到 NAP 数据。")
        return
    controls = st.columns([1.2, 1, 1])
    sector_labels = {_sector_label(sector): sector for sector in MARKET_GROUPS}
    selected_sector_labels = controls[0].multiselect("分组", list(sector_labels), default=list(sector_labels))
    sector_filter = [sector_labels[label] for label in selected_sector_labels]
    search = controls[1].text_input("卡片筛选", "")
    max_cards = controls[2].slider("每组最多显示", 4, 30, 12)
    filtered = snapshot[snapshot["sector"].isin(sector_filter)]
    if search:
        q = search.lower()
        haystack = (
            filtered["display_name"].astype(str)
            + " "
            + filtered["ric"].astype(str)
            + " "
            + filtered["product"].astype(str)
            + " "
            + filtered["region"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(q, regex=False, na=False)]

    _download_csv("下载行情快照 CSV", snapshot, "nap_market_snapshot.csv", "download_market_snapshot")
    for sector in MARKET_GROUPS:
        group = filtered[filtered["sector"] == sector]
        if group.empty:
            continue
        group = group.sort_values(["contract_month", "product", "display_name"]).head(max_cards)
        st.markdown(
            f'<div class="nap-section-title"><span>{_sector_label(sector)}</span><span>{len(group)} 条序列</span></div>',
            unsafe_allow_html=True,
        )
        cards = "\n".join(_render_market_card(row) for _, row in group.iterrows())
        st.markdown(f'<div class="nap-card-grid">{cards}</div>', unsafe_allow_html=True)


def render_series_detail(df: pd.DataFrame, explanations: dict) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    left, middle, right = st.columns([0.85, 2.0, 0.95])
    with left:
        st.markdown("#### 序列分类")
        series_id = _series_selector(catalog, "detail")
        sector_counts = catalog.groupby(["sector", "product"]).size().reset_index(name="序列数")
        sector_counts["sector"] = sector_counts["sector"].map(_sector_label)
        sector_counts = sector_counts.rename(columns={"sector": "板块", "product": "品种"})
        _safe_dataframe(sector_counts, use_container_width=True, hide_index=True)
    if not series_id:
        return
    series = _series_from_wide(wide, series_id)
    meta = catalog.set_index("series_id").loc[series_id]
    with middle:
        fig = px.line(series.rename("价格"), title=str(meta["display_name"]))
        fig.update_yaxes(title="价格")
        _apply_fig_layout(fig, str(meta["display_name"]))
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载当前序列 CSV", series.to_frame("value"), f"{series_id}.csv", "detail_download")
        metrics = st.columns(5)
        metrics[0].metric("最新", _fmt(series.iloc[-1] if not series.empty else np.nan))
        metrics[1].metric("1D", _fmt(series.diff().iloc[-1] if len(series) > 1 else np.nan))
        metrics[2].metric("20D", _fmt(series.diff(20).iloc[-1] if len(series) > 20 else np.nan))
        metrics[3].metric("Z60", _fmt(zscore_of_value(series, series.iloc[-1], 60) if not series.empty else np.nan))
        metrics[4].metric("最大回撤", _fmt(max_drawdown(series), 1, pct=True))
        _safe_dataframe(series.tail(120).to_frame("value").rename(columns={"value": "数值"}), use_container_width=True)
    with right:
        explanation = _lookup_explanation(explanations, series_id, meta)
        st.markdown(
            f"""
            <div class="nap-note">
              <strong>{meta["display_name"]}</strong><br>
              RIC: {meta.get("ric", "-")}<br>
              单位: {meta.get("unit_normalized") or meta.get("unit_native", "-")}<br>
              市场角色: {explanation.get("market_role", "价格发现 / 交易参考")}<br><br>
              <strong>主要驱动</strong><br>{explanation.get("drivers", "原油、区域供需、炼厂开工、运费和宏观风险偏好。")}<br><br>
              <strong>交易用途</strong><br>{explanation.get("trading_use", "跟踪绝对价格、结构、裂解价差和相对价值。")}<br><br>
              <strong>注意事项</strong><br>{explanation.get("notes", "跨市场比较价差前请先确认单位口径。")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_seasonality(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    controls = st.columns([1, 0.55, 0.55, 0.65])
    with controls[0]:
        series_id = _series_selector(catalog, "season", label="季节性序列")
    years = controls[1].selectbox("历史窗口", [5, 10], index=0, format_func=lambda x: f"过去 {x} 年")
    remove_leap = controls[2].checkbox("剔除 2月29日", value=True)
    lunar = controls[3].checkbox("农历季节性", value=False, help="主要用于中国相关品种；当前页面保留公历视图。")
    if not series_id:
        return
    series = _series_from_wide(wide, series_id)
    if remove_leap:
        series = remove_feb29(series.to_frame("value"))["value"]
    meta = catalog.set_index("series_id").loc[series_id]
    if lunar and str(meta.get("region")) != "China":
        st.caption("农历季节性主要保留给中国相关品种；当前选择显示公历季节性。")

    matrix = seasonal_matrix(series, years=years)
    band = seasonal_percentile_band(series, years=years)
    left, right = st.columns([1.5, 1])
    with left:
        fig = go.Figure()
        x_axis = pd.to_datetime("2001-" + matrix.index.astype(str), errors="coerce") if not matrix.empty else pd.Series(dtype="datetime64[ns]")
        if not band.empty:
            band_x = pd.to_datetime("2001-" + band["doy"].astype(str), errors="coerce")
            fig.add_trace(go.Scatter(x=band_x, y=band["upper"], line=dict(width=0), showlegend=False, name="P90"))
            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band["lower"],
                    fill="tonexty",
                    fillcolor="rgba(47,125,140,0.18)",
                    line=dict(width=0),
                    name="10%-90% 分位带",
                )
            )
            fig.add_trace(go.Scatter(x=band_x, y=band["median"], name="历史中位数", line=dict(color=AMBER, width=1.7)))
        if not matrix.empty:
            latest_year = int(max(matrix.columns))
            for year in sorted(matrix.columns):
                is_latest = int(year) == latest_year
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=matrix[year],
                        mode="lines",
                        name=str(year),
                        line=dict(color=ACCENT if is_latest else "rgba(120,132,140,0.55)", width=2.7 if is_latest else 1.25),
                        opacity=1.0 if is_latest else 0.8,
                    )
                )
        fig.update_xaxes(title="月份", tickformat="%m月", dtick="M1")
        fig.update_yaxes(title="数值")
        _apply_fig_layout(fig, f"{meta['display_name']} 季节性走势")
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载季节性矩阵 CSV", matrix, "nap_seasonal_matrix.csv", "download_seasonal")
    with right:
        box = monthly_box_frame(series, years=years)
        fig_box = px.box(box, x="month", y="value", points=False, title="月度分布箱线图", template=_plot_template())
        fig_box.update_xaxes(title="月份", tickmode="array", tickvals=list(range(1, 13)), ticktext=[f"{m}月" for m in range(1, 13)])
        fig_box.update_yaxes(title="数值")
        _apply_fig_layout(fig_box, "月度分布箱线图")
        st.plotly_chart(fig_box, use_container_width=True)

    heat = calendar_heatmap_frame(series)
    if not heat.empty:
        fig_heat = px.density_heatmap(
            heat,
            x="week",
            y="weekday",
            z="value",
            histfunc="avg",
            title="日历热力图",
            template=_plot_template(),
            color_continuous_scale=["#bf5b5b", "#f1ece4", "#2f7d8c"] if st.session_state.get("nap_theme") in {"浅色", "Light"} else "Tealgrn",
        )
        fig_heat.update_xaxes(title="周数")
        fig_heat.update_yaxes(title="星期", tickmode="array", tickvals=list(range(7)), ticktext=["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
        _apply_fig_layout(fig_heat, "日历热力图")
        st.plotly_chart(fig_heat, use_container_width=True)


def render_relationship_lab(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    labels = display_lookup(catalog)
    mode = st.radio("分析模式", ["双序列分析", "公式库", "自定义价差"], horizontal=True)

    if mode == "公式库":
        registry = load_formula_registry()
        if not registry:
            st.info("没有找到公式库条目。")
            return
        formula_names = [item["name"] for item in registry]
        selected = st.selectbox("公式", formula_names)
        formula = next(item for item in registry if item["name"] == selected)
        spread = evaluate_registry_formula(wide, catalog, formula)
        if spread.empty:
            st.warning("当前 catalog 无法解析这个公式。")
            return
        fig = px.line(spread.rename(selected), title=selected, template=_plot_template())
        fig.update_yaxes(title="价差")
        _apply_fig_layout(fig, selected)
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载公式序列 CSV", spread.to_frame(selected), "nap_formula_series.csv", "download_registry_formula")
        return

    if mode == "自定义价差":
        leg_count = st.slider("腿数", 2, 5, 2)
        legs: list[tuple[str, float]] = []
        for idx in range(leg_count):
            cols = st.columns([1.7, 0.45])
            with cols[0]:
                leg_id = _series_selector(catalog, f"spread_leg_{idx}", label=f"第 {idx + 1} 条腿", allow_all_sector=True)
            weight = cols[1].number_input(f"权重 {idx + 1}", value=1.0 if idx == 0 else -1.0, step=0.25, key=f"spread_weight_{idx}")
            if leg_id:
                legs.append((leg_id, weight))
        if len(legs) < 2:
            st.info("至少选择两条腿。")
            return
        spread = build_spread_series(wide, legs)
        fig = px.line(spread.rename("自定义价差"), title="自定义价差", template=_plot_template())
        fig.update_yaxes(title="价差")
        _apply_fig_layout(fig, "自定义价差")
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载自定义价差 CSV", spread.to_frame("spread"), "nap_custom_spread.csv", "download_custom_spread")
        return

    selector_cols = st.columns(2)
    with selector_cols[0]:
        series_a = _series_selector(catalog, "rel_a", default_query="WTI", label="序列 A")
    with selector_cols[1]:
        series_b = _series_selector(catalog, "rel_b", default_query="Brent", label="序列 B")
    if not series_a or not series_b:
        return
    if series_a == series_b:
        st.warning("请选择两个不同的序列。")
        return
    window = st.slider("滚动窗口", 20, 250, 60)
    package = relationship_package(wide, series_a, series_b, window=window)
    if not package:
        st.warning("两个序列的重叠数据不足。")
        return
    aligned = package["aligned"]
    top = st.columns([1.35, 1])
    with top[0]:
        norm = aligned / aligned.iloc[0] * 100
        fig_norm = px.line(
            norm.rename(columns={"a": labels.get(series_a, series_a), "b": labels.get(series_b, series_b)}),
            title="归一化走势（起点=100）",
            template=_plot_template(),
        )
        fig_norm.update_yaxes(title="指数")
        _apply_fig_layout(fig_norm, "归一化走势（起点=100）")
        st.plotly_chart(fig_norm, use_container_width=True)
    with top[1]:
        scatter = aligned.rename(columns={"a": labels.get(series_a, series_a), "b": labels.get(series_b, series_b)})
        fig_scatter = px.scatter(scatter, x=scatter.columns[1], y=scatter.columns[0], title="散点关系", template=_plot_template())
        _apply_fig_layout(fig_scatter, "散点关系")
        st.plotly_chart(fig_scatter, use_container_width=True)

    lower = st.columns(3)
    with lower[0]:
        fig_corr = px.line(package["rolling_corr"].rename("相关系数"), title="滚动相关性", template=_plot_template())
        _apply_fig_layout(fig_corr, "滚动相关性")
        st.plotly_chart(fig_corr, use_container_width=True)
    with lower[1]:
        fig_beta = px.line(package["rolling_beta"].rename("Beta"), title="滚动 Beta", template=_plot_template())
        _apply_fig_layout(fig_beta, "滚动 Beta")
        st.plotly_chart(fig_beta, use_container_width=True)
    with lower[2]:
        residual = package["residual_z"].dropna()
        fig_resid = px.line(residual.rename("残差 Z-score"), title="回归残差 Z-score", template=_plot_template())
        fig_resid.add_hline(y=2, line_dash="dot", line_color=NEGATIVE)
        fig_resid.add_hline(y=-2, line_dash="dot", line_color=POSITIVE)
        _apply_fig_layout(fig_resid, "回归残差 Z-score")
        st.plotly_chart(fig_resid, use_container_width=True)

    lead_lag = package["lead_lag"]
    fig_lag = px.bar(lead_lag, x="lag", y="correlation", title="领先滞后相关性", template=_plot_template())
    fig_lag.update_xaxes(title="滞后天数")
    fig_lag.update_yaxes(title="相关系数")
    _apply_fig_layout(fig_lag, "领先滞后相关性")
    st.plotly_chart(fig_lag, use_container_width=True)
    _download_csv("下载关系分析数据 CSV", aligned, "nap_relationship_aligned.csv", "download_relationship")


def render_forward_curve(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    groups = available_curve_groups(catalog)
    if groups.empty:
        st.info("没有识别到远期曲线。")
        return
    options = {f"{_sector_label(r.sector)} / {r.product} / {r.region} ({r.count})": (r.sector, r.product, r.region) for r in groups.itertuples()}
    selected = st.selectbox("曲线组", list(options))
    sector, product, region = options[selected]
    curve_hist = build_curve_history(df, sector, product, region)
    if curve_hist.empty:
        st.warning("当前曲线组没有可用数据。")
        return
    curve_plot = curve_hist.copy()
    curve_plot["快照"] = curve_plot["snapshot"].map(SNAPSHOT_CN).fillna(curve_plot["snapshot"])
    fig = px.line(curve_plot, x="month_num", y="value", color="快照", markers=True, title=f"{selected} 远期曲线", template=_plot_template())
    fig.update_xaxes(title="合约月份", tickmode="array", tickvals=curve_hist["month_num"], ticktext=curve_hist["contract_month"])
    fig.update_yaxes(title="数值")
    _apply_fig_layout(fig, f"{selected} 远期曲线")
    st.plotly_chart(fig, use_container_width=True)
    current = build_forward_curve(df, sector, product, region)
    spreads = forward_curve_spreads(current)
    cols = st.columns(3)
    for idx, (name, value) in enumerate(spreads.items()):
        cols[idx].metric(name, _fmt(value))
    heat = curve_plot.pivot_table(index="快照", columns="contract_month", values="value", aggfunc="last")
    fig_heat = px.imshow(heat, title="曲线热力图", template=_plot_template(), aspect="auto", color_continuous_scale="RdBu")
    _apply_fig_layout(fig_heat, "曲线热力图")
    st.plotly_chart(fig_heat, use_container_width=True)
    _download_csv("下载曲线历史 CSV", curve_hist, "nap_forward_curve.csv", "download_curve")


def render_vol_risk(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    series_id = _series_selector(catalog, "risk", default_query="WTI", label="风险分析序列")
    if not series_id:
        return
    series = _series_from_wide(wide, series_id)
    vol_frame = pd.DataFrame(
        {
            "20D RV": realized_volatility(series, 20),
            "60D RV": realized_volatility(series, 60),
            "120D RV": realized_volatility(series, 120),
            "250D RV": realized_volatility(series, 250),
            "EWMA 60D": ewma_volatility(series, 60),
        }
    )
    left, right = st.columns([1.45, 1])
    with left:
        fig = px.line(vol_frame, title="实现波动率", template=_plot_template())
        fig.update_yaxes(title="年化波动率")
        _apply_fig_layout(fig, "实现波动率")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dd = drawdown_series(series)
        fig_dd = px.area(dd.rename("回撤"), title="回撤", template=_plot_template())
        fig_dd.update_yaxes(title="回撤")
        _apply_fig_layout(fig_dd, "回撤")
        st.plotly_chart(fig_dd, use_container_width=True)
    risk_rows = []
    for horizon in [1, 5, 20]:
        for conf in [0.95, 0.99]:
            var, es, basis = var_es_over_window(series, lookback_window=250, horizon=horizon, confidence=conf)
            risk_rows.append({"期限": f"{horizon}D", "置信度": f"{conf:.0%}", "VaR": var, "ES": es, "口径": basis})
    _safe_dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    selected = st.multiselect("风险贡献篮子", list(catalog["label"]), max_selections=6)
    if len(selected) >= 2:
        label_to_id = dict(zip(catalog["label"], catalog["series_id"]))
        returns = wide[[label_to_id[label] for label in selected]].apply(log_returns).dropna(how="all")
        rc = risk_contribution(returns)
        rc["display_name"] = rc["series_id"].map(display_lookup(catalog))
        rc_view = rc[["display_name", "weight", "volatility", "risk_contribution", "pct_contribution"]].rename(
            columns={
                "display_name": "序列",
                "weight": "权重",
                "volatility": "波动率",
                "risk_contribution": "风险贡献",
                "pct_contribution": "贡献占比",
            }
        )
        _safe_dataframe(rc_view, use_container_width=True, hide_index=True)


def render_freight_arbitrage(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    freight = catalog[catalog["sector"] == "Freight"]
    if freight.empty:
        st.info("没有加载到运费路线。")
        return
    wide = long_to_wide(df, normalized=True)
    route_labels = dict(zip(freight["label"], freight["series_id"]))
    selected = st.multiselect("运费路线", list(route_labels), default=list(route_labels)[:3], max_selections=6)
    if selected:
        selected_ids = [route_labels[label] for label in selected]
        display_names = display_lookup(catalog)
        frame = wide[selected_ids].copy()
        frame.columns = [display_names.get(series_id, series_id) for series_id in selected_ids]
        frame = frame.dropna(how="all")
        fig = px.line(frame.tail(500), title="运费路线", template=_plot_template())
        fig.update_yaxes(title="运费")
        _apply_fig_layout(fig, "运费路线")
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载运费路线 CSV", frame, "nap_freight_routes.csv", "download_freight")

    st.markdown("#### 运费调整套利")
    price_cols = st.columns(2)
    with price_cols[0]:
        origin_id = _series_selector(catalog, "arb_origin", default_query="WTI", label="起点价格", allow_all_sector=True)
    with price_cols[1]:
        dest_id = _series_selector(catalog, "arb_dest", default_query="Singapore", label="终点价格", allow_all_sector=True)
    route_cols = st.columns([1, 0.35])
    route = route_cols[0].selectbox("运费", list(route_labels), key="arb_route")
    factor = route_cols[1].number_input("运费倍数", value=1.0, step=0.25)
    if not origin_id or not dest_id:
        return
    if origin_id == dest_id:
        st.info("请选择不同的起点和终点价格。")
        return
    spread = build_spread_series(wide, [(dest_id, 1.0), (origin_id, -1.0), (route_labels[route], -factor)])
    if not spread.empty:
        fig = px.line(spread.rename("运费调整套利"), title="终点 - 起点 - 运费", template=_plot_template())
        fig.update_yaxes(title="价差")
        _apply_fig_layout(fig, "终点 - 起点 - 运费")
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("下载套利序列 CSV", spread.to_frame("spread"), "nap_freight_adjusted_arbitrage.csv", "download_arbitrage")


def render_glossary(df: pd.DataFrame, explanations: dict) -> None:
    catalog = catalog_with_labels(df)
    search = st.text_input("搜索价格词典", "")
    view = catalog.copy()
    if search:
        q = search.lower()
        haystack = (
            view["display_name"].astype(str)
            + " "
            + view["ric"].astype(str)
            + " "
            + view["sector"].astype(str)
            + " "
            + view["product"].astype(str)
            + " "
            + view["region"].astype(str)
        ).str.lower()
        view = view[haystack.str.contains(q, regex=False, na=False)]
    rows = []
    for _, row in view.iterrows():
        exp = _lookup_explanation(explanations, row["series_id"], row)
        rows.append(
            {
                "中文解释": exp.get("zh_name") or row["display_name"],
                "英文名": exp.get("english_name") or row["display_name"],
                "单位": row.get("unit_normalized") or row.get("unit_native"),
                "Reuters RIC": row.get("ric"),
                "板块": _sector_label(row.get("sector")),
                "品种": row.get("product"),
                "地区": row.get("region"),
                "市场角色": exp.get("market_role", ""),
                "主要驱动": exp.get("drivers", ""),
                "相关价差": exp.get("related_spreads", ""),
                "交易用途": exp.get("trading_use", ""),
                "注意事项": exp.get("notes", ""),
            }
        )
    table = pd.DataFrame(rows)
    _safe_dataframe(table, use_container_width=True, hide_index=True)
    _download_csv("下载价格词典 CSV", table, "nap_price_glossary.csv", "download_glossary")


def run_nap_dashboard() -> None:
    st.set_page_config(page_title="Reuters NAP 交易看板", layout="wide")
    controls = _sidebar()
    _inject_theme(str(controls["theme"]))
    try:
        df = _cached_load(
            str(controls["workbook_path"]),
            str(controls["cache_path"]),
            str(controls["catalog_path"]),
            int(controls["refresh_token"]),
        )
    except Exception as exc:
        st.error(f"加载 NAP workbook 失败: {exc}")
        logger.exception("Failed to load NAP workbook")
        return

    _render_topbar(df, str(controls["workbook_path"]))
    explanations = _load_yaml(default_explanations_path())
    page = str(controls["page"])
    if page == "market":
        render_market_map(df)
    elif page == "detail":
        render_series_detail(df, explanations)
    elif page == "seasonality":
        render_seasonality(df)
    elif page == "relationship":
        render_relationship_lab(df)
    elif page == "curve":
        render_forward_curve(df)
    elif page == "risk":
        render_vol_risk(df)
    elif page == "freight":
        render_freight_arbitrage(df)
    elif page == "glossary":
        render_glossary(df, explanations)


def main() -> None:
    run_nap_dashboard()


if __name__ == "__main__":
    main()
