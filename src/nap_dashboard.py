from __future__ import annotations

import logging
import sys
import textwrap
from html import escape
from pathlib import Path

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
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


def _plot_template() -> str:
    return DARK_TEMPLATE if st.session_state.get("nap_theme", "Light") == "Dark" else LIGHT_TEMPLATE


def _inject_theme(theme: str) -> None:
    dark = theme == "Dark"
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
    st.sidebar.markdown("### Reuters NAP")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Market Map",
            "Series Detail",
            "Seasonality",
            "Relationship Lab",
            "Forward Curve",
            "Volatility / Risk",
            "Freight / Arbitrage",
            "Price Glossary",
        ],
        label_visibility="collapsed",
    )
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
    st.session_state["nap_theme"] = theme
    workbook_path = st.sidebar.text_input("Nap.xlsx path", value=str(DEFAULT_NAP_WORKBOOK))
    cache_path = st.sidebar.text_input("Cache path", value=str(default_cache_path()))
    catalog_path = st.sidebar.text_input("Catalog path", value=str(default_catalog_path()))
    if "nap_refresh_token" not in st.session_state:
        st.session_state["nap_refresh_token"] = 0
    if st.sidebar.button("Refresh workbook cache", use_container_width=True):
        st.session_state["nap_refresh_token"] += 1
        _cached_load.clear()
    st.sidebar.caption("Cache writes parquet when pyarrow/fastparquet is available; otherwise a local pickle fallback is used.")
    return {
        "page": page,
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
            <div class="nap-brand-kicker">Reuters Excel Research Terminal</div>
            <div class="nap-brand-title">NAP Multi-Commodity Trading Dashboard</div>
          </div>
          <div class="nap-status"><div class="nap-status-label">Data updated</div><div class="nap-status-value">{mtime_text}</div></div>
          <div class="nap-status"><div class="nap-status-label">Latest trade date</div><div class="nap-status-value">{latest_text}</div></div>
          <div class="nap-status"><div class="nap-status-label">Series</div><div class="nap-status-value">{status["series_count"]:,}</div></div>
          <div class="nap-status"><div class="nap-status-label">Rows</div><div class="nap-status-value">{status["row_count"]:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _series_selector(catalog: pd.DataFrame, key: str, default_query: str = "") -> str | None:
    query = st.text_input("Search series", value=default_query, key=f"{key}_query")
    filtered = catalog.copy()
    if query:
        q = query.lower()
        haystack = (
            filtered["display_name"].astype(str)
            + " "
            + filtered["ric"].astype(str)
            + " "
            + filtered["sector"].astype(str)
            + " "
            + filtered["product"].astype(str)
            + " "
            + filtered["region"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(q, regex=False, na=False)]
    if filtered.empty:
        st.info("No matching series.")
        return None
    label_map = dict(zip(filtered["label"], filtered["series_id"]))
    selected_label = st.selectbox("Series", list(label_map), key=f"{key}_select")
    return label_map[selected_label]


def _series_from_wide(wide: pd.DataFrame, series_id: str) -> pd.Series:
    if series_id not in wide.columns:
        return pd.Series(dtype=float)
    return wide[series_id].dropna()


def _apply_fig_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=_plot_template(),
        title=title,
        margin=dict(l=36, r=24, t=54 if title else 28, b=32),
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
    structure = escape(str(row.get("structure", "-")))
    sector = escape(str(row.get("sector", "")))
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
        <div class="nap-micro"><label>Struct</label><strong>{structure}</strong></div>
        <div class="nap-micro"><label>Date</label><strong>{row.get("latest_date").strftime("%m-%d") if pd.notna(row.get("latest_date")) else "-"}</strong></div>
      </div>
      <span class="nap-pill">{sector}</span>
    </div>
    """).strip()


def render_market_map(df: pd.DataFrame) -> None:
    snapshot = build_market_snapshot(df)
    if snapshot.empty:
        st.warning("No NAP data loaded.")
        return
    controls = st.columns([1.2, 1, 1])
    sector_filter = controls[0].multiselect("Groups", MARKET_GROUPS, default=MARKET_GROUPS)
    search = controls[1].text_input("Filter cards", "")
    max_cards = controls[2].slider("Cards per group", 4, 30, 12)
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

    _download_csv("Download market snapshot", snapshot, "nap_market_snapshot.csv", "download_market_snapshot")
    for sector in MARKET_GROUPS:
        group = filtered[filtered["sector"] == sector]
        if group.empty:
            continue
        group = group.sort_values(["contract_month", "product", "display_name"]).head(max_cards)
        st.markdown(
            f'<div class="nap-section-title"><span>{sector}</span><span>{len(group)} visible series</span></div>',
            unsafe_allow_html=True,
        )
        cards = "\n".join(_render_market_card(row) for _, row in group.iterrows())
        st.markdown(f'<div class="nap-card-grid">{cards}</div>', unsafe_allow_html=True)


def render_series_detail(df: pd.DataFrame, explanations: dict) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    left, middle, right = st.columns([0.85, 2.0, 0.95])
    with left:
        st.markdown("#### Series Tree")
        series_id = _series_selector(catalog, "detail")
        sector_counts = catalog.groupby(["sector", "product"]).size().reset_index(name="series")
        _safe_dataframe(sector_counts, use_container_width=True, hide_index=True)
    if not series_id:
        return
    series = _series_from_wide(wide, series_id)
    meta = catalog.set_index("series_id").loc[series_id]
    with middle:
        fig = px.line(series.rename("value"), title=str(meta["display_name"]))
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("Download selected series", series.to_frame("value"), f"{series_id}.csv", "detail_download")
        metrics = st.columns(5)
        metrics[0].metric("Latest", _fmt(series.iloc[-1] if not series.empty else np.nan))
        metrics[1].metric("1D", _fmt(series.diff().iloc[-1] if len(series) > 1 else np.nan))
        metrics[2].metric("20D", _fmt(series.diff(20).iloc[-1] if len(series) > 20 else np.nan))
        metrics[3].metric("Z60", _fmt(zscore_of_value(series, series.iloc[-1], 60) if not series.empty else np.nan))
        metrics[4].metric("MDD", _fmt(max_drawdown(series), 1, pct=True))
        _safe_dataframe(series.tail(120).to_frame("value"), use_container_width=True)
    with right:
        explanation = _lookup_explanation(explanations, series_id, meta)
        st.markdown(
            f"""
            <div class="nap-note">
              <strong>{meta["display_name"]}</strong><br>
              RIC: {meta.get("ric", "-")}<br>
              Unit: {meta.get("unit_normalized") or meta.get("unit_native", "-")}<br>
              Role: {explanation.get("market_role", "Price discovery / trading reference")}<br><br>
              <strong>Drivers</strong><br>{explanation.get("drivers", "Crude, local balances, refinery runs, freight and macro risk appetite.")}<br><br>
              <strong>Trading use</strong><br>{explanation.get("trading_use", "Track outright level, structure, cracks and relative value.")}<br><br>
              <strong>Notes</strong><br>{explanation.get("notes", "Review unit conventions before comparing cross-market spreads.")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_seasonality(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    controls = st.columns([1, 0.55, 0.55, 0.65])
    with controls[0]:
        series_id = _series_selector(catalog, "season")
    years = controls[1].selectbox("History", [5, 10], index=0)
    remove_leap = controls[2].checkbox("Remove Feb 29", value=True)
    lunar = controls[3].checkbox("Lunar option", value=False, help="Shown for China-related series; current view keeps Gregorian dates.")
    if not series_id:
        return
    series = _series_from_wide(wide, series_id)
    if remove_leap:
        series = remove_feb29(series.to_frame("value"))["value"]
    meta = catalog.set_index("series_id").loc[series_id]
    if lunar and str(meta.get("region")) != "China":
        st.caption("Lunar seasonality is mainly reserved for China-related series; showing Gregorian seasonality for this selection.")

    matrix = seasonal_matrix(series, years=years)
    band = seasonal_percentile_band(series, years=years)
    current_year = series[series.index.year == series.index.max().year]
    left, right = st.columns([1.5, 1])
    with left:
        fig = go.Figure()
        if not band.empty:
            fig.add_trace(go.Scatter(x=band["doy"], y=band["upper"], line=dict(width=0), showlegend=False, name="P90"))
            fig.add_trace(
                go.Scatter(
                    x=band["doy"],
                    y=band["lower"],
                    fill="tonexty",
                    fillcolor="rgba(47,125,140,0.18)",
                    line=dict(width=0),
                    name="P10-P90",
                )
            )
            fig.add_trace(go.Scatter(x=band["doy"], y=band["median"], name="Median", line=dict(color=AMBER, width=1.7)))
        if not current_year.empty:
            fig.add_trace(
                go.Scatter(
                    x=current_year.index.strftime("%m-%d"),
                    y=current_year.values,
                    name=str(current_year.index.max().year),
                    line=dict(color=ACCENT, width=2.4),
                )
            )
        _apply_fig_layout(fig, f"{meta['display_name']} seasonality")
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("Download seasonal matrix", matrix, "nap_seasonal_matrix.csv", "download_seasonal")
    with right:
        box = monthly_box_frame(series, years=years)
        fig_box = px.box(box, x="month", y="value", points=False, title="Monthly distribution", template=_plot_template())
        _apply_fig_layout(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    heat = calendar_heatmap_frame(series)
    if not heat.empty:
        fig_heat = px.density_heatmap(
            heat,
            x="week",
            y="weekday",
            z="value",
            histfunc="avg",
            title="Calendar heatmap",
            template=_plot_template(),
            color_continuous_scale=["#bf5b5b", "#f1ece4", "#2f7d8c"] if st.session_state.get("nap_theme") == "Light" else "Tealgrn",
        )
        fig_heat.update_yaxes(tickmode="array", tickvals=list(range(7)), ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        _apply_fig_layout(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)


def render_relationship_lab(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    labels = display_lookup(catalog)
    label_options = {f"{row.display_name} · {row.ric}": row.series_id for row in catalog.itertuples()}
    mode = st.radio("Relationship mode", ["Two-series lab", "Formula registry", "Manual spread builder"], horizontal=True)

    if mode == "Formula registry":
        registry = load_formula_registry()
        if not registry:
            st.info("No formula registry entries found.")
            return
        formula_names = [item["name"] for item in registry]
        selected = st.selectbox("Formula", formula_names)
        formula = next(item for item in registry if item["name"] == selected)
        spread = evaluate_registry_formula(wide, catalog, formula)
        if spread.empty:
            st.warning("The selected formula could not be resolved against the current catalog.")
            return
        fig = px.line(spread.rename(selected), title=selected, template=_plot_template())
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("Download formula series", spread.to_frame(selected), "nap_formula_series.csv", "download_registry_formula")
        return

    if mode == "Manual spread builder":
        selected = st.multiselect("Legs", list(label_options), max_selections=5)
        if not selected:
            st.info("Select two or more legs.")
            return
        legs: list[tuple[str, float]] = []
        cols = st.columns(min(len(selected), 4))
        for idx, label in enumerate(selected):
            weight = cols[idx % len(cols)].number_input(f"Weight {idx + 1}", value=1.0 if idx == 0 else -1.0, step=0.25, key=f"spread_weight_{idx}")
            legs.append((label_options[label], weight))
        spread = build_spread_series(wide, legs)
        fig = px.line(spread.rename("spread"), title="Custom spread", template=_plot_template())
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("Download custom spread", spread.to_frame("spread"), "nap_custom_spread.csv", "download_custom_spread")
        return

    selected = st.multiselect("Select two or more series", list(label_options), max_selections=4)
    if len(selected) < 2:
        st.info("Select at least two series.")
        return
    ids = [label_options[label] for label in selected]
    window = st.slider("Rolling window", 20, 250, 60)
    package = relationship_package(wide, ids[0], ids[1], window=window)
    if not package:
        st.warning("Not enough overlapping data.")
        return
    aligned = package["aligned"]
    top = st.columns([1.35, 1])
    with top[0]:
        norm = aligned / aligned.iloc[0] * 100
        fig_norm = px.line(norm.rename(columns={"a": labels.get(ids[0], ids[0]), "b": labels.get(ids[1], ids[1])}), title="Indexed path (start=100)", template=_plot_template())
        _apply_fig_layout(fig_norm)
        st.plotly_chart(fig_norm, use_container_width=True)
    with top[1]:
        scatter = aligned.rename(columns={"a": labels.get(ids[0], ids[0]), "b": labels.get(ids[1], ids[1])})
        fig_scatter = px.scatter(scatter, x=scatter.columns[1], y=scatter.columns[0], trendline="ols", title="Scatter", template=_plot_template())
        _apply_fig_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    lower = st.columns(3)
    with lower[0]:
        fig_corr = px.line(package["rolling_corr"].rename("correlation"), title="Rolling correlation", template=_plot_template())
        _apply_fig_layout(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
    with lower[1]:
        fig_beta = px.line(package["rolling_beta"].rename("beta"), title="Rolling beta", template=_plot_template())
        _apply_fig_layout(fig_beta)
        st.plotly_chart(fig_beta, use_container_width=True)
    with lower[2]:
        residual = package["residual_z"].dropna()
        fig_resid = px.line(residual.rename("residual z"), title="Regression residual z-score", template=_plot_template())
        fig_resid.add_hline(y=2, line_dash="dot", line_color=NEGATIVE)
        fig_resid.add_hline(y=-2, line_dash="dot", line_color=POSITIVE)
        _apply_fig_layout(fig_resid)
        st.plotly_chart(fig_resid, use_container_width=True)

    lead_lag = package["lead_lag"]
    fig_lag = px.bar(lead_lag, x="lag", y="correlation", title="Lead-lag correlation", template=_plot_template())
    _apply_fig_layout(fig_lag)
    st.plotly_chart(fig_lag, use_container_width=True)
    _download_csv("Download relationship data", aligned, "nap_relationship_aligned.csv", "download_relationship")


def render_forward_curve(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    groups = available_curve_groups(catalog)
    if groups.empty:
        st.info("No forward curves detected.")
        return
    options = {f"{r.sector} / {r.product} / {r.region} ({r.count})": (r.sector, r.product, r.region) for r in groups.itertuples()}
    selected = st.selectbox("Curve group", list(options))
    sector, product, region = options[selected]
    curve_hist = build_curve_history(df, sector, product, region)
    if curve_hist.empty:
        st.warning("No curve data for this group.")
        return
    fig = px.line(curve_hist, x="month_num", y="value", color="snapshot", markers=True, title=selected, template=_plot_template())
    fig.update_xaxes(tickmode="array", tickvals=curve_hist["month_num"], ticktext=curve_hist["contract_month"])
    _apply_fig_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
    current = build_forward_curve(df, sector, product, region)
    spreads = forward_curve_spreads(current)
    cols = st.columns(3)
    for idx, (name, value) in enumerate(spreads.items()):
        cols[idx].metric(name, _fmt(value))
    heat = curve_hist.pivot_table(index="snapshot", columns="contract_month", values="value", aggfunc="last")
    fig_heat = px.imshow(heat, title="Curve heatmap", template=_plot_template(), aspect="auto", color_continuous_scale="RdBu")
    _apply_fig_layout(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)
    _download_csv("Download curve history", curve_hist, "nap_forward_curve.csv", "download_curve")


def render_vol_risk(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    series_id = _series_selector(catalog, "risk", default_query="WTI")
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
        fig = px.line(vol_frame, title="Realized volatility", template=_plot_template())
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dd = drawdown_series(series)
        fig_dd = px.area(dd.rename("drawdown"), title="Drawdown", template=_plot_template())
        _apply_fig_layout(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)
    risk_rows = []
    for horizon in [1, 5, 20]:
        for conf in [0.95, 0.99]:
            var, es, basis = var_es_over_window(series, lookback_window=250, horizon=horizon, confidence=conf)
            risk_rows.append({"horizon": f"{horizon}D", "confidence": f"{conf:.0%}", "VaR": var, "ES": es, "basis": basis})
    _safe_dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    selected = st.multiselect("Risk contribution basket", list(catalog["label"]), max_selections=6)
    if len(selected) >= 2:
        label_to_id = dict(zip(catalog["label"], catalog["series_id"]))
        returns = wide[[label_to_id[label] for label in selected]].apply(log_returns).dropna(how="all")
        rc = risk_contribution(returns)
        rc["display_name"] = rc["series_id"].map(display_lookup(catalog))
        _safe_dataframe(rc[["display_name", "weight", "volatility", "risk_contribution", "pct_contribution"]], use_container_width=True, hide_index=True)


def render_freight_arbitrage(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    freight = catalog[catalog["sector"] == "Freight"]
    if freight.empty:
        st.info("No freight routes loaded.")
        return
    wide = long_to_wide(df, normalized=True)
    route_labels = dict(zip(freight["label"], freight["series_id"]))
    selected = st.multiselect("Freight routes", list(route_labels), default=list(route_labels)[:3], max_selections=6)
    if selected:
        frame = wide[[route_labels[label] for label in selected]].rename(columns=display_lookup(catalog)).dropna(how="all")
        fig = px.line(frame.tail(500), title="Freight routes", template=_plot_template())
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        _download_csv("Download freight routes", frame, "nap_freight_routes.csv", "download_freight")

    st.markdown("#### Freight-adjusted arbitrage")
    cols = st.columns(4)
    origin = cols[0].selectbox("Origin price", list(catalog["label"]), key="arb_origin")
    dest = cols[1].selectbox("Destination price", list(catalog["label"]), key="arb_dest")
    route = cols[2].selectbox("Freight", list(route_labels), key="arb_route")
    factor = cols[3].number_input("Freight multiplier", value=1.0, step=0.25)
    label_to_id = dict(zip(catalog["label"], catalog["series_id"]))
    spread = build_spread_series(wide, [(label_to_id[dest], 1.0), (label_to_id[origin], -1.0), (route_labels[route], -factor)])
    if not spread.empty:
        fig = px.line(spread.rename("freight_adjusted_arbitrage"), title="Destination - origin - freight", template=_plot_template())
        _apply_fig_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_glossary(df: pd.DataFrame, explanations: dict) -> None:
    catalog = catalog_with_labels(df)
    search = st.text_input("Search glossary", "")
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
                "English name": exp.get("english_name") or row["display_name"],
                "Unit": row.get("unit_normalized") or row.get("unit_native"),
                "Reuters RIC": row.get("ric"),
                "Sector": row.get("sector"),
                "Product": row.get("product"),
                "Region": row.get("region"),
                "市场角色": exp.get("market_role", ""),
                "主要驱动": exp.get("drivers", ""),
                "相关价差": exp.get("related_spreads", ""),
                "交易用途": exp.get("trading_use", ""),
                "注意事项": exp.get("notes", ""),
            }
        )
    table = pd.DataFrame(rows)
    _safe_dataframe(table, use_container_width=True, hide_index=True)
    _download_csv("Download glossary", table, "nap_price_glossary.csv", "download_glossary")


def run_nap_dashboard() -> None:
    st.set_page_config(page_title="Reuters NAP Trading Dashboard", layout="wide")
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
        st.error(f"Failed to load NAP workbook: {exc}")
        logger.exception("Failed to load NAP workbook")
        return

    _render_topbar(df, str(controls["workbook_path"]))
    explanations = _load_yaml(default_explanations_path())
    page = str(controls["page"])
    if page == "Market Map":
        render_market_map(df)
    elif page == "Series Detail":
        render_series_detail(df, explanations)
    elif page == "Seasonality":
        render_seasonality(df)
    elif page == "Relationship Lab":
        render_relationship_lab(df)
    elif page == "Forward Curve":
        render_forward_curve(df)
    elif page == "Volatility / Risk":
        render_vol_risk(df)
    elif page == "Freight / Arbitrage":
        render_freight_arbitrage(df)
    elif page == "Price Glossary":
        render_glossary(df, explanations)


def main() -> None:
    run_nap_dashboard()


if __name__ == "__main__":
    main()
