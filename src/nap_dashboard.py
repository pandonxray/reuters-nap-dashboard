from __future__ import annotations

import logging
import hashlib
import re
import shutil
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
from plotly.subplots import make_subplots
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
NEUTRAL = "#7f8c8d"


PAGE_OPTIONS = {
    "行情地图": "market",
    "序列详情": "detail",
    "季节性": "seasonality",
    "关系实验室": "relationship",
    "组合价差": "combos",
    "周报出图": "weekly",
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
    "backwardation": "现货升水",
    "contango": "远期升水",
    "flat": "平水",
    "flat/spot": "现货 / 无曲线",
    "calendar": "自然月",
}

SNAPSHOT_CN = {
    "Current": "当前",
    "7D ago": "1周前",
    "30D ago": "1个月前",
    "90D ago": "3个月前",
}

NAPHTHA_BBLS_PER_MT = 8.9
EBOB_BBLS_PER_MT = 8.33
GASOIL_BBLS_PER_MT = 7.45
CONTRACT_MONTHS = [f"M{idx}" for idx in range(1, 13)]
CALENDAR_MONTHS = list(range(1, 13))
PPT_WIDTH = 1600
PPT_HEIGHT = 900
TERM_MODE_OPTIONS = {
    "连续月 C1-C12": "continuous",
    "自然月 1-12月": "calendar",
}
TERM_MODE_CN = {
    "continuous": "连续月",
    "calendar": "自然月",
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
            font-size: 0.74rem;
            line-height: 1.1;
            font-weight: 680;
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
        .nap-report-strip {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.75rem 0 0.85rem;
            padding: 0.85rem 1rem;
            background: var(--nap-panel);
            border: 1px solid var(--nap-line);
            border-radius: 8px;
        }}
        .nap-report-strip strong {{
            display: block;
            color: var(--nap-text);
            font-size: 1rem;
            line-height: 1.2;
        }}
        .nap-report-strip em {{
            display: block;
            color: var(--nap-muted);
            font-size: 0.82rem;
            font-style: normal;
            margin-top: 0.2rem;
        }}
        .nap-report-meta {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}
        .nap-report-meta span {{
            border: 1px solid var(--nap-line);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: var(--nap-panel-soft);
            color: var(--nap-muted);
            font-size: 0.75rem;
        }}
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
        section[data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] span {{
            font-size: 0;
        }}
        section[data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] span::after {{
            content: "拖拽 Excel 到这里";
            font-size: 0.92rem;
        }}
        section[data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] small {{
            font-size: 0;
        }}
        section[data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] small::after {{
            content: "单个文件上限 200MB · XLSX";
            font-size: 0.82rem;
        }}
        section[data-testid="stFileUploaderDropzone"] button {{
            font-size: 0;
        }}
        section[data-testid="stFileUploaderDropzone"] button::after {{
            content: "选择文件";
            font-size: 0.9rem;
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


def _plotly_config(filename: str) -> dict[str, object]:
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": PPT_HEIGHT,
            "width": PPT_WIDTH,
            "scale": 2,
        },
    }


def _download_figure_png(label: str, fig: go.Figure, filename: str, key: str) -> None:
    try:
        png = fig.to_image(format="png", width=PPT_WIDTH, height=PPT_HEIGHT, scale=2)
    except Exception as exc:
        st.caption(f"{label} 暂不可用：{exc}")
        return
    st.download_button(label, png, file_name=filename, mime="image/png", key=key)


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


def _default_index(options: list[str], *keywords: str) -> int:
    lowered = [keyword.lower() for keyword in keywords if keyword]
    for idx, option in enumerate(options):
        text = str(option).lower()
        if all(keyword in text for keyword in lowered):
            return idx
    for idx, option in enumerate(options):
        text = str(option).lower()
        if any(keyword in text for keyword in lowered):
            return idx
    return 0


def _workbook_signature(path: str | Path) -> str:
    workbook = Path(path)
    if not workbook.exists():
        return "missing"
    stat = workbook.stat()
    try:
        resolved = str(workbook.resolve())
    except OSError:
        resolved = str(workbook)
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def _signature_cache_path(base_cache_path: str | Path, signature: str) -> Path:
    base = Path(base_cache_path)
    token = hashlib.blake2b(signature.encode("utf-8"), digest_size=6).hexdigest()
    return base.with_name(f"{base.stem}_{token}{base.suffix}")


def _cache_files(cache_path: Path) -> list[Path]:
    return [cache_path, cache_path.with_suffix(cache_path.suffix + ".pkl")]


def _existing_cache_file(cache_path: Path) -> Path | None:
    for candidate in _cache_files(cache_path):
        if candidate.exists():
            return candidate
    return None


def _bootstrap_signature_cache(cache_path: Path, workbook_path: str | Path) -> None:
    workbook = Path(workbook_path)
    if _existing_cache_file(cache_path) is not None or not workbook.exists():
        return
    source = _existing_cache_file(default_cache_path())
    if source is None or source.stat().st_mtime < workbook.stat().st_mtime:
        return
    target = cache_path.with_suffix(cache_path.suffix + ".pkl") if source.name.endswith(".pkl") else cache_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _persist_uploaded_workbook(uploaded_file) -> tuple[str | None, str | None]:
    if uploaded_file is None:
        return None, None
    payload = uploaded_file.getvalue()
    digest = hashlib.blake2b(payload, digest_size=10).hexdigest()
    upload_dir = BASE_DIR / "data" / "raw" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"Nap_upload_{digest}.xlsx"
    if not path.exists() or path.stat().st_size != len(payload):
        path.write_bytes(payload)
    return str(path), digest


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


@st.cache_resource(show_spinner=False)
def _cached_load(workbook_path: str, cache_path: str, catalog_path: str, refresh_token: int, workbook_signature: str) -> pd.DataFrame:
    return load_nap_timeseries(workbook_path, cache_path=cache_path, catalog_path=catalog_path, refresh=refresh_token > 0)


def _filter_term_view(df: pd.DataFrame, term_mode: str, calendar_months: list[int] | None = None) -> pd.DataFrame:
    if df.empty or "term_type" not in df.columns:
        return df
    if term_mode == "calendar":
        out = df[df["term_type"].astype(str).eq("calendar")]
        if calendar_months and "calendar_month" in out.columns:
            selected = {int(month) for month in calendar_months}
            months = pd.to_numeric(out["calendar_month"], errors="coerce")
            out = out[months.isin(selected)]
        return out
    return df[~df["term_type"].astype(str).eq("calendar")]


def _sidebar() -> dict[str, object]:
    st.sidebar.markdown("### Reuters NAP 交易看板")
    page = st.sidebar.radio(
        "导航",
        list(PAGE_OPTIONS),
        label_visibility="collapsed",
    )
    theme = st.sidebar.radio("主题", ["浅色", "深色"], horizontal=True)
    st.session_state["nap_theme"] = theme
    view_label = st.sidebar.radio("查看模式", list(TERM_MODE_OPTIONS), horizontal=False)
    term_mode = TERM_MODE_OPTIONS[view_label]
    selected_calendar_months = CALENDAR_MONTHS
    if term_mode == "calendar":
        selected_calendar_months = st.sidebar.multiselect(
            "自然月组合",
            CALENDAR_MONTHS,
            default=CALENDAR_MONTHS,
            format_func=lambda month: f"{month}月",
            help="自然月模式只显示已勾选月份；可用于只看旺季、淡季或自定义月份组合。",
        )
        if not selected_calendar_months:
            selected_calendar_months = CALENDAR_MONTHS
    uploaded = st.sidebar.file_uploader("拖入 NAP Excel", type=["xlsx"], help="可拖入 Nap.xlsx 或 Nap_calendar_month_live_formula.xlsx；系统会按文件内容生成专属缓存，不会复用旧 workbook 的缓存。")
    uploaded_path, uploaded_digest = _persist_uploaded_workbook(uploaded)
    workbook_input = st.sidebar.text_input("或输入 NAP Excel 路径", value=str(DEFAULT_NAP_WORKBOOK))
    workbook_path = uploaded_path or workbook_input
    signature = _workbook_signature(workbook_path)
    cache_path = _signature_cache_path(default_cache_path(), signature)
    _bootstrap_signature_cache(cache_path, workbook_path)
    catalog_path = st.sidebar.text_input("序列目录路径", value=str(default_catalog_path()))
    if "nap_refresh_token" not in st.session_state:
        st.session_state["nap_refresh_token"] = 0
    if st.sidebar.button("重新解析当前 Excel", use_container_width=True):
        st.session_state["nap_refresh_token"] += 1
        _cached_load.clear()
    source_label = "拖拽上传" if uploaded_path else "路径读取"
    st.sidebar.caption(
        f"数据来源：{source_label}。当前文件会按路径、大小和修改时间生成独立缓存；Excel 更新后会自动重新计算，不会被旧缓存覆盖。"
    )
    st.sidebar.caption(f"当前文件：{workbook_path}")
    st.sidebar.caption(f"缓存文件：{cache_path.name}")
    return {
        "page": PAGE_OPTIONS[page],
        "theme": theme,
        "term_mode": term_mode,
        "calendar_months": selected_calendar_months,
        "workbook_path": workbook_path,
        "cache_path": cache_path,
        "catalog_path": catalog_path,
        "refresh_token": st.session_state["nap_refresh_token"],
        "workbook_signature": signature,
        "uploaded_digest": uploaded_digest or "",
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
            <div class="nap-brand-kicker">REUTERS EXCEL 交易研究终端</div>
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

    if "term_type" in frame.columns and frame["term_type"].astype(str).eq("calendar").all() and "calendar_month" in frame.columns:
        month_values = sorted(
            {
                int(value)
                for value in pd.to_numeric(frame["calendar_month"], errors="coerce").dropna().astype(int)
                if 1 <= int(value) <= 12
            }
        )
        if len(month_values) > 1:
            selected_month = st.selectbox(
                "自然月",
                month_values,
                format_func=lambda month: f"{month}月",
                key=f"{key}_calendar_month",
            )
            frame = frame[pd.to_numeric(frame["calendar_month"], errors="coerce").eq(selected_month)]

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


def _find_series_id(
    catalog: pd.DataFrame,
    *,
    sector: str | None = None,
    product: str | None = None,
    region: str | None = None,
    contract_month: str | None = None,
    term_type: str | None = None,
    calendar_month: int | str | None = None,
    ric_prefix: str | None = None,
    contains: list[str] | None = None,
) -> str | None:
    frame = catalog.copy()
    for column, value in {
        "sector": sector,
        "product": product,
        "region": region,
        "contract_month": contract_month,
        "term_type": term_type,
        "calendar_month": calendar_month,
    }.items():
        if value is None or column not in frame.columns:
            continue
        if column == "calendar_month":
            frame = frame[pd.to_numeric(frame[column], errors="coerce").eq(int(value))]
        else:
            frame = frame[frame[column].astype(str).str.casefold() == str(value).casefold()]
    if ric_prefix:
        frame = frame[frame["ric"].astype(str).str.upper().str.startswith(ric_prefix.upper(), na=False)]
    if contains:
        haystack = (
            frame.get("display_name", pd.Series("", index=frame.index)).astype(str)
            + " "
            + frame.get("short_name", pd.Series("", index=frame.index)).astype(str)
            + " "
            + frame.get("ric", pd.Series("", index=frame.index)).astype(str)
        ).str.casefold()
        for keyword in contains:
            frame = frame[haystack.str.contains(str(keyword).casefold(), regex=False, na=False)]
            haystack = haystack.loc[frame.index]
    if frame.empty:
        return None
    return str(frame.sort_values(["contract_month", "display_name", "ric"]).iloc[0]["series_id"])


def _spread_series(
    wide: pd.DataFrame,
    left_id: str | None,
    right_id: str | None,
    *,
    right_divisor: float = 1.0,
) -> pd.Series:
    if not left_id or not right_id or left_id not in wide.columns or right_id not in wide.columns:
        return pd.Series(dtype=float)
    aligned = pd.concat([wide[left_id].rename("left"), wide[right_id].rename("right")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["left"] - aligned["right"] / float(right_divisor)


def _combo_definitions() -> dict[str, dict[str, object]]:
    return {
        "新加坡92汽油纸货 - MOPJ（统一 USD/bbl）": {
            "left": {"sector": "Gasoline", "product": "新加坡92汽油", "region": "新加坡"},
            "right": {"sector": "Naphtha", "product": "日本CFR石脑油", "region": "日本/东北亚"},
            "unit": "USD/bbl",
            "formula": "Singapore 92 - MOPJ/8.90",
            "description": "新加坡92RON汽油纸货逐月减 MOPJ CFR Japan 石脑油逐月。计算统一成 USD/bbl：新加坡92原始为 USD/bbl；MOPJ 原始为 USD/mt，先除以 8.90 转为 USD/bbl；公式 = Singapore 92 - MOPJ/8.90。",
        },
        "欧洲EBOB汽油纸货 - NWE CIF石脑油（统一 USD/bbl）": {
            "left": {"sector": "Gasoline", "product": "欧洲EBOB汽油", "region": "欧洲"},
            "right": {"sector": "Naphtha", "product": "NWE CIF石脑油", "region": "西北欧"},
            "unit": "USD/bbl",
            "formula": "EBOB/8.33 - NWE/8.90",
            "description": "欧洲 EBOB 汽油纸货逐月减 Naphtha CIF NWE outright swap 逐月。计算统一成 USD/bbl：EBOB 原始为 USD/mt，先除以 8.33 转为 USD/bbl；NWE 石脑油原始为 USD/mt，先除以 8.90 转为 USD/bbl；公式 = EBOB/8.33 - NWE/8.90。",
        },
    }


def _monthly_combo_frame(wide: pd.DataFrame, catalog: pd.DataFrame, combo: dict[str, object]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows: list[dict[str, object]] = []
    series_by_month: dict[str, pd.Series] = {}
    left_selector = dict(combo["left"])  # type: ignore[index]
    right_selector = dict(combo["right"])  # type: ignore[index]
    right_divisor = float(combo.get("right_divisor", 1.0))
    is_calendar = "term_type" in catalog.columns and catalog["term_type"].astype(str).eq("calendar").all()
    month_axis: list[tuple[str, dict[str, object]]] = []
    if is_calendar:
        available = sorted(pd.to_numeric(catalog.get("calendar_month"), errors="coerce").dropna().astype(int).unique())
        month_axis = [(f"{month}月", {"term_type": "calendar", "calendar_month": int(month)}) for month in available if 1 <= int(month) <= 12]
    else:
        month_axis = [(month, {"contract_month": month}) for month in CONTRACT_MONTHS]
    for month, month_selector in month_axis:
        left_id = _find_series_id(catalog, **month_selector, **left_selector)
        right_id = _find_series_id(catalog, **month_selector, **right_selector)
        spread = _spread_series(wide, left_id, right_id, right_divisor=right_divisor)
        if spread.empty:
            rows.append({"合约": month, "价差": np.nan, "最新日期": pd.NaT, "左腿": left_id or "-", "右腿": right_id or "-"})
            continue
        spread.name = month
        series_by_month[month] = spread
        rows.append(
            {
                "合约": month,
                "价差": float(spread.iloc[-1]),
                "1D": float(spread.diff().iloc[-1]) if len(spread) > 1 else np.nan,
                "5D": float(spread.diff(5).iloc[-1]) if len(spread) > 5 else np.nan,
                "20D": float(spread.diff(20).iloc[-1]) if len(spread) > 20 else np.nan,
                "最新日期": spread.index[-1],
                "左腿": left_id or "-",
                "右腿": right_id or "-",
            }
        )
    return pd.DataFrame(rows), series_by_month


def _contract_number(month: str) -> int:
    match = re.search(r"(\d+)", str(month))
    return int(match.group(1)) if match else 999


def _seasonal_year_palette() -> list[str]:
    return [
        "#2f7d8c",
        "#d29b47",
        "#7a5c9e",
        "#4b9b72",
        "#bf5b5b",
        "#3f6ea8",
        "#a86f3f",
        "#6c8f3f",
        "#9a5477",
        "#5d8791",
    ]


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


def _weekly_year_color(year: int, latest_year: int) -> str:
    fixed = {
        latest_year: "#c75b5b",
        latest_year - 1: "#55a173",
        latest_year - 2: "#8064a2",
        latest_year - 3: "#e39c35",
        latest_year - 4: "#2f8ca3",
    }
    return fixed.get(int(year), _seasonal_year_palette()[int(year) % len(_seasonal_year_palette())])


def _add_weekly_seasonality_panel(
    fig: go.Figure,
    series: pd.Series,
    *,
    row: int,
    col: int,
    years: int,
    show_legend: bool,
    y_title: str,
    zero_line: bool = False,
) -> None:
    cleaned = series.dropna()
    if cleaned.empty:
        fig.add_annotation(
            text="无可用数据",
            x=0.5,
            y=0.5,
            xref=f"x{(row - 1) * 2 + col} domain",
            yref=f"y{(row - 1) * 2 + col} domain",
            showarrow=False,
            font=dict(color="#7b8790", size=13),
        )
        return
    cleaned = remove_feb29(cleaned.to_frame("value"))["value"]
    matrix = seasonal_matrix(cleaned, years=years)
    band = seasonal_percentile_band(cleaned, years=max(years, 10))
    range_values: list[float] = []
    x_axis = pd.to_datetime("2001-" + matrix.index.astype(str), errors="coerce") if not matrix.empty else pd.Series(dtype="datetime64[ns]")
    if not band.empty:
        range_values.extend(pd.concat([band["lower"], band["median"], band["upper"]]).dropna().astype(float).tolist())
        band_x = pd.to_datetime("2001-" + band["doy"].astype(str), errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band["upper"],
                line=dict(width=0),
                showlegend=False,
                legendgroup="band",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band["lower"],
                fill="tonexty",
                fillcolor="rgba(47,125,140,0.15)",
                line=dict(width=0),
                name="10%-90% 分位带",
                legendgroup="band",
                legendrank=90,
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band["median"],
                name="历史中位数",
                legendgroup="median",
                legendrank=80,
                showlegend=show_legend,
                mode="lines",
                line=dict(color="#c28b38", width=1.5),
            ),
            row=row,
            col=col,
        )
    if not matrix.empty:
        range_values.extend(matrix.stack().dropna().astype(float).tolist())
        latest_year = int(max(matrix.columns))
        for year in sorted(matrix.columns):
            year_int = int(year)
            is_latest = year_int == latest_year
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=matrix[year],
                    name=str(year_int),
                    legendgroup=str(year_int),
                    legendrank=10 + max(0, latest_year - year_int),
                    showlegend=show_legend,
                    mode="lines",
                    line=dict(color=_weekly_year_color(year_int, latest_year), width=3.0 if is_latest else 1.7),
                    opacity=1.0 if is_latest else 0.95,
                ),
                row=row,
                col=col,
            )
    if zero_line:
        fig.add_hline(y=0, line_dash="dot", line_color="#8c9aa3", line_width=1, row=row, col=col)
    fig.update_xaxes(title_text="月份", tickformat="%m月", dtick="M1", showgrid=True, gridcolor="#e5ebef", row=row, col=col)
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=row, col=col)
    if range_values:
        sample = pd.Series(range_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if not sample.empty:
            low = float(sample.quantile(0.01))
            high = float(sample.quantile(0.99))
            if not matrix.empty:
                visible_years = matrix.stack().dropna().astype(float)
                if not visible_years.empty:
                    low = min(low, float(visible_years.min()))
                    high = max(high, float(visible_years.max()))
            if zero_line:
                low = min(low, 0.0)
                high = max(high, 0.0)
            if high > low:
                pad = (high - low) * 0.08
                fig.update_yaxes(range=[low - pad, high + pad], row=row, col=col)


def _weekly_seasonality_figure(
    panels: list[dict[str, object]],
    *,
    title: str,
    years: int = 5,
    y_title: str = "数值",
    zero_line: bool = False,
) -> go.Figure:
    slots = panels[:4]
    while len(slots) < 4:
        slots.append({"title": "", "series": pd.Series(dtype=float)})
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[str(item.get("title", "")) for item in slots],
        horizontal_spacing=0.075,
        vertical_spacing=0.15,
    )
    for idx, item in enumerate(slots):
        row = idx // 2 + 1
        col = idx % 2 + 1
        series = item.get("series")
        _add_weekly_seasonality_panel(
            fig,
            series if isinstance(series, pd.Series) else pd.Series(dtype=float),
            row=row,
            col=col,
            years=years,
            show_legend=idx == 0,
            y_title=y_title,
            zero_line=zero_line,
        )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=""),
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=54, r=36, t=92, b=54),
        legend=dict(orientation="h", yanchor="bottom", y=1.035, xanchor="right", x=1, font=dict(size=12)),
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, sans-serif", size=12, color="#27323a"),
    )
    fig.update_annotations(font=dict(size=15, color="#1f2a33"))
    return fig


def _weekly_chart(fig: go.Figure, filename: str, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config(filename))
    _download_figure_png("下载 PNG", fig, f"{filename}.png", key)


def _render_market_card(row: pd.Series) -> str:
    chg = row.get("chg_1d")
    chg5 = row.get("chg_5d")
    chg20 = row.get("chg_20d")
    display = escape(str(row.get("display_name", "")))
    month_meta = ""
    if str(row.get("term_type", "continuous")) == "calendar" and pd.notna(row.get("calendar_month")) and str(row.get("calendar_month")).strip():
        month_meta = f" / {int(float(row.get('calendar_month')))}月"
    elif str(row.get("contract_month", "")).strip():
        month_meta = f" / {row.get('contract_month')}"
    meta = escape(f"{row.get('product', '')} / {row.get('region', '')}{month_meta} / {row.get('ric', '')}")
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
        group = group.copy()
        group["sort_month"] = pd.to_numeric(group.get("calendar_month", ""), errors="coerce").fillna(
            group["contract_month"].map(_contract_number)
        )
        group = group.sort_values(["product", "region", "sort_month", "contract_month", "display_name"]).head(max_cards)
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
        unit_native = meta.get("unit_native", "-")
        unit_normalized = meta.get("unit_normalized") or unit_native
        unit_conversion = meta.get("unit_conversion") or "未设置额外换算；使用原始报价口径。"
        unit_source = meta.get("unit_source") or "当前 workbook/catalog 推断。"
        if unit_native != unit_normalized:
            unit_note = f"原始单位为 {unit_native}，看板计算和跨品种比较使用 {unit_normalized}。"
        else:
            unit_note = f"当前序列使用 {unit_normalized}，跨品种比较前仍需确认是否为同一报价口径。"
        st.markdown(
            f"""
            <div class="nap-note">
              <strong>{meta["display_name"]}</strong><br>
              RIC: {meta.get("ric", "-")}<br>
              板块 / 品种 / 地区: {_sector_label(meta.get("sector"))} / {meta.get("product", "-")} / {meta.get("region", "-")}<br>
              单位: {unit_normalized}<br><br>
              <strong>市场角色</strong><br>{explanation.get("market_role", "用于观察该市场的绝对价格、远期结构、相对强弱和交易参考。")}<br><br>
              <strong>主要驱动</strong><br>{explanation.get("drivers", "原油、区域供需、炼厂开工、运费和宏观风险偏好。")}<br><br>
              <strong>相关价差</strong><br>{explanation.get("related_spreads", "可与同区域月差、跨区价差、裂解价差、炼厂利润和运费调整套利一起观察。")}<br><br>
              <strong>交易用途</strong><br>{explanation.get("trading_use", "跟踪绝对价格、结构、裂解价差和相对价值。")}<br><br>
              <strong>数据口径</strong><br>{unit_note}<br>换算公式: {unit_conversion}<br>单位依据: {unit_source}<br><br>
              <strong>注意事项</strong><br>{explanation.get("notes", "跨市场比较价差前请先确认单位、合约月份、报价地点和是否为评估价/期货连续合约。")}
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
            year_palette = [
                "#2f7d8c",
                "#d29b47",
                "#7a5c9e",
                "#4b9b72",
                "#bf5b5b",
                "#3f6ea8",
                "#a86f3f",
                "#6c8f3f",
                "#9a5477",
                "#5d8791",
            ]
            for idx, year in enumerate(sorted(matrix.columns)):
                is_latest = int(year) == latest_year
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=matrix[year],
                        mode="lines",
                        name=str(year),
                        line=dict(color=year_palette[idx % len(year_palette)], width=3.0 if is_latest else 1.8),
                        opacity=1.0 if is_latest else 0.92,
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


def _render_monthly_combo(catalog: pd.DataFrame, wide: pd.DataFrame) -> None:
    combos = _combo_definitions()
    selected_name = st.selectbox("组合", list(combos), key="combo_spread_name")
    combo = combos[selected_name]
    formula = str(combo.get("formula", "两腿统一单位后相减"))
    st.markdown(
        f"""
        <div class="nap-note">
        <strong>逐月组合口径</strong><br>
        {combo["description"]}<br>
        <strong>本图公式：</strong>{formula}<br>
        <strong>单位规则：</strong>RBOB/HO 为 USD/gal × 42；EBOB 为 USD/mt ÷ {EBOB_BBLS_PER_MT:.2f}；MOPJ/NWE 石脑油为 USD/mt ÷ {NAPHTHA_BBLS_PER_MT:.2f}；LSGO 为 USD/mt ÷ {GASOIL_BBLS_PER_MT:.2f}。
        </div>
        """,
        unsafe_allow_html=True,
    )
    curve, series_by_month = _monthly_combo_frame(wide, catalog, combo)
    if curve.empty or curve["价差"].dropna().empty:
        st.warning("当前组合没有足够数据，请确认 workbook 已刷新且相关 M1-M12 序列存在。")
        _safe_dataframe(curve, use_container_width=True, hide_index=True)
        return
    curve = curve.copy()
    curve["month_num"] = curve["合约"].map(_contract_number)
    curve = curve.sort_values("month_num")

    metric_cols = st.columns(4)
    front_label = "1月" if "1月" in set(curve["合约"].astype(str)) else "M1"
    latest_front = curve[curve["合约"].astype(str) == front_label]["价差"].dropna()
    metric_cols[0].metric(f"{front_label} 最新价差", _fmt(latest_front.iloc[-1] if not latest_front.empty else np.nan))
    metric_cols[1].metric("最强月份", str(curve.dropna(subset=["价差"]).sort_values("价差", ascending=False).iloc[0]["合约"]))
    metric_cols[2].metric("最弱月份", str(curve.dropna(subset=["价差"]).sort_values("价差", ascending=True).iloc[0]["合约"]))
    metric_cols[3].metric("可用月份", f"{curve['价差'].notna().sum()}/12")

    left, right = st.columns([1.35, 1])
    with left:
        fig_curve = px.line(curve, x="合约", y="价差", markers=True, title=f"{selected_name} 当前逐月价差", template=_plot_template())
        fig_curve.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig_curve.update_yaxes(title=str(combo.get("unit", "价差")))
        _apply_fig_layout(fig_curve, f"{selected_name} 当前逐月价差")
        st.plotly_chart(fig_curve, use_container_width=True)
    with right:
        table = curve[["合约", "价差", "1D", "5D", "20D", "最新日期"]].copy()
        _safe_dataframe(table, use_container_width=True, hide_index=True)

    month_options = [month for month in CONTRACT_MONTHS if month in series_by_month]
    if month_options:
        selected_month = st.selectbox("历史走势月份", month_options, key="combo_spread_month")
        spread = series_by_month[selected_month]
        fig_hist = px.line(spread.rename(f"{selected_month} 价差"), title=f"{selected_name} {selected_month} 历史走势", template=_plot_template())
        fig_hist.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig_hist.update_yaxes(title=str(combo.get("unit", "价差")))
        _apply_fig_layout(fig_hist, f"{selected_name} {selected_month} 历史走势")
        st.plotly_chart(fig_hist, use_container_width=True)
        wide_combo = pd.DataFrame(series_by_month).sort_index()
        _download_csv("下载逐月组合价差 CSV", wide_combo, "nap_monthly_combo_spreads.csv", "download_combo_spreads")
        _render_combo_m1_seasonality(selected_name, series_by_month, str(combo.get("unit", "价差")))


def _render_combo_m1_seasonality(selected_name: str, series_by_month: dict[str, pd.Series], unit: str) -> None:
    front_key = "1月" if "1月" in series_by_month else "M1"
    spread = series_by_month.get(front_key)
    if spread is None or spread.dropna().empty:
        return

    st.markdown(f"#### {front_key} 季节图")
    controls = st.columns([0.7, 0.7, 1.8])
    years = controls[0].selectbox("历史窗口", [5, 10], index=1, format_func=lambda value: f"过去 {value} 年", key="combo_m1_season_years")
    remove_leap = controls[1].checkbox("剔除 2月29日", value=True, key="combo_m1_season_remove_feb29")

    seasonal_series = spread.dropna()
    if remove_leap:
        seasonal_series = remove_feb29(seasonal_series.to_frame("value"))["value"]
    matrix = seasonal_matrix(seasonal_series, years=int(years))
    if matrix.empty:
        st.info(f"{front_key} 价差暂时没有足够历史数据生成季节图。")
        return

    x_axis = pd.to_datetime("2001-" + matrix.index.astype(str), errors="coerce")
    latest_year = int(max(matrix.columns))
    palette = _seasonal_year_palette()
    fig = go.Figure()
    for idx, year in enumerate(sorted(matrix.columns)):
        is_latest = int(year) == latest_year
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=matrix[year],
                mode="lines",
                name=str(year),
                line=dict(color=palette[idx % len(palette)], width=3.0 if is_latest else 1.8),
                opacity=1.0 if is_latest else 0.9,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_xaxes(title="月份", tickformat="%m月", dtick="M1")
    fig.update_yaxes(title=unit)
    _apply_fig_layout(fig, f"{selected_name} {front_key} 季节图")
    st.plotly_chart(fig, use_container_width=True)
    _download_csv(f"下载 {front_key} 季节图 CSV", matrix, "nap_combo_front_seasonality.csv", "download_combo_m1_seasonality")


def _render_mopj_driver_chart(catalog: pd.DataFrame, wide: pd.DataFrame) -> None:
    st.markdown("#### MOPJ 升贴水 / 裂解 / Brent 对比")
    st.caption("若 workbook 中没有 Dated Brent 序列，默认用 Brent M1 作为原油价格代理；三个序列默认用 250D z-score 放在同一坐标上比较节奏。")
    selectors = st.columns(3)
    with selectors[0]:
        premium_id = _series_selector(catalog, "mopj_driver_premium", default_query="新加坡贴水", label="MOPJ 升贴水")
    with selectors[1]:
        crack_id = _series_selector(catalog, "mopj_driver_crack", default_query="MOPJ裂差", label="MOPJ Crk")
    with selectors[2]:
        brent_id = _series_selector(catalog, "mopj_driver_brent", default_query="Brent Swap连1", label="Dated Brent / Brent")
    if not premium_id or not crack_id or not brent_id:
        return
    labels = display_lookup(catalog)
    frame = pd.concat(
        [
            _series_from_wide(wide, premium_id).rename("MOPJ升贴水"),
            _series_from_wide(wide, crack_id).rename("MOPJ裂解"),
            _series_from_wide(wide, brent_id).rename("Dated Brent/Brent"),
        ],
        axis=1,
    ).dropna(how="all")
    if frame.empty:
        st.warning("三条序列没有可用数据。")
        return
    mode = st.radio("对比方式", ["250D z-score", "原始值"], horizontal=True, key="mopj_driver_mode")
    if mode == "250D z-score":
        plot_frame = frame.apply(lambda col: (col - col.rolling(250, min_periods=60).mean()) / col.rolling(250, min_periods=60).std())
        y_title = "Z-score"
    else:
        plot_frame = frame
        y_title = "原始值"
    fig = px.line(plot_frame.tail(750), title="MOPJ 升贴水 / 裂解 / Brent 对比", template=_plot_template())
    fig.update_yaxes(title=y_title)
    _apply_fig_layout(fig, "MOPJ 升贴水 / 裂解 / Brent 对比")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"当前选择：{labels.get(premium_id, premium_id)}；{labels.get(crack_id, crack_id)}；{labels.get(brent_id, brent_id)}")
    _download_csv("下载 MOPJ 对比 CSV", frame, "nap_mopj_driver_frame.csv", "download_mopj_driver")


def _render_time_spread_lab(catalog: pd.DataFrame, wide: pd.DataFrame) -> None:
    st.markdown("#### 月差分析")
    groups = available_curve_groups(catalog)
    if groups.empty:
        st.info("没有可用于月差分析的 M1-Mn 曲线。")
        return
    groups = groups[groups["count"] >= 2].copy()
    cols = st.columns(3)
    sector_options = _sector_order(sorted(groups["sector"].dropna().astype(str).unique()))
    sector_map = {_sector_label(value): value for value in sector_options}
    sector_label = cols[0].selectbox("月差板块", list(sector_map), key="ts_sector")
    sector = sector_map[sector_label]
    sector_groups = groups[groups["sector"] == sector]
    product = cols[1].selectbox("月差品种", sorted(sector_groups["product"].dropna().astype(str).unique()), key="ts_product")
    product_groups = sector_groups[sector_groups["product"] == product]
    region = cols[2].selectbox("月差地区", sorted(product_groups["region"].dropna().astype(str).unique()), key="ts_region")

    group_catalog = catalog[
        (catalog["sector"] == sector)
        & (catalog["product"] == product)
        & (catalog["region"] == region)
    ].copy()
    is_calendar = "term_type" in group_catalog.columns and group_catalog["term_type"].astype(str).eq("calendar").all()
    if is_calendar:
        group_catalog["month_num"] = pd.to_numeric(group_catalog["calendar_month"], errors="coerce")
        group_catalog = group_catalog[group_catalog["month_num"].between(1, 12, inclusive="both")].copy()
        group_catalog["month_label"] = group_catalog["month_num"].astype(int).map(lambda month: f"{month}月")
    else:
        group_catalog = group_catalog[group_catalog["contract_month"].astype(str).str.match(r"^M\d+$", na=False)].copy()
        group_catalog["month_num"] = group_catalog["contract_month"].map(_contract_number)
        group_catalog["month_label"] = group_catalog["contract_month"]
    group_catalog = group_catalog.sort_values("month_num")
    month_ids = dict(zip(group_catalog["month_label"], group_catalog["series_id"]))
    pairs = [("1月", "2月"), ("1月", "3月"), ("1月", "6月"), ("2月", "3月")] if is_calendar else [("M1", "M2"), ("M1", "M3"), ("M1", "M6"), ("M2", "M3")]
    spreads: dict[str, pd.Series] = {}
    for front, back in pairs:
        series = _spread_series(wide, month_ids.get(front), month_ids.get(back))
        if not series.empty:
            spreads[f"{front}-{back}"] = series
    if not spreads:
        st.warning("当前曲线缺少可计算的月差组合。")
        return
    spread_frame = pd.DataFrame(spreads).sort_index()
    latest = spread_frame.dropna(how="all").tail(1).T.reset_index()
    latest.columns = ["月差", "最新"]
    fig = px.line(spread_frame.tail(750), title=f"{sector_label} / {product} / {region} 月差走势", template=_plot_template())
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_yaxes(title="近月 - 远月")
    _apply_fig_layout(fig, f"{sector_label} / {product} / {region} 月差走势")
    st.plotly_chart(fig, use_container_width=True)
    _safe_dataframe(latest, hide_index=True, use_container_width=True)
    _download_csv("下载月差分析 CSV", spread_frame, "nap_time_spreads.csv", "download_time_spreads")


def render_combo_spreads(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    st.markdown(
        """
        <div class="nap-note">
        <strong>组合价差</strong><br>
        本页把汽油-石脑油的逐月组合、MOPJ 驱动三线图和标准 M1/M2 月差放在一起。所有派生组合只在同日两腿都有报价时计算。
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_combo, tab_driver, tab_time = st.tabs(["汽油-石脑油逐月组合", "MOPJ 驱动对比", "月差分析"])
    with tab_combo:
        _render_monthly_combo(catalog, wide)
    with tab_driver:
        _render_mopj_driver_chart(catalog, wide)
    with tab_time:
        _render_time_spread_lab(catalog, wide)


def _weekly_panel_from_selector(
    catalog: pd.DataFrame,
    wide: pd.DataFrame,
    *,
    title: str,
    selector: dict[str, object],
) -> dict[str, object]:
    series_id = _find_series_id(catalog, **selector)
    if not series_id:
        return {"title": f"{title}（缺失）", "series": pd.Series(dtype=float)}
    return {"title": title, "series": _series_from_wide(wide, series_id)}


def _weekly_freight_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("原油轮运费 / 美国湾-欧洲 | TD-CRP-MLF", {"sector": "Freight", "ric_prefix": "TD-CRP-MLF"}),
        ("原油轮运费 / 地中海-新加坡 | TD-LPP-SIN", {"sector": "Freight", "ric_prefix": "TD-LPP-SIN"}),
        ("成品油轮运费 / 新加坡-中国 | TC-SIN1-NGB", {"sector": "Freight", "ric_prefix": "TC-SIN1-NGB"}),
        ("原油轮运费 / 中东-中国 | TD-RTA-NGB", {"sector": "Freight", "ric_prefix": "TD-RTA-NGB"}),
    ]
    return [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]


def _weekly_crack_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("NWE石脑油裂解 M1 季节性走势", {"sector": "Cracks", "product": "NWE石脑油裂解", "contract_month": "M1"}),
        ("MOPJ裂解 M1 季节性走势", {"sector": "Cracks", "product": "MOPJ裂解", "contract_month": "M1"}),
        ("NWE石脑油裂解 M2 季节性走势", {"sector": "Cracks", "product": "NWE石脑油裂解", "contract_month": "M2"}),
        ("MOPJ裂解 M2 季节性走势", {"sector": "Cracks", "product": "MOPJ裂解", "contract_month": "M2"}),
    ]
    return [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]


def _weekly_combo_figure(
    selected_name: str,
    series_by_month: dict[str, pd.Series],
    curve: pd.DataFrame,
    *,
    unit: str,
    front_key: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{selected_name} {front_key} 季节图", f"{selected_name} 当前逐月价差"],
        horizontal_spacing=0.11,
        column_widths=[0.62, 0.38],
    )
    _add_weekly_seasonality_panel(
        fig,
        series_by_month.get(front_key, pd.Series(dtype=float)),
        row=1,
        col=1,
        years=5,
        show_legend=True,
        y_title=unit,
        zero_line=True,
    )
    curve_plot = curve.dropna(subset=["价差"]).copy()
    if not curve_plot.empty:
        curve_plot["month_num"] = curve_plot["合约"].map(_contract_number)
        curve_plot = curve_plot.sort_values("month_num")
        fig.add_trace(
            go.Scatter(
                x=curve_plot["合约"],
                y=curve_plot["价差"],
                mode="lines+markers",
                name="当前逐月价差",
                showlegend=False,
                line=dict(color="#6070ff", width=2.6),
                marker=dict(size=7),
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#8c9aa3", line_width=1, row=1, col=2)
    fig.update_xaxes(title_text="月份", tickformat="%m月", dtick="M1", showgrid=True, gridcolor="#e5ebef", row=1, col=1)
    fig.update_xaxes(title_text="合约", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=""),
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=54, r=36, t=92, b=54),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12)),
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, sans-serif", size=12, color="#27323a"),
    )
    fig.update_annotations(font=dict(size=15, color="#1f2a33"))
    return fig


def _weekly_combo_short_name(name: str) -> str:
    if "EBOB" in name or "欧洲" in name:
        return "EBOB - NWE石脑油（折USD/bbl）"
    if "新加坡" in name or "MOPJ" in name:
        return "新加坡92汽油纸货 - MOPJ（折USD/bbl）"
    return name


def _front_month_key(series_by_month: dict[str, pd.Series], mode: str) -> str:
    preferred = "1月" if mode == "calendar" else "M1"
    if preferred in series_by_month:
        return preferred
    fallback = "M1" if "M1" in series_by_month else "1月"
    if fallback in series_by_month:
        return fallback
    return next(iter(series_by_month), "")


def _add_weekly_curve_panel(
    fig: go.Figure,
    curve: pd.DataFrame,
    *,
    row: int,
    col: int,
    unit: str,
) -> None:
    curve_plot = curve.dropna(subset=["价差"]).copy()
    if curve_plot.empty:
        fig.add_annotation(
            text="无可用数据",
            x=0.5,
            y=0.5,
            xref=f"x{(row - 1) * 2 + col} domain",
            yref=f"y{(row - 1) * 2 + col} domain",
            showarrow=False,
            font=dict(color="#7b8790", size=13),
        )
        return
    curve_plot["month_num"] = curve_plot["合约"].map(_contract_number)
    curve_plot = curve_plot.sort_values("month_num")
    fig.add_trace(
        go.Scatter(
            x=curve_plot["合约"],
            y=curve_plot["价差"],
            mode="lines+markers",
            name="当前逐月价差",
            showlegend=False,
            line=dict(color="#6070ff", width=2.6),
            marker=dict(size=7),
        ),
        row=row,
        col=col,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8c9aa3", line_width=1, row=row, col=col)
    fig.update_xaxes(title_text="合约", showgrid=False, row=row, col=col)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=row, col=col)
    low = float(min(curve_plot["价差"].min(), 0.0))
    high = float(max(curve_plot["价差"].max(), 0.0))
    if high > low:
        pad = (high - low) * 0.08
        fig.update_yaxes(range=[low - pad, high + pad], row=row, col=col)


def _weekly_combo_quad_figure(items: list[dict[str, object]], *, unit: str) -> go.Figure:
    slots = items[:2]
    while len(slots) < 2:
        slots.append({"short_name": "", "front_key": "", "series_by_month": {}, "curve": pd.DataFrame()})
    subplot_titles: list[str] = []
    for item in slots:
        short_name = str(item.get("short_name", ""))
        front_key = str(item.get("front_key", ""))
        subplot_titles.extend([f"{short_name} {front_key} 季节图", f"{short_name} 当前逐月价差"])
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.105,
        vertical_spacing=0.16,
        column_widths=[0.6, 0.4],
    )
    for idx, item in enumerate(slots):
        row = idx + 1
        series_by_month = item.get("series_by_month")
        if not isinstance(series_by_month, dict):
            series_by_month = {}
        front_key = str(item.get("front_key", ""))
        curve = item.get("curve")
        _add_weekly_seasonality_panel(
            fig,
            series_by_month.get(front_key, pd.Series(dtype=float)),
            row=row,
            col=1,
            years=5,
            show_legend=idx == 0,
            y_title=unit,
            zero_line=True,
        )
        _add_weekly_curve_panel(
            fig,
            curve if isinstance(curve, pd.DataFrame) else pd.DataFrame(),
            row=row,
            col=2,
            unit=unit,
        )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=""),
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=54, r=36, t=92, b=54),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12)),
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, sans-serif", size=12, color="#27323a"),
    )
    fig.update_annotations(font=dict(size=15, color="#1f2a33"))
    return fig


def _render_weekly_freight(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    fig = _weekly_seasonality_figure(
        _weekly_freight_panels(catalog, wide),
        title="运费路线季节性走势",
        years=5,
        y_title="USD/bbl",
        zero_line=False,
    )
    _weekly_chart(fig, "weekly_freight_seasonality", "weekly_freight_png")


def _render_weekly_cracks(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    wide = long_to_wide(df, normalized=True)
    fig = _weekly_seasonality_figure(
        _weekly_crack_panels(catalog, wide),
        title="石脑油裂解价差季节性走势",
        years=5,
        y_title="USD/bbl",
        zero_line=True,
    )
    _weekly_chart(fig, "weekly_naphtha_cracks", "weekly_cracks_png")


def _render_weekly_combo(df: pd.DataFrame) -> None:
    mode_label = st.radio("组合口径", list(TERM_MODE_OPTIONS), horizontal=True, key="weekly_combo_mode")
    combo_mode = TERM_MODE_OPTIONS[mode_label]
    st.caption(
        f"单位统一：新加坡92为 USD/bbl；MOPJ/NWE 石脑油为 USD/mt ÷ {NAPHTHA_BBLS_PER_MT:.2f}；"
        f"EBOB 为 USD/mt ÷ {EBOB_BBLS_PER_MT:.2f}。周报四宫格分别使用 "
        "Singapore 92 - MOPJ/8.90 和 EBOB/8.33 - NWE/8.90。"
    )
    combo_df = _filter_term_view(df, combo_mode, CALENDAR_MONTHS)
    catalog = catalog_with_labels(combo_df)
    wide = long_to_wide(combo_df, normalized=True)
    combos = _combo_definitions()
    items: list[dict[str, object]] = []
    csv_frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name, combo in combos.items():
        curve, series_by_month = _monthly_combo_frame(wide, catalog, combo)
        short_name = _weekly_combo_short_name(name)
        if curve.empty or not series_by_month:
            missing.append(short_name)
            continue
        front_key = _front_month_key(series_by_month, combo_mode)
        items.append(
            {
                "short_name": short_name,
                "front_key": front_key,
                "series_by_month": series_by_month,
                "curve": curve,
            }
        )
        csv_frames[short_name] = pd.DataFrame(series_by_month).sort_index()
    if missing:
        st.warning("以下组合缺少可出图数据：" + "；".join(missing))
    if not items:
        st.warning("当前组合缺少可出图的数据。")
        return
    fig = _weekly_combo_quad_figure(items, unit="USD/bbl")
    _weekly_chart(fig, "weekly_gasoline_naphtha_spread", "weekly_combo_png")
    if csv_frames:
        _download_csv("下载组合数据 CSV", pd.concat(csv_frames, axis=1), "weekly_gasoline_naphtha_spread.csv", "weekly_combo_csv")


def render_weekly_report(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="nap-report-strip">
          <div>
            <strong>周报出图</strong>
            <em>固定白底 16:9 版式，适合直接下载 PNG 后放入 PPT；图右上角相机按钮也可单独导出。</em>
          </div>
          <div class="nap-report-meta">
            <span>1600×900</span>
            <span>白底</span>
            <span>周报模板</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    continuous_df = _filter_term_view(df, "continuous", CALENDAR_MONTHS)
    tab_freight, tab_cracks, tab_combo = st.tabs(["运费季节性四宫格", "石脑油裂解四宫格", "汽油-石脑油价差"])
    with tab_freight:
        _render_weekly_freight(continuous_df)
    with tab_cracks:
        _render_weekly_cracks(continuous_df)
    with tab_combo:
        _render_weekly_combo(df)


def render_forward_curve(df: pd.DataFrame) -> None:
    catalog = catalog_with_labels(df)
    groups = available_curve_groups(catalog)
    if groups.empty:
        st.info("没有识别到远期曲线。")
        return
    st.markdown(
        """
        <div class="nap-note">
        <strong>远期曲线说明</strong><br>
        这里按同一板块、同一品种、同一地区的 M1-Mn 连续合约组装曲线。当前曲线与 1周前、1个月前、3个月前对比，
        用来观察月间结构、升贴水变化以及近月是否处于 backwardation 或 contango。
        </div>
        """,
        unsafe_allow_html=True,
    )
    groups = groups.copy()
    sector_options = _sector_order(sorted(groups["sector"].dropna().astype(str).unique()))
    cols = st.columns(3)
    sector_label_map = {_sector_label(value): value for value in sector_options}
    selected_sector_label = cols[0].selectbox("板块", list(sector_label_map), key="curve_sector")
    sector = sector_label_map[selected_sector_label]
    sector_groups = groups[groups["sector"] == sector]

    product_options = sorted(sector_groups["product"].dropna().astype(str).unique())
    product = cols[1].selectbox("品种", product_options, key="curve_product")
    product_groups = sector_groups[sector_groups["product"] == product]

    region_options = sorted(product_groups["region"].dropna().astype(str).unique())
    region = cols[2].selectbox("地区", region_options, key="curve_region")
    selected_count = int(product_groups[product_groups["region"] == region]["count"].iloc[0])
    selected = f"{selected_sector_label} / {product} / {region} ({selected_count})"
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
    st.markdown(
        """
        <div class="nap-note">
        <strong>套利计算口径</strong><br>
        图中序列按 <strong>终点价格 - 起点价格 - 运费 × 运费倍数</strong> 计算，并且只保留三条序列同日都有报价的日期。
        结果大于 0 表示终点市场价格相对“起点价格 + 运费”更强；结果小于 0 表示把货运到终点后的经济性偏弱。
        运费倍数用于处理不同报价口径或需要折算的路线，默认为 1。
        </div>
        """,
        unsafe_allow_html=True,
    )
    price_cols = st.columns(2)
    with price_cols[0]:
        origin_id = _series_selector(catalog, "arb_origin", default_query="Jet NEW M1", label="起点价格", allow_all_sector=True)
    with price_cols[1]:
        dest_id = _series_selector(catalog, "arb_dest", default_query="Jet Sin M1", label="终点价格", allow_all_sector=True)
    route_cols = st.columns([1, 0.35])
    route_options = list(route_labels)
    route = route_cols[0].selectbox("运费", route_options, index=_default_index(route_options, "TC-LAV-SIN"), key="arb_route")
    factor = route_cols[1].number_input("运费倍数", value=1.0, step=0.25)
    if not origin_id or not dest_id:
        return
    if origin_id == dest_id:
        st.info("请选择不同的起点和终点价格。")
        return
    route_id = route_labels[route]
    labels = display_lookup(catalog)
    aligned = pd.concat(
        [
            _series_from_wide(wide, dest_id).rename("终点价格"),
            _series_from_wide(wide, origin_id).rename("起点价格"),
            _series_from_wide(wide, route_id).rename("运费"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        st.warning("起点、终点和运费三条序列没有重叠日期，请调整选择或检查报价频率。")
        return
    aligned["运费调整套利"] = aligned["终点价格"] - aligned["起点价格"] - aligned["运费"] * float(factor)
    latest = aligned.iloc[-1]
    metric_cols = st.columns(4)
    metric_cols[0].metric("最新套利", _fmt(latest["运费调整套利"]))
    metric_cols[1].metric("终点价格", _fmt(latest["终点价格"]))
    metric_cols[2].metric("起点价格", _fmt(latest["起点价格"]))
    metric_cols[3].metric("运费调整项", _fmt(latest["运费"] * float(factor)))
    st.caption(
        f"当前公式：{labels.get(dest_id, dest_id)} - {labels.get(origin_id, origin_id)} - "
        f"{labels.get(route_id, route_id)} × {_fmt(factor)}"
    )
    fig = px.line(aligned["运费调整套利"].rename("运费调整套利"), title="终点 - 起点 - 运费", template=_plot_template())
    fig.update_yaxes(title="价差")
    _apply_fig_layout(fig, "终点 - 起点 - 运费")
    st.plotly_chart(fig, use_container_width=True)
    _download_csv("下载套利序列 CSV", aligned, "nap_freight_adjusted_arbitrage.csv", "download_arbitrage")


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
                "原始单位": row.get("unit_native"),
                "计算单位": row.get("unit_normalized") or row.get("unit_native"),
                "换算说明": row.get("unit_conversion", ""),
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
        raw_df = _cached_load(
            str(controls["workbook_path"]),
            str(controls["cache_path"]),
            str(controls["catalog_path"]),
            int(controls["refresh_token"]),
            str(controls["workbook_signature"]),
        )
    except Exception as exc:
        st.error(f"加载 NAP workbook 失败: {exc}")
        logger.exception("Failed to load NAP workbook")
        return

    page = str(controls["page"])
    df = _filter_term_view(
        raw_df,
        str(controls.get("term_mode", "continuous")),
        list(controls.get("calendar_months", CALENDAR_MONTHS)),  # type: ignore[arg-type]
    )
    if df.empty:
        st.warning("当前查看模式下没有可用数据，请切换连续月/自然月模式，或重新解析当前 Excel。")
        return

    _render_topbar(df, str(controls["workbook_path"]))
    explanations = _load_yaml(default_explanations_path())
    if page == "market":
        render_market_map(df)
    elif page == "detail":
        render_series_detail(df, explanations)
    elif page == "seasonality":
        render_seasonality(df)
    elif page == "relationship":
        render_relationship_lab(df)
    elif page == "combos":
        render_combo_spreads(df)
    elif page == "weekly":
        render_weekly_report(raw_df)
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
