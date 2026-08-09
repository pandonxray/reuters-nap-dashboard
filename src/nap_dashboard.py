from __future__ import annotations

import logging
import hashlib
import re
import shutil
import sys
import textwrap
import types
from html import escape
from importlib.metadata import PackageNotFoundError, version as package_version
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
        CACHE_SCHEMA_VERSION,
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
        build_core_market_matrix,
        build_forward_curve,
        build_market_snapshot,
        build_spread_series,
        catalog_with_labels,
        curve_contract_catalog,
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
        percentile_of_value,
        realized_volatility,
        risk_contribution,
        var_es_over_window,
        zscore_of_value,
    )
    from .seasonal_engine import calendar_heatmap_frame, monthly_box_frame, remove_feb29, seasonal_matrix, seasonal_percentile_band, seasonal_stats
    from .unit_conversion import (
        DEFAULT_BBL_PER_MT,
        DISPLAY_UNIT_OPTIONS,
        FACTOR_LABELS,
        convert_frame_rows,
        convert_quote_values,
        product_factor_key,
    )
except ImportError:
    from src.nap_adapter import (
        CACHE_SCHEMA_VERSION,
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
        build_core_market_matrix,
        build_forward_curve,
        build_market_snapshot,
        build_spread_series,
        catalog_with_labels,
        curve_contract_catalog,
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
        percentile_of_value,
        realized_volatility,
        risk_contribution,
        var_es_over_window,
        zscore_of_value,
    )
    from src.seasonal_engine import calendar_heatmap_frame, monthly_box_frame, remove_feb29, seasonal_matrix, seasonal_percentile_band, seasonal_stats
    from src.unit_conversion import (
        DEFAULT_BBL_PER_MT,
        DISPLAY_UNIT_OPTIONS,
        FACTOR_LABELS,
        convert_frame_rows,
        convert_quote_values,
        product_factor_key,
    )


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
SEASONAL_HOVER_FORMAT = "%m月%d日"


def _arrow_render_available() -> bool:
    try:
        pyarrow_major = int(package_version("pyarrow").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return False
    numpy_major = int(np.__version__.split(".", 1)[0])
    return not (numpy_major >= 2 and pyarrow_major < 15)


ARROW_RENDER_AVAILABLE = _arrow_render_available()


PAGE_OPTIONS = {
    "行情｜市场地图": "market",
    "行情｜序列详情": "detail",
    "研究｜季节性": "seasonality",
    "研究｜关系实验室": "relationship",
    "研究｜组合价差": "combos",
    "输出｜周报出图": "weekly",
    "结构｜远期曲线": "curve",
    "风险｜波动与风险": "risk",
    "套利｜运费与套利": "freight",
    "资料｜价格词典": "glossary",
    "资料｜数据健康": "health",
}

NAV_GROUPS = {
    "市场": ["行情｜市场地图", "行情｜序列详情", "结构｜远期曲线"],
    "研究": ["研究｜季节性", "研究｜关系实验室", "研究｜组合价差"],
    "风险与套利": ["风险｜波动与风险", "套利｜运费与套利"],
    "输出": ["输出｜周报出图"],
    "资料": ["资料｜价格词典", "资料｜数据健康"],
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
SPREAD_UNIT_OPTIONS = {
    "USD/bbl": "bbl",
    "USD/mt": "mt",
    "原始单位（仅同单位可计算）": "native",
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
        .nap-html-table {{
            max-width: 100%;
            max-height: 680px;
            overflow: auto;
            border: 1px solid var(--nap-line);
            border-radius: 8px;
            background: var(--nap-panel);
        }}
        .nap-html-table table {{
            width: 100%;
            border-collapse: collapse;
            color: var(--nap-text);
            font-size: 0.79rem;
        }}
        .nap-html-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: var(--nap-panel-soft);
            color: var(--nap-muted);
            text-align: left;
        }}
        .nap-html-table th, .nap-html-table td {{
            padding: 0.42rem 0.55rem;
            border-bottom: 1px solid var(--nap-line);
            white-space: nowrap;
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
    if not ARROW_RENDER_AVAILABLE:
        view = df.reset_index(drop=True) if hide_index else df
        st.markdown(f'<div class="nap-html-table">{view.to_html(index=not hide_index, escape=True)}</div>', unsafe_allow_html=True)
        return
    try:
        st.dataframe(df, width="stretch" if use_container_width else "content", hide_index=hide_index)
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
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|{CACHE_SCHEMA_VERSION}"


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
    frame = load_nap_timeseries(workbook_path, cache_path=cache_path, catalog_path=catalog_path, refresh=refresh_token > 0)
    return _optimize_frame_memory(frame)


def _optimize_frame_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Compact repeated workbook metadata after all catalog overrides are applied."""
    if df.empty:
        return df
    out = df.copy(deep=False)
    string_columns = [
        "series_id",
        "display_name",
        "sheet",
        "sector",
        "product",
        "region",
        "contract_month",
        "term_type",
        "calendar_month",
        "unit_native",
        "unit_normalized",
        "unit_conversion",
        "unit_source",
        "ric",
        "source",
        "name_native",
        "short_name",
    ]
    for column in string_columns:
        if column not in out.columns:
            continue
        out[column] = out[column].astype("category")
    return out


def _view_cache_key(df: pd.DataFrame, suffix: str) -> str:
    base = str(df.attrs.get("nap_view_key", ""))
    if not base:
        latest = pd.to_datetime(df.get("date"), errors="coerce").max() if "date" in df.columns else pd.NaT
        base = f"{len(df)}|{df['series_id'].nunique() if 'series_id' in df.columns else 0}|{latest}"
    return f"{base}|{suffix}"


def _cached_view(df: pd.DataFrame, name: str, builder):
    cache = st.session_state.setdefault("nap_derived_views", {})
    key = _view_cache_key(df, name)
    if key not in cache:
        cache[key] = builder()
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
    return cache[key]


def _view_catalog(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(df, "catalog", lambda: catalog_with_labels(df))


def _view_wide(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(df, "wide_normalized", lambda: long_to_wide(df, normalized=True))


def _view_wide_raw(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(df, "wide_raw", lambda: long_to_wide(df, normalized=False))


def _view_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(
        df,
        "market_snapshot",
        lambda: build_market_snapshot(df, wide=_view_wide(df), catalog=_view_catalog(df)),
    )


def _view_market_snapshot_raw(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(
        df,
        "market_snapshot_raw",
        lambda: build_market_snapshot(df, wide=_view_wide_raw(df), catalog=_view_catalog(df)),
    )


def _view_core_market_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(
        df,
        "core_market_matrix",
        lambda: build_core_market_matrix(_view_catalog(df), _view_wide(df)),
    )


def _view_core_market_matrix_raw(df: pd.DataFrame) -> pd.DataFrame:
    return _cached_view(
        df,
        "core_market_matrix_raw",
        lambda: build_core_market_matrix(_view_catalog(df), _view_wide_raw(df)),
    )


def _append_continuous_crude(df: pd.DataFrame, raw_df: pd.DataFrame, term_mode: str) -> pd.DataFrame:
    """Keep crude selectable when calendar-month derivatives do not exist."""
    if term_mode != "calendar" or raw_df.empty:
        return df
    term = raw_df.get("term_type", pd.Series("continuous", index=raw_df.index)).astype("string").fillna("continuous")
    sector = raw_df.get("sector", pd.Series("", index=raw_df.index)).astype("string").fillna("")
    crude = raw_df[term.ne("calendar") & sector.eq("Crude")]
    if crude.empty:
        return df
    out = pd.concat([df, crude], ignore_index=True)
    out.attrs.update(df.attrs)
    out.attrs["nap_crude_fallback"] = True
    out.attrs["nap_view_key"] = f"{df.attrs.get('nap_view_key', '')}|with-continuous-crude"
    return out


def _filter_term_view(df: pd.DataFrame, term_mode: str, calendar_months: list[int] | None = None) -> pd.DataFrame:
    if df.empty or "term_type" not in df.columns:
        return df
    if term_mode == "calendar":
        out = df[df["term_type"].astype("string").fillna("continuous").eq("calendar")]
        if calendar_months and "calendar_month" in out.columns:
            selected = {int(month) for month in calendar_months}
            months = pd.to_numeric(out["calendar_month"], errors="coerce")
            out = out[months.isin(selected)]
        return out
    return df[df["term_type"].astype("string").fillna("continuous").ne("calendar")]


def _sidebar() -> dict[str, object]:
    st.sidebar.markdown("### Reuters NAP 交易看板")
    if not ARROW_RENDER_AVAILABLE:
        st.sidebar.error("当前 Python 环境的 NumPy / PyArrow 不兼容。请关闭旧实例后使用项目启动脚本重新打开。")
    pending_page = st.session_state.pop("nap_pending_page", None)
    if pending_page in PAGE_OPTIONS:
        st.session_state["nap_page_label"] = pending_page
        pending_group = next((group for group, pages in NAV_GROUPS.items() if pending_page in pages), "市场")
        st.session_state["nap_nav_group"] = pending_group
    current_page = st.session_state.get("nap_page_label", "行情｜市场地图")
    default_group = next((group for group, pages in NAV_GROUPS.items() if current_page in pages), "市场")
    if st.session_state.get("nap_nav_group") not in NAV_GROUPS:
        st.session_state["nap_nav_group"] = default_group
    nav_group = st.sidebar.selectbox("工作区", list(NAV_GROUPS), key="nap_nav_group")
    page_options = NAV_GROUPS[nav_group]
    if st.session_state.get("nap_page_label") not in page_options:
        st.session_state["nap_page_label"] = page_options[0]
    page = st.sidebar.radio(
        "页面",
        page_options,
        label_visibility="collapsed",
        key="nap_page_label",
    )
    theme = st.sidebar.radio("主题", ["浅色", "深色"], horizontal=True)
    st.session_state["nap_theme"] = theme
    unit_label = st.sidebar.selectbox(
        "报价单位",
        list(DISPLAY_UNIT_OPTIONS),
        index=0,
        help="地区默认：美国物理货显示 USD/bbl，其他地区物理货显示 USD/mt；裂解、利润、运费和 LNG 保留原始业务单位。",
    )
    display_unit_mode = DISPLAY_UNIT_OPTIONS[unit_label]
    unit_factors = dict(DEFAULT_BBL_PER_MT)
    with st.sidebar.expander("桶吨换算参数", expanded=False):
        st.caption("系数为每公吨对应桶数，只影响展示，不改变价差和风险计算底稿。")
        for factor_key, default_value in DEFAULT_BBL_PER_MT.items():
            unit_factors[factor_key] = st.number_input(
                f"{FACTOR_LABELS[factor_key]}（bbl/mt）",
                min_value=0.01,
                max_value=30.0,
                value=float(default_value),
                step=0.01,
                key=f"unit_factor_{factor_key}",
            )
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
    with st.sidebar.expander("数据与缓存", expanded=False):
        uploaded = st.file_uploader(
            "拖入 NAP Excel",
            type=["xlsx"],
            help="可拖入 Nap.xlsx 或 Nap_calendar_month_ultralight_formula.xlsx；系统会按文件内容生成专属缓存，不会复用旧 workbook 的缓存。",
        )
        uploaded_path, uploaded_digest = _persist_uploaded_workbook(uploaded)
        workbook_input = st.text_input("或输入 NAP Excel 路径", value=str(DEFAULT_NAP_WORKBOOK))
        workbook_path = uploaded_path or workbook_input
        signature = _workbook_signature(workbook_path)
        cache_path = _signature_cache_path(default_cache_path(), signature)
        _bootstrap_signature_cache(cache_path, workbook_path)
        catalog_path = st.text_input("序列目录路径", value=str(default_catalog_path()))
        if "nap_refresh_token" not in st.session_state:
            st.session_state["nap_refresh_token"] = 0
        if st.button("重新解析当前 Excel", width="stretch"):
            st.session_state["nap_refresh_token"] += 1
            st.session_state.pop("nap_derived_views", None)
            _cached_load.clear()
        source_label = "拖拽上传" if uploaded_path else "路径读取"
        st.caption(
            f"数据来源：{source_label}。当前文件按路径、大小和修改时间生成独立缓存；Excel 更新后会自动重新计算。"
        )
        st.caption(f"当前文件：{workbook_path}")
        st.caption(f"缓存文件：{cache_path.name}")
    return {
        "page": PAGE_OPTIONS[page],
        "theme": theme,
        "display_unit_mode": display_unit_mode,
        "unit_factors": unit_factors,
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
    today = pd.Timestamp.now().normalize()
    if pd.notna(latest) and pd.Timestamp(latest).normalize() > today:
        st.error(
            f"数据日期异常：最新交易日 {latest_text} 晚于本机日期 {today:%Y-%m-%d}。"
            "请检查 Excel 日期列、时区或公式区域后再使用最新信号。"
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
    active_series = st.session_state.get("nap_active_series")
    if active_series in label_map.values():
        selected_index = list(label_map.values()).index(active_series)
    if default_query:
        q = default_query.lower()
        for idx, option in enumerate(labels):
            if q in option.lower():
                selected_index = idx
                break
    selected_label = st.selectbox(label, labels, index=selected_index, key=f"{key}_select")
    selected_series = label_map[selected_label]
    st.session_state["nap_active_series"] = selected_series
    return selected_series


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


def _spread_unit_selector(
    key: str,
    default_mode: str = "bbl",
    label: str = "价差单位",
    *,
    allow_native: bool = True,
) -> tuple[str, str]:
    labels = list(SPREAD_UNIT_OPTIONS) if allow_native else ["USD/bbl", "USD/mt"]
    wanted = "USD/mt" if default_mode in {"mt", "regional"} else "USD/bbl" if default_mode == "bbl" else labels[-1]
    selected = st.selectbox(label, labels, index=labels.index(wanted), key=key)
    return SPREAD_UNIT_OPTIONS[selected], selected.split("（", 1)[0]


def _build_converted_spread(
    wide_raw: pd.DataFrame,
    catalog: pd.DataFrame,
    legs: list[tuple[str, float]],
    unit_mode: str,
    unit_factors: dict[str, float],
) -> tuple[pd.Series, str, list[str]]:
    if wide_raw.empty or catalog.empty or len(legs) < 2:
        return pd.Series(dtype=float), "", []
    metadata = catalog.set_index("series_id").to_dict(orient="index")
    converted_legs: list[pd.Series] = []
    actual_units: list[str] = []
    formulas: list[str] = []
    weights: list[float] = []
    for idx, (series_id, weight) in enumerate(legs):
        if series_id not in wide_raw.columns or series_id not in metadata:
            return pd.Series(dtype=float), "", [f"第 {idx + 1} 条腿没有可用数据。"]
        meta = metadata[series_id]
        values, unit, formula = convert_quote_values(
            wide_raw[series_id].dropna(), meta, unit_mode, unit_factors
        )
        factor_key = product_factor_key(meta)
        barrels_per_mt = float(unit_factors.get(factor_key, DEFAULT_BBL_PER_MT.get(factor_key, 0.0))) if factor_key else 0.0
        if unit_mode == "mt" and unit == "USD/bbl" and barrels_per_mt > 0:
            values = values * barrels_per_mt
            unit = "USD/mt"
            formula = f"USD/bbl × {barrels_per_mt:g} = USD/mt。"
        elif unit_mode == "bbl" and unit == "USD/mt" and barrels_per_mt > 0:
            values = values / barrels_per_mt
            unit = "USD/bbl"
            formula = f"USD/mt ÷ {barrels_per_mt:g} = USD/bbl。"
        converted_legs.append(values.rename(f"leg_{idx}"))
        actual_units.append(unit)
        formulas.append(formula)
        weights.append(float(weight))
    unique_units = {unit for unit in actual_units if unit}
    if len(unique_units) != 1:
        return (
            pd.Series(dtype=float),
            " / ".join(sorted(unique_units)),
            ["各腿换算后的单位不一致，不能直接相加减。请选择 USD/bbl，或调整所选序列。"],
        )
    aligned = pd.concat(converted_legs, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float), next(iter(unique_units), ""), formulas
    spread = sum(aligned.iloc[:, idx] * weights[idx] for idx in range(len(weights)))
    spread.name = "custom_spread"
    return spread, next(iter(unique_units), ""), formulas


def _curve_family_for_series(catalog: pd.DataFrame, series_id: str) -> pd.DataFrame:
    if catalog.empty or series_id not in set(catalog["series_id"].astype(str)):
        return pd.DataFrame()
    meta = catalog[catalog["series_id"].astype(str).eq(str(series_id))].iloc[0]
    groups = available_curve_groups(catalog)
    if groups.empty:
        return pd.DataFrame()
    candidates = groups[
        groups["sector"].astype(str).eq(str(meta.get("sector", "")))
        & groups["product"].astype(str).eq(str(meta.get("product", "")))
        & groups["region"].astype(str).eq(str(meta.get("region", "")))
    ]
    for _, group in candidates.iterrows():
        contracts = curve_contract_catalog(
            catalog,
            str(group["sector"]),
            str(group["product"]),
            str(group["region"]),
            str(group["family_key"]),
        )
        if series_id in set(contracts["series_id"].astype(str)):
            return contracts.sort_values("month_num")
    return pd.DataFrame()


def _custom_structure_frame(
    wide_raw: pd.DataFrame,
    catalog: pd.DataFrame,
    legs: list[tuple[str, float]],
    unit_mode: str,
    unit_factors: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, pd.Series], str]:
    families = [_curve_family_for_series(catalog, series_id) for series_id, _ in legs]
    if not families or any(family.empty for family in families):
        return pd.DataFrame(), {}, "所选序列并非都属于可识别的连续合约曲线。"
    if any(not family["curve_mode"].astype(str).eq("continuous").all() for family in families):
        return pd.DataFrame(), {}, "自然月序列不生成连续合约结构。"
    ordered_months = CONTRACT_MONTHS
    rows: list[dict[str, object]] = []
    series_by_month: dict[str, pd.Series] = {}
    actual_unit = ""
    for month in ordered_months:
        month_legs: list[tuple[str, float]] = []
        for family, (_, weight) in zip(families, legs):
            match = family[family["curve_label"].astype(str).eq(month)]
            if match.empty:
                month_legs = []
                break
            month_legs.append((str(match.iloc[0]["series_id"]), weight))
        if not month_legs:
            rows.append(
                {
                    "合约": month,
                    "价差": np.nan,
                    "1D": np.nan,
                    "5D": np.nan,
                    "20D": np.nan,
                    "最新日期": pd.NaT,
                }
            )
            continue
        spread, unit, _ = _build_converted_spread(wide_raw, catalog, month_legs, unit_mode, unit_factors)
        if spread.empty:
            rows.append(
                {
                    "合约": month,
                    "价差": np.nan,
                    "1D": np.nan,
                    "5D": np.nan,
                    "20D": np.nan,
                    "最新日期": pd.NaT,
                }
            )
            continue
        actual_unit = unit or actual_unit
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
            }
        )
    return pd.DataFrame(rows), series_by_month, actual_unit


def _combo_definitions() -> dict[str, dict[str, object]]:
    return {
        "新加坡92汽油纸货 - MOPJ": {
            "left": {"sector": "Gasoline", "product": "新加坡92汽油", "region": "新加坡"},
            "right": {"sector": "Naphtha", "product": "日本CFR石脑油", "region": "日本/东北亚"},
            "formula_bbl": "Singapore 92 - MOPJ/8.90",
            "formula_mt": "Singapore 92×8.33 - MOPJ",
            "description": "新加坡92RON汽油纸货逐月减 MOPJ CFR Japan 石脑油逐月；两腿先分别换算到所选单位，再做同日相减。",
        },
        "欧洲EBOB汽油纸货 - NWE CIF石脑油": {
            "left": {"sector": "Gasoline", "product": "欧洲EBOB汽油", "region": "欧洲"},
            "right": {"sector": "Naphtha", "product": "NWE CIF石脑油", "region": "西北欧"},
            "formula_bbl": "EBOB/8.33 - NWE/8.90",
            "formula_mt": "EBOB - NWE",
            "description": "欧洲 EBOB 汽油纸货逐月减 Naphtha CIF NWE outright swap 逐月；两腿先分别换算到所选单位，再做同日相减。",
        },
    }


def _monthly_combo_frame(
    wide: pd.DataFrame,
    catalog: pd.DataFrame,
    combo: dict[str, object],
    *,
    unit_mode: str | None = None,
    unit_factors: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
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
        if unit_mode:
            spread, _, _ = _build_converted_spread(
                wide,
                catalog,
                [(left_id, 1.0), (right_id, -1.0)] if left_id and right_id else [],
                unit_mode,
                unit_factors or dict(DEFAULT_BBL_PER_MT),
            )
        else:
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


def _curve_asof(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty:
        return None
    candidates: list[pd.Timestamp] = []
    for column in ("最新日期", "asof"):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            candidates.append(pd.Timestamp(values.max()).normalize())
    if candidates:
        return max(candidates)
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
        return pd.Timestamp(frame.index.max()).normalize()
    return None


def _continuous_month_mapping(asof: pd.Timestamp | None) -> pd.DataFrame:
    base = pd.Timestamp(asof).normalize() if asof is not None and not pd.isna(asof) else None
    rows: list[dict[str, object]] = []
    for month_num in range(1, 13):
        natural_date = base + pd.DateOffset(months=month_num - 1) if base is not None else pd.NaT
        rows.append(
            {
                "month_num": month_num,
                "contract_month": f"M{month_num}",
                "natural_month": natural_date,
                "natural_month_label": natural_date.strftime("%Y-%m") if not pd.isna(natural_date) else "",
                "tick_label": f"M{month_num}<br>{natural_date.month:02d}月" if not pd.isna(natural_date) else f"M{month_num}",
            }
        )
    return pd.DataFrame(rows)


def _complete_contract_curve(
    curve: pd.DataFrame,
    *,
    contract_column: str,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    asof = _curve_asof(curve)
    axis = _continuous_month_mapping(asof)
    if curve.empty:
        return axis, asof
    values = curve.copy()
    values["month_num"] = values[contract_column].map(_contract_number)
    values = values[values["month_num"].between(1, 12, inclusive="both")]
    values = values.sort_values("month_num").drop_duplicates("month_num", keep="last")
    values = values.drop(columns=[contract_column, "contract_month", "natural_month", "natural_month_label", "tick_label"], errors="ignore")
    return axis.merge(values, on="month_num", how="left"), asof


def _apply_continuous_contract_axis(
    fig: go.Figure,
    asof: pd.Timestamp | None,
    *,
    row: int | None = None,
    col: int | None = None,
    title: str = "连续月 / 自然月",
) -> None:
    mapping = _continuous_month_mapping(asof)
    axis_kwargs = {
        "title_text": title,
        "tickmode": "array",
        "tickvals": mapping["month_num"].tolist(),
        "ticktext": mapping["tick_label"].tolist(),
        "range": [0.5, 12.5],
        "showgrid": False,
        "tickfont": dict(size=10),
    }
    if row is None or col is None:
        fig.update_xaxes(**axis_kwargs)
    else:
        fig.update_xaxes(**axis_kwargs, row=row, col=col)


def _contract_month_mapping_caption(asof: pd.Timestamp | None) -> str:
    mapping = _continuous_month_mapping(asof)
    if asof is None:
        return "连续月自然月映射暂缺：当前曲线没有可用的最新报价日期。"
    pairs = " · ".join(
        f"{row.contract_month}={pd.Timestamp(row.natural_month).year}年{pd.Timestamp(row.natural_month).month}月"
        for row in mapping.itertuples(index=False)
    )
    return f"连续月对应自然月（按最新有效报价日 {pd.Timestamp(asof):%Y-%m-%d} 自动换算）：{pairs}"


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
    fig.update_xaxes(
        title_text="月份",
        tickformat="%m月",
        hoverformat=SEASONAL_HOVER_FORMAT,
        dtick="M1",
        showgrid=True,
        gridcolor="#e5ebef",
        row=row,
        col=col,
    )
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
    st.plotly_chart(fig, width="stretch", config=_plotly_config(filename))
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


def _market_contract_view(snapshot: pd.DataFrame, include_other_contracts: bool) -> pd.DataFrame:
    if snapshot.empty or include_other_contracts or "term_type" not in snapshot.columns:
        return snapshot
    term_type = snapshot["term_type"].astype("string").fillna("continuous")
    if term_type.eq("calendar").all():
        return snapshot
    contracts = snapshot["contract_month"].astype("string").fillna("").str.upper().str.strip()
    return snapshot[term_type.eq("continuous") & contracts.isin({"M1", "M2"})]


def _market_matrix_frame(group: pd.DataFrame) -> pd.DataFrame:
    view = group.copy()
    view["板块"] = view["sector"].map(_sector_label)
    view["品种"] = view["product"].astype("string").fillna("未分类")
    view["地区"] = view["region"].astype("string").fillna("未标注")
    calendar = view.get("term_type", pd.Series("continuous", index=view.index)).astype(str).eq("calendar")
    view["合约"] = view["contract_month"].astype("string").fillna("")
    if "calendar_month" in view.columns:
        months = pd.to_numeric(view["calendar_month"], errors="coerce")
        view.loc[calendar & months.notna(), "合约"] = months[calendar & months.notna()].astype(int).astype(str) + "月"
    view["结构"] = view["structure"].map(lambda value: STRUCTURE_CN.get(str(value), str(value)))
    view["单位"] = view["unit"].astype("string").fillna("")
    return view.rename(
        columns={
            "display_name": "序列",
            "latest": "最新",
            "chg_1d": "1D",
            "chg_5d": "5D",
            "chg_20d": "20D",
            "z_60d": "Z60",
            "pct_250d": "P250",
            "vol_20d": "Vol20",
        }
    )[["板块", "品种", "地区", "合约", "序列", "最新", "1D", "5D", "20D", "Z60", "P250", "Vol20", "结构", "单位"]]


def _render_market_matrix(group: pd.DataFrame) -> None:
    table = _market_matrix_frame(group)
    column_config = {
        "最新": st.column_config.NumberColumn(format="%.2f"),
        "1D": st.column_config.NumberColumn(format="%+.2f"),
        "5D": st.column_config.NumberColumn(format="%+.2f"),
        "20D": st.column_config.NumberColumn(format="%+.2f"),
        "Z60": st.column_config.NumberColumn(format="%+.2f"),
        "P250": st.column_config.NumberColumn(format="%.0f"),
        "Vol20": st.column_config.NumberColumn(format="%.1%%"),
    }
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(680, 42 + 35 * len(table)),
        column_config=column_config,
    )


def _core_market_display_frame(matrix: pd.DataFrame, include_far_structure: bool) -> pd.DataFrame:
    view = matrix.copy()
    view["板块"] = view["sector"].map(_sector_label)
    view["品种"] = view["product"].astype("string").fillna("未分类")
    view["地区"] = view["region"].astype("string").fillna("未标注")
    view["报价"] = view["quote"].astype("string").fillna("")
    same_as_product = view["报价"].str.casefold().eq(view["品种"].str.casefold())
    view.loc[same_as_product, "报价"] = ""
    view["曲线"] = view["structure"].map(STRUCTURE_CN).fillna(view["structure"])
    view["数据日"] = pd.to_datetime(view["latest_date"], errors="coerce").dt.strftime("%m-%d")
    view["单位"] = view["unit"].astype("string").fillna("")
    view = view.rename(
        columns={
            "m1": "M1",
            "m2": "M2",
            "m1_m2": "M1-M2",
            "m1_m3": "M1-M3",
            "m1_m6": "M1-M6",
            "m1_chg_1d": "M1 1D",
            "spread_chg_1d": "结构1D",
            "spread_z_60d": "结构Z60",
            "spread_pct_250d": "结构P250",
        }
    )
    columns = ["板块", "品种", "地区", "报价", "M1", "M2", "M1-M2"]
    if include_far_structure:
        columns.extend(["M1-M3", "M1-M6"])
    columns.extend(["M1 1D", "结构1D", "结构Z60", "结构P250", "曲线", "数据日", "单位"])
    return view[columns]


def _render_core_market_matrix(matrix: pd.DataFrame, include_far_structure: bool) -> None:
    display = _core_market_display_frame(matrix, include_far_structure)
    number_columns = ["M1", "M2", "M1-M2", "M1-M3", "M1-M6", "M1 1D", "结构1D", "结构Z60"]
    column_config = {
        column: st.column_config.NumberColumn(format="%+.2f" if column in {"M1 1D", "结构1D", "结构Z60"} else "%.2f")
        for column in number_columns
        if column in display.columns
    }
    column_config["结构P250"] = st.column_config.NumberColumn(format="%.0f")
    event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=min(720, 54 + 35 * len(display)),
        column_config=column_config,
        key="nap_core_market_matrix",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = list(event.selection.rows)
    if selected_rows:
        selected_idx = int(selected_rows[0])
        if 0 <= selected_idx < len(matrix):
            st.session_state["nap_active_series"] = str(matrix.iloc[selected_idx]["m1_series_id"])
            st.session_state["nap_pending_page"] = "行情｜序列详情"
            st.rerun()


def _render_structure_opportunities(matrix: pd.DataFrame) -> None:
    ranked = matrix.dropna(subset=["spread_z_60d"]).copy()
    if ranked.empty:
        return
    ranked["abs_z"] = ranked["spread_z_60d"].abs()
    ranked = ranked.nlargest(10, "abs_z").sort_values("spread_z_60d")
    ranked["label"] = ranked["product"].astype(str) + " · " + ranked["region"].astype(str)
    duplicate_labels = ranked["label"].duplicated(keep=False)
    ranked.loc[duplicate_labels, "label"] += " · " + ranked.loc[duplicate_labels, "quote"].astype(str)
    colors = [ACCENT if value >= 0 else AMBER for value in ranked["spread_z_60d"]]
    fig = go.Figure(
        go.Bar(
            x=ranked["spread_z_60d"],
            y=ranked["label"],
            orientation="h",
            marker_color=colors,
            text=ranked["spread_z_60d"].map(lambda value: f"{value:+.2f}"),
            textposition="outside",
            customdata=np.column_stack([ranked["m1_m2"], ranked["spread_pct_250d"]]),
            hovertemplate="%{y}<br>结构Z60 %{x:+.2f}<br>M1-M2 %{customdata[0]:.2f}<br>P250 %{customdata[1]:.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_xaxes(title="M1-M2 的 60日 Z-score", zeroline=False)
    fig.update_yaxes(title="")
    fig.update_layout(height=max(330, 38 * len(ranked) + 110), showlegend=False)
    _apply_fig_layout(fig, "近端结构偏离")
    st.plotly_chart(fig, width="stretch", config=_plotly_config("nap_structure_opportunities"))


def render_market_map(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    catalog = _view_catalog(df)
    snapshot = convert_frame_rows(
        _view_market_snapshot_raw(df),
        catalog,
        series_column="series_id",
        value_columns=["latest", "chg_1d", "chg_5d", "chg_20d"],
        mode=unit_mode,
        factors=unit_factors,
    )
    if snapshot.empty:
        st.warning("没有加载到 NAP 数据。")
        return
    core_matrix = convert_frame_rows(
        _view_core_market_matrix_raw(df),
        catalog,
        series_column="m1_series_id",
        value_columns=["m1", "m2", "m1_m2", "m1_m3", "m1_m6", "m1_chg_1d", "spread_chg_1d"],
        mode=unit_mode,
        factors=unit_factors,
    )
    st.markdown(
        '<div class="nap-report-strip"><div><strong>近端价格与结构</strong><em>M1、M2 与结构价差使用同日有效报价；点击矩阵行可进入对应 M1 序列详情。</em></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("当前单位规则只转换物理货展示值；裂解、利润、运费和 LNG 保留其业务报价单位，底层结构信号仍按标准化口径计算。")
    core_available = not core_matrix.empty
    view_options = ["核心结构矩阵", "序列明细", "指标卡片"] if core_available else ["序列明细", "指标卡片"]
    controls = st.columns([1.25, 0.95, 0.8, 0.7])
    sector_values = [sector for sector in MARKET_GROUPS if sector in set(snapshot["sector"].astype(str))]
    sector_labels = {_sector_label(sector): sector for sector in sector_values}
    selected_sector_labels = controls[0].multiselect("板块", list(sector_labels), default=list(sector_labels))
    sector_filter = [sector_labels[label] for label in selected_sector_labels]
    search = controls[1].text_input("筛选品种 / 地区 / 报价", "")
    view_mode = controls[2].selectbox("视图", view_options)
    if view_mode == "核心结构矩阵":
        include_far_structure = controls[3].toggle("显示 M1-M3 / M1-M6", value=False)
        filtered_core = core_matrix[core_matrix["sector"].isin(sector_filter)].copy()
        if search:
            q = search.casefold()
            haystack = (
                filtered_core["product"].astype(str)
                + " "
                + filtered_core["region"].astype(str)
                + " "
                + filtered_core["quote"].astype(str)
            ).str.casefold()
            filtered_core = filtered_core[haystack.str.contains(q, regex=False, na=False)]
        if filtered_core.empty:
            st.info("当前筛选下没有可配对的 M1/M2 曲线。")
            return

        latest_date = pd.to_datetime(filtered_core["latest_date"], errors="coerce").max()
        backwardation = int(filtered_core["structure"].eq("backwardation").sum())
        contango = int(filtered_core["structure"].eq("contango").sum())
        extreme_z = pd.to_numeric(filtered_core["spread_z_60d"], errors="coerce").abs().max()
        summary = st.columns(4)
        summary[0].metric("M1/M2 曲线", f"{len(filtered_core):,}")
        summary[1].metric("现货升水 / 远期升水", f"{backwardation} / {contango}")
        summary[2].metric("最极端结构 Z60", _fmt(extreme_z))
        summary[3].metric("共同数据日", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "-")

        _render_core_market_matrix(filtered_core.reset_index(drop=True), include_far_structure)
        _download_csv("下载结构矩阵 CSV", filtered_core, "nap_core_market_matrix.csv", "download_core_market_matrix")
        st.markdown(
            '<div class="nap-section-title"><span>结构机会</span><span>按 M1-M2 的绝对 Z60 排序；青色为正结构，琥珀色为负结构。</span></div>',
            unsafe_allow_html=True,
        )
        _render_structure_opportunities(filtered_core)
        return

    include_other_contracts = controls[3].toggle("显示其他合约", value=False)
    filtered = _market_contract_view(snapshot, include_other_contracts)
    filtered = filtered[filtered["sector"].isin(sector_filter)]
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

    if filtered.empty:
        st.info("当前筛选下没有序列。默认只显示 M1 / M2，可打开“显示其他合约”查看完整曲线或现货序列。")
        return

    gainers = int((pd.to_numeric(filtered["chg_1d"], errors="coerce") > 0).sum())
    decliners = int((pd.to_numeric(filtered["chg_1d"], errors="coerce") < 0).sum())
    latest_date = pd.to_datetime(filtered["latest_date"], errors="coerce").max()
    summary = st.columns(4)
    summary[0].metric("当前序列", f"{len(filtered):,}")
    summary[1].metric("上涨 / 下跌", f"{gainers} / {decliners}")
    summary[2].metric("极端 Z60", f"{pd.to_numeric(filtered['z_60d'], errors='coerce').abs().max():.2f}")
    summary[3].metric("行情日期", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "-")

    _download_csv("下载当前行情 CSV", filtered, "nap_market_snapshot.csv", "download_market_snapshot")
    for sector in MARKET_GROUPS:
        group = filtered[filtered["sector"] == sector]
        if group.empty:
            continue
        group = group.copy()
        group["sort_month"] = pd.to_numeric(group.get("calendar_month", ""), errors="coerce").fillna(
            group["contract_month"].map(_contract_number)
        )
        group = group.sort_values(["product", "region", "sort_month", "contract_month", "display_name"])
        st.markdown(
            f'<div class="nap-section-title"><span>{_sector_label(sector)}</span><span>{len(group)} 条序列</span></div>',
            unsafe_allow_html=True,
        )
        if view_mode == "序列明细":
            _render_market_matrix(group)
        else:
            max_cards = st.slider(
                f"{_sector_label(sector)}最多显示",
                4,
                30,
                12,
                key=f"market_cards_{sector}",
                label_visibility="collapsed",
            )
            cards = "\n".join(_render_market_card(row) for _, row in group.head(max_cards).iterrows())
            st.markdown(f'<div class="nap-card-grid">{cards}</div>', unsafe_allow_html=True)


def render_series_detail(
    df: pd.DataFrame,
    explanations: dict,
    unit_mode: str,
    unit_factors: dict[str, float],
) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide_raw(df)
    if bool(df.attrs.get("nap_crude_fallback")):
        st.info("自然月数据源目前没有原油派生序列，因此原油板块自动补入连续月 M1-Mn；其他板块仍按所选自然月显示。")
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
    meta = catalog.set_index("series_id").loc[series_id]
    raw_series = _series_from_wide(wide, series_id)
    series, display_unit, display_conversion = convert_quote_values(raw_series, meta, unit_mode, unit_factors)
    with middle:
        fig = px.line(series.rename("价格"), title=str(meta["display_name"]))
        fig.update_yaxes(title=f"价格（{display_unit or '未标注'}）")
        _apply_fig_layout(fig, str(meta["display_name"]))
        st.plotly_chart(fig, width="stretch")
        _download_csv("下载当前序列 CSV", series.to_frame(f"value_{display_unit or 'raw'}"), f"{series_id}.csv", "detail_download")
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
              展示单位: {display_unit or "未标注"}<br>
              展示换算: {display_conversion}<br><br>
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


def render_seasonality(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide_raw(df)
    controls = st.columns([1, 0.55, 0.55, 0.65])
    with controls[0]:
        series_id = _series_selector(catalog, "season", label="季节性序列")
    years = controls[1].selectbox("历史窗口", [5, 10], index=0, format_func=lambda x: f"过去 {x} 年")
    remove_leap = controls[2].checkbox("剔除 2月29日", value=True)
    lunar = controls[3].checkbox("农历季节性", value=False, help="主要用于中国相关品种；当前页面保留公历视图。")
    if not series_id:
        return
    meta = catalog.set_index("series_id").loc[series_id]
    raw_series = _series_from_wide(wide, series_id)
    series, display_unit, _ = convert_quote_values(raw_series, meta, unit_mode, unit_factors)
    if remove_leap:
        series = remove_feb29(series.to_frame("value"))["value"]
    if lunar and str(meta.get("region")) != "China":
        st.caption("农历季节性主要保留给中国相关品种；当前选择显示公历季节性。")

    matrix = seasonal_matrix(series, years=years)
    band = seasonal_percentile_band(series, years=years)
    stats = seasonal_stats(series, years=years)
    current_value = float(series.iloc[-1]) if not series.empty else np.nan
    seasonal_deviation = stats.get("seasonal_deviation", np.nan)
    seasonal_percentile = stats.get("seasonal_percentile", np.nan)
    metric_cols = st.columns(4)
    metric_cols[0].metric("最新", _fmt(current_value))
    metric_cols[1].metric("同日历史百分位", f"{_fmt(seasonal_percentile, 0)}%" if pd.notna(seasonal_percentile) else "-")
    metric_cols[2].metric("偏离同日均值", _fmt(seasonal_deviation))
    metric_cols[3].metric("历史窗口", f"{years} 年")
    st.caption(f"展示单位：{display_unit or '未标注'}")
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
            fig.add_trace(go.Scatter(x=band_x, y=band["median"], name="历史中位数", line=dict(color=NEUTRAL, width=1.7, dash="dash")))
        if not matrix.empty:
            latest_year = int(max(matrix.columns))
            historical_color = "rgba(127,140,141,0.34)" if _plot_template() == LIGHT_TEMPLATE else "rgba(147,164,173,0.38)"
            for year in sorted(matrix.columns):
                is_latest = int(year) == latest_year
                is_previous = int(year) == latest_year - 1
                line_color = ACCENT if is_latest else AMBER if is_previous else historical_color
                line_width = 3.2 if is_latest else 2.0 if is_previous else 1.0
                line_dash = "solid" if is_latest else "dash" if is_previous else "solid"
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=matrix[year],
                        mode="lines",
                        name=str(year),
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        opacity=1.0,
                        showlegend=is_latest or is_previous,
                    )
                )
            latest_values = matrix[latest_year].dropna()
            if not latest_values.empty:
                latest_x = pd.to_datetime(f"2001-{latest_values.index[-1]}", errors="coerce")
                fig.add_trace(
                    go.Scatter(
                        x=[latest_x],
                        y=[latest_values.iloc[-1]],
                        mode="markers+text",
                        marker=dict(color=ACCENT, size=9),
                        text=[_fmt(latest_values.iloc[-1])],
                        textposition="top center",
                        name="最新点",
                        showlegend=False,
                    )
                )
        fig.update_xaxes(title="月份", tickformat="%m月", hoverformat=SEASONAL_HOVER_FORMAT, dtick="M1")
        fig.update_yaxes(title="数值")
        _apply_fig_layout(fig, f"{meta['display_name']} 季节性走势")
        st.plotly_chart(fig, width="stretch")
        _download_csv("下载季节性矩阵 CSV", matrix, "nap_seasonal_matrix.csv", "download_seasonal")
    with right:
        box = monthly_box_frame(series, years=years)
        fig_box = px.box(box, x="month", y="value", points=False, title="月度分布箱线图", template=_plot_template())
        fig_box.update_xaxes(title="月份", tickmode="array", tickvals=list(range(1, 13)), ticktext=[f"{m}月" for m in range(1, 13)])
        fig_box.update_yaxes(title="数值")
        _apply_fig_layout(fig_box, "月度分布箱线图")
        st.plotly_chart(fig_box, width="stretch")

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
        st.plotly_chart(fig_heat, width="stretch")


def _render_spread_seasonality(spread: pd.Series, title: str, unit: str, key_prefix: str) -> None:
    controls = st.columns([0.65, 0.65, 1.7])
    years = controls[0].selectbox(
        "历史窗口",
        [5, 10, 15, 20],
        index=2,
        format_func=lambda value: f"过去 {value} 年",
        key=f"{key_prefix}_years",
    )
    remove_leap = controls[1].checkbox("剔除 2月29日", value=True, key=f"{key_prefix}_remove_feb29")
    seasonal = spread.dropna()
    if remove_leap:
        seasonal = remove_feb29(seasonal.to_frame("value"))["value"]
    matrix = seasonal_matrix(seasonal, years=int(years))
    band = seasonal_percentile_band(seasonal, years=max(int(years), 10))
    if matrix.empty:
        st.info("当前价差没有足够历史数据生成季节图。")
        return
    x_axis = pd.to_datetime("2001-" + matrix.index.astype(str), errors="coerce")
    latest_year = int(max(matrix.columns))
    fig = go.Figure()
    if not band.empty:
        band_x = pd.to_datetime("2001-" + band["doy"].astype(str), errors="coerce")
        fig.add_trace(go.Scatter(x=band_x, y=band["upper"], line=dict(width=0), showlegend=False, name="P90"))
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band["lower"],
                fill="tonexty",
                fillcolor="rgba(47,125,140,0.14)",
                line=dict(width=0),
                name="P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=band_x, y=band["median"], name="历史中位数", line=dict(color=NEUTRAL, width=1.7, dash="dash")))
    for year in sorted(matrix.columns):
        year_int = int(year)
        is_latest = year_int == latest_year
        is_previous = year_int == latest_year - 1
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=matrix[year],
                mode="lines",
                name=str(year_int),
                line=dict(
                    color=ACCENT if is_latest else AMBER if is_previous else "rgba(127,140,141,0.30)",
                    width=3.0 if is_latest else 2.0 if is_previous else 1.0,
                ),
                showlegend=is_latest or is_previous,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_xaxes(title="月份", tickformat="%m月", hoverformat=SEASONAL_HOVER_FORMAT, dtick="M1")
    fig.update_yaxes(title=unit or "价差")
    _apply_fig_layout(fig, f"{title} 季节图")
    st.plotly_chart(fig, width="stretch")
    _download_csv("下载季节图 CSV", matrix, "nap_custom_spread_seasonality.csv", f"{key_prefix}_download")


def _custom_spread_title(catalog: pd.DataFrame, legs: list[tuple[str, float]]) -> str:
    metadata = catalog.set_index("series_id").to_dict(orient="index") if not catalog.empty else {}
    parts: list[str] = []
    for idx, (series_id, weight) in enumerate(legs):
        meta = metadata.get(series_id, {})
        product = str(meta.get("product") or meta.get("display_name") or series_id)
        region = str(meta.get("region") or "")
        contract = str(meta.get("contract_month") or "")
        name = " / ".join(part for part in (product, region, contract) if part)
        magnitude = abs(float(weight))
        weighted_name = name if np.isclose(magnitude, 1.0) else f"{magnitude:g}×{name}"
        if idx == 0:
            parts.append(weighted_name if weight >= 0 else f"-{weighted_name}")
        else:
            parts.append((" + " if weight >= 0 else " - ") + weighted_name)
    return "".join(parts) or "自定义价差"


def render_relationship_lab(
    df: pd.DataFrame,
    default_unit_mode: str,
    unit_factors: dict[str, float],
) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide(df)
    wide_raw = _view_wide_raw(df)
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
        st.plotly_chart(fig, width="stretch")
        _download_csv("下载公式序列 CSV", spread.to_frame(selected), "nap_formula_series.csv", "download_registry_formula")
        return

    if mode == "自定义价差":
        unit_cols = st.columns([0.65, 1.5])
        with unit_cols[0]:
            spread_unit_mode, requested_unit = _spread_unit_selector(
                "custom_spread_unit",
                default_unit_mode,
                allow_native=True,
            )
        leg_count = unit_cols[1].slider("腿数", 2, 5, 2)
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
        spread, actual_unit, formulas = _build_converted_spread(
            wide_raw,
            catalog,
            legs,
            spread_unit_mode,
            unit_factors,
        )
        if spread.empty:
            st.warning("自定义价差无法计算。" + (" ".join(formulas) if formulas else "请确认各腿有重叠数据且单位兼容。"))
            return
        spread_title = _custom_spread_title(catalog, legs)
        label_lookup = display_lookup(catalog)
        formula_label = " + ".join(f"{weight:+g}×{label_lookup.get(series_id, series_id)}" for series_id, weight in legs).lstrip("+")
        st.caption(f"公式：{formula_label}；展示单位：{actual_unit or requested_unit}。各腿先换算到同一单位，再在共同交易日按权重合成。")
        trend_tab, season_tab, structure_tab = st.tabs(["历史走势", "季节图", "连续合约结构"])
        with trend_tab:
            metrics = st.columns(4)
            metrics[0].metric("最新", _fmt(spread.iloc[-1]))
            metrics[1].metric("1D", _fmt(spread.diff().iloc[-1] if len(spread) > 1 else np.nan))
            metrics[2].metric("Z60", _fmt(zscore_of_value(spread, spread.iloc[-1], 60)))
            metrics[3].metric("P250", _fmt(percentile_of_value(spread, spread.iloc[-1], 250), 0))
            fig = px.line(spread.rename("价差"), title=spread_title, template=_plot_template())
            fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
            fig.update_yaxes(title=actual_unit or requested_unit)
            _apply_fig_layout(fig, spread_title)
            st.plotly_chart(fig, width="stretch")
            _download_csv("下载自定义价差 CSV", spread.to_frame("spread"), "nap_custom_spread.csv", "download_custom_spread")
        with season_tab:
            _render_spread_seasonality(spread, spread_title, actual_unit or requested_unit, "custom_spread_season")
        with structure_tab:
            curve, series_by_month, structure_unit = _custom_structure_frame(
                wide_raw,
                catalog,
                legs,
                spread_unit_mode,
                unit_factors,
            )
            if curve.empty:
                st.info(structure_unit or "所选序列没有可用连续合约结构。")
            else:
                m1 = series_by_month.get("M1", pd.Series(dtype=float))
                m2 = series_by_month.get("M2", pd.Series(dtype=float))
                structure = pd.concat([m1.rename("m1"), m2.rename("m2")], axis=1).dropna()
                structure_spread = structure["m1"] - structure["m2"] if not structure.empty else pd.Series(dtype=float)
                cards = st.columns(4)
                cards[0].metric("M1", _fmt(m1.iloc[-1] if not m1.empty else np.nan))
                cards[1].metric("M2", _fmt(m2.iloc[-1] if not m2.empty else np.nan))
                cards[2].metric("结构 M1-M2", _fmt(structure_spread.iloc[-1] if not structure_spread.empty else np.nan))
                cards[3].metric(
                    "结构 Z60",
                    _fmt(zscore_of_value(structure_spread, structure_spread.iloc[-1], 60) if not structure_spread.empty else np.nan),
                )
                chart, table_col = st.columns([1.35, 1])
                with chart:
                    curve_plot, structure_asof = _complete_contract_curve(curve, contract_column="合约")
                    fig_curve = go.Figure(
                        go.Scatter(
                            x=curve_plot["month_num"],
                            y=curve_plot["价差"],
                            mode="lines+markers",
                            name="价差",
                            connectgaps=False,
                            customdata=curve_plot[["contract_month", "natural_month_label"]],
                            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.2f}<extra></extra>",
                        )
                    )
                    fig_curve.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
                    fig_curve.update_yaxes(title=structure_unit or actual_unit or requested_unit)
                    _apply_continuous_contract_axis(fig_curve, structure_asof)
                    _apply_fig_layout(fig_curve, f"{spread_title} 当前连续合约结构")
                    fig_curve.update_layout(margin=dict(l=36, r=24, t=54, b=72))
                    st.plotly_chart(fig_curve, width="stretch")
                    st.caption(_contract_month_mapping_caption(structure_asof))
                with table_col:
                    structure_table = curve_plot.rename(
                        columns={"contract_month": "合约", "natural_month_label": "自然月"}
                    )[["合约", "自然月", "价差", "1D", "5D", "20D", "最新日期"]]
                    _safe_dataframe(structure_table, hide_index=True, use_container_width=True)
                _download_csv("下载自定义结构 CSV", pd.DataFrame(series_by_month), "nap_custom_spread_structure.csv", "download_custom_structure")
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
        st.plotly_chart(fig_norm, width="stretch")
    with top[1]:
        scatter = aligned.rename(columns={"a": labels.get(series_a, series_a), "b": labels.get(series_b, series_b)})
        fig_scatter = px.scatter(scatter, x=scatter.columns[1], y=scatter.columns[0], title="散点关系", template=_plot_template())
        _apply_fig_layout(fig_scatter, "散点关系")
        st.plotly_chart(fig_scatter, width="stretch")

    lower = st.columns(3)
    with lower[0]:
        fig_corr = px.line(package["rolling_corr"].rename("相关系数"), title="滚动相关性", template=_plot_template())
        _apply_fig_layout(fig_corr, "滚动相关性")
        st.plotly_chart(fig_corr, width="stretch")
    with lower[1]:
        fig_beta = px.line(package["rolling_beta"].rename("Beta"), title="滚动 Beta", template=_plot_template())
        _apply_fig_layout(fig_beta, "滚动 Beta")
        st.plotly_chart(fig_beta, width="stretch")
    with lower[2]:
        residual = package["residual_z"].dropna()
        fig_resid = px.line(residual.rename("残差 Z-score"), title="回归残差 Z-score", template=_plot_template())
        fig_resid.add_hline(y=2, line_dash="dot", line_color=NEGATIVE)
        fig_resid.add_hline(y=-2, line_dash="dot", line_color=POSITIVE)
        _apply_fig_layout(fig_resid, "回归残差 Z-score")
        st.plotly_chart(fig_resid, width="stretch")

    lead_lag = package["lead_lag"]
    fig_lag = px.bar(lead_lag, x="lag", y="correlation", title="领先滞后相关性", template=_plot_template())
    fig_lag.update_xaxes(title="滞后天数")
    fig_lag.update_yaxes(title="相关系数")
    _apply_fig_layout(fig_lag, "领先滞后相关性")
    st.plotly_chart(fig_lag, width="stretch")
    _download_csv("下载关系分析数据 CSV", aligned, "nap_relationship_aligned.csv", "download_relationship")


def _render_monthly_combo(
    catalog: pd.DataFrame,
    wide_raw: pd.DataFrame,
    default_unit_mode: str,
    unit_factors: dict[str, float],
) -> None:
    combos = _combo_definitions()
    selectors = st.columns([1.45, 0.55])
    selected_name = selectors[0].selectbox("组合", list(combos), key="combo_spread_name")
    with selectors[1]:
        combo_unit_mode, combo_unit = _spread_unit_selector(
            "combo_spread_unit",
            default_unit_mode,
            allow_native=False,
        )
    combo = combos[selected_name]
    formula = str(combo.get(f"formula_{combo_unit_mode}", "两腿统一单位后相减"))
    st.markdown(
        f"""
        <div class="nap-note">
        <strong>逐月组合口径</strong><br>
        {combo["description"]}<br>
        <strong>本图公式：</strong>{formula}<br>
        <strong>展示单位：</strong>{combo_unit}。桶吨系数使用左侧“桶吨换算参数”，每条腿先独立换算，再在共同交易日相减。
        </div>
        """,
        unsafe_allow_html=True,
    )
    curve, series_by_month = _monthly_combo_frame(
        wide_raw,
        catalog,
        combo,
        unit_mode=combo_unit_mode,
        unit_factors=unit_factors,
    )
    if curve.empty or curve["价差"].dropna().empty:
        st.warning("当前组合没有足够数据，请确认 workbook 已刷新且相关 M1-M12 序列存在。")
        _safe_dataframe(curve, use_container_width=True, hide_index=True)
        return
    curve = curve.copy()
    curve["month_num"] = curve["合约"].map(_contract_number)
    curve = curve.sort_values("month_num")
    missing_contracts = curve.loc[curve["价差"].isna(), "合约"].astype(str).tolist()
    if missing_contracts:
        st.warning("源数据缺少可配对报价的合约：" + "、".join(missing_contracts) + "。结构图保留空位，不做插值或月份替代。")

    metric_cols = st.columns(4)
    front_label = "1月" if "1月" in set(curve["合约"].astype(str)) else "M1"
    latest_front = curve[curve["合约"].astype(str) == front_label]["价差"].dropna()
    metric_cols[0].metric(f"{front_label} 最新价差", _fmt(latest_front.iloc[-1] if not latest_front.empty else np.nan))
    metric_cols[1].metric("最强月份", str(curve.dropna(subset=["价差"]).sort_values("价差", ascending=False).iloc[0]["合约"]))
    metric_cols[2].metric("最弱月份", str(curve.dropna(subset=["价差"]).sort_values("价差", ascending=True).iloc[0]["合约"]))
    metric_cols[3].metric("可用月份", f"{curve['价差'].notna().sum()}/12")

    left, right = st.columns([1.35, 1])
    with left:
        curve_plot, combo_asof = _complete_contract_curve(curve, contract_column="合约")
        fig_curve = go.Figure(
            go.Scatter(
                x=curve_plot["month_num"],
                y=curve_plot["价差"],
                mode="lines+markers",
                name="价差",
                connectgaps=False,
                customdata=curve_plot[["contract_month", "natural_month_label"]],
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.2f}<extra></extra>",
            )
        )
        fig_curve.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig_curve.update_yaxes(title=combo_unit)
        _apply_continuous_contract_axis(fig_curve, combo_asof)
        _apply_fig_layout(fig_curve, f"{selected_name} 当前逐月价差")
        fig_curve.update_layout(margin=dict(l=36, r=24, t=54, b=72))
        st.plotly_chart(fig_curve, width="stretch")
        st.caption(_contract_month_mapping_caption(combo_asof))
    with right:
        table = curve_plot.rename(
            columns={"contract_month": "合约", "natural_month_label": "自然月"}
        )[["合约", "自然月", "价差", "1D", "5D", "20D", "最新日期"]]
        _safe_dataframe(table, use_container_width=True, hide_index=True)

    month_options = sorted(series_by_month, key=_contract_number)
    if month_options:
        selected_month = st.selectbox("历史走势月份", month_options, key="combo_spread_month")
        spread = series_by_month[selected_month]
        fig_hist = px.line(spread.rename(f"{selected_month} 价差"), title=f"{selected_name} {selected_month} 历史走势", template=_plot_template())
        fig_hist.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig_hist.update_yaxes(title=combo_unit)
        _apply_fig_layout(fig_hist, f"{selected_name} {selected_month} 历史走势")
        st.plotly_chart(fig_hist, width="stretch")
        wide_combo = pd.DataFrame(series_by_month).sort_index()
        _download_csv("下载逐月组合价差 CSV", wide_combo, "nap_monthly_combo_spreads.csv", "download_combo_spreads")
        _render_combo_m1_seasonality(selected_name, series_by_month, combo_unit)


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
    fig.update_xaxes(title="月份", tickformat="%m月", hoverformat=SEASONAL_HOVER_FORMAT, dtick="M1")
    fig.update_yaxes(title=unit)
    _apply_fig_layout(fig, f"{selected_name} {front_key} 季节图")
    st.plotly_chart(fig, width="stretch")
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
    st.plotly_chart(fig, width="stretch")
    st.caption(f"当前选择：{labels.get(premium_id, premium_id)}；{labels.get(crack_id, crack_id)}；{labels.get(brent_id, brent_id)}")
    _download_csv("下载 MOPJ 对比 CSV", frame, "nap_mopj_driver_frame.csv", "download_mopj_driver")


def _render_time_spread_lab(catalog: pd.DataFrame, wide: pd.DataFrame) -> None:
    st.markdown("#### 月差分析")
    groups = available_curve_groups(catalog)
    if groups.empty:
        st.info("没有可用于月差分析的 M1-Mn 曲线。")
        return
    groups = groups[groups["count"] >= 2].copy()
    cols = st.columns(4)
    sector_options = _sector_order(sorted(groups["sector"].dropna().astype(str).unique()))
    sector_map = {_sector_label(value): value for value in sector_options}
    sector_label = cols[0].selectbox("月差板块", list(sector_map), key="ts_sector")
    sector = sector_map[sector_label]
    sector_groups = groups[groups["sector"] == sector]
    product = cols[1].selectbox("月差品种", sorted(sector_groups["product"].dropna().astype(str).unique()), key="ts_product")
    product_groups = sector_groups[sector_groups["product"] == product]
    region = cols[2].selectbox("月差地区", sorted(product_groups["region"].dropna().astype(str).unique()), key="ts_region")
    family_rows = product_groups[product_groups["region"] == region].sort_values("quote")
    family_options = {
        f"{row['quote']} ({int(row['count'])})": str(row["family_key"])
        for _, row in family_rows.iterrows()
    }
    family_label = cols[3].selectbox("报价", list(family_options), key="ts_family")
    family_key = family_options[family_label]

    group_catalog = curve_contract_catalog(catalog, sector, product, region, family_key)
    is_calendar = not group_catalog.empty and group_catalog["curve_mode"].astype(str).eq("calendar").all()
    group_catalog["month_label"] = group_catalog["curve_label"]
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
    fig = px.line(spread_frame.tail(750), title=f"{sector_label} / {product} / {region} / {family_label} 月差走势", template=_plot_template())
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_yaxes(title="近月 - 远月")
    _apply_fig_layout(fig, f"{sector_label} / {product} / {region} / {family_label} 月差走势")
    st.plotly_chart(fig, width="stretch")
    _safe_dataframe(latest, hide_index=True, use_container_width=True)
    _download_csv("下载月差分析 CSV", spread_frame, "nap_time_spreads.csv", "download_time_spreads")


def render_combo_spreads(
    df: pd.DataFrame,
    default_unit_mode: str,
    unit_factors: dict[str, float],
) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide(df)
    wide_raw = _view_wide_raw(df)
    st.markdown(
        """
        <div class="nap-note">
        <strong>组合价差</strong><br>
        本页把汽油-石脑油的逐月组合、MOPJ 驱动三线图和标准 M1/M2 月差放在一起。所有派生组合只在同日两腿都有报价时计算。
        </div>
        """,
        unsafe_allow_html=True,
    )
    section = st.radio(
        "组合分析",
        ["汽油-石脑油逐月组合", "MOPJ 驱动对比", "月差分析"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if section == "汽油-石脑油逐月组合":
        _render_monthly_combo(catalog, wide_raw, default_unit_mode, unit_factors)
    elif section == "MOPJ 驱动对比":
        _render_mopj_driver_chart(catalog, wide)
    else:
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
    panels = [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]
    for panel, factor_key in zip(panels, ["crude", "crude", "gasoil", "crude"]):
        panel["factor_key"] = factor_key
    return panels


def _weekly_crack_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("NWE石脑油裂解 M1 季节性走势", {"sector": "Cracks", "product": "NWE石脑油裂解", "contract_month": "M1"}),
        ("MOPJ裂解 M1 季节性走势", {"sector": "Cracks", "product": "MOPJ裂解", "contract_month": "M1"}),
        ("NWE石脑油裂解 M2 季节性走势", {"sector": "Cracks", "product": "NWE石脑油裂解", "contract_month": "M2"}),
        ("MOPJ裂解 M2 季节性走势", {"sector": "Cracks", "product": "MOPJ裂解", "contract_month": "M2"}),
    ]
    panels = [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]
    for panel in panels:
        panel["factor_key"] = "naphtha"
    return panels


def _weekly_naphtha_price_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("MOPJ CFR Japan M1", {"sector": "Naphtha", "product": "日本CFR石脑油", "contract_month": "M1"}),
        ("MOPJ CFR Japan M2", {"sector": "Naphtha", "product": "日本CFR石脑油", "contract_month": "M2"}),
        ("NWE CIF石脑油 M1", {"sector": "Naphtha", "product": "NWE CIF石脑油", "contract_month": "M1"}),
        ("NWE CIF石脑油 M2", {"sector": "Naphtha", "product": "NWE CIF石脑油", "contract_month": "M2"}),
    ]
    panels = [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]
    for panel in panels:
        panel["factor_key"] = "naphtha"
        panel["base_unit"] = "USD/bbl"
    return panels


def _weekly_product_crack_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("新加坡92汽油裂解 M1", {"sector": "Cracks", "product": "新加坡92汽油裂解", "contract_month": "M1"}, "gasoline"),
        ("新加坡柴油裂解 M1", {"sector": "Cracks", "product": "新加坡柴油裂解", "contract_month": "M1"}, "gasoil"),
        ("欧洲汽油裂解 M1", {"sector": "Cracks", "product": "欧洲汽油裂解", "contract_month": "M1"}, "gasoline"),
        ("欧洲航煤裂解 M1", {"sector": "Cracks", "product": "欧洲航煤裂解", "contract_month": "M1"}, "jet"),
    ]
    panels = [
        _weekly_panel_from_selector(catalog, wide, title=title, selector=selector)
        for title, selector, _ in presets
    ]
    for panel, (_, _, factor_key) in zip(panels, presets):
        panel["factor_key"] = factor_key
        panel["base_unit"] = "USD/bbl"
    return panels


def _weekly_propane_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("FEI丙烷 M1", {"sector": "Propane/LPG", "product": "FEI丙烷", "contract_month": "M1"}),
        ("FEI丙烷 M2", {"sector": "Propane/LPG", "product": "FEI丙烷", "contract_month": "M2"}),
        ("西北欧LPG M1", {"sector": "Propane/LPG", "product": "西北欧LPG", "contract_month": "M1"}),
        ("西北欧LPG M2", {"sector": "Propane/LPG", "product": "西北欧LPG", "contract_month": "M2"}),
    ]
    panels = [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]
    for panel in panels:
        panel["factor_key"] = "propane"
        panel["base_unit"] = "USD/mt"
    return panels


def _weekly_margin_panels(catalog: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, object]]:
    presets = [
        ("新加坡 Cracking / Dubai", {"sector": "Margins", "ric_prefix": "SGMDUBCRK"}),
        ("西北欧 Cracking / Brent", {"sector": "Margins", "ric_prefix": "NWEMBRTCRK"}),
        ("西北欧 Coking / Brent", {"sector": "Margins", "ric_prefix": "NWEMBRTCOK"}),
        ("地中海 Cracking / Azeri", {"sector": "Margins", "ric_prefix": "MEDMAZECRK"}),
    ]
    panels = [_weekly_panel_from_selector(catalog, wide, title=title, selector=selector) for title, selector in presets]
    for panel in panels:
        panel["factor_key"] = "crude"
        panel["base_unit"] = "USD/bbl"
    return panels


def _weekly_convert_panels(
    panels: list[dict[str, object]],
    unit_mode: str,
    unit_factors: dict[str, float],
) -> list[dict[str, object]]:
    target_unit = "USD/mt" if unit_mode == "mt" else "USD/bbl"
    converted: list[dict[str, object]] = []
    for panel in panels:
        item = dict(panel)
        series = item.get("series")
        factor_key = str(item.get("factor_key", ""))
        factor = float(unit_factors.get(factor_key, 1.0))
        base_unit = str(item.get("base_unit", "USD/bbl"))
        if isinstance(series, pd.Series) and base_unit == "USD/bbl" and target_unit == "USD/mt":
            item["series"] = series * factor
        elif isinstance(series, pd.Series) and base_unit == "USD/mt" and target_unit == "USD/bbl" and factor > 0:
            item["series"] = series / factor
        item["display_unit"] = target_unit
        converted.append(item)
    return converted


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
    curve_plot, curve_asof = _complete_contract_curve(curve, contract_column="合约")
    if "价差" in curve_plot.columns and curve_plot["价差"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=curve_plot["month_num"],
                y=curve_plot["价差"],
                mode="lines+markers",
                name="当前逐月价差",
                showlegend=False,
                line=dict(color="#6070ff", width=2.6),
                marker=dict(size=7),
                connectgaps=False,
                customdata=curve_plot[["contract_month", "natural_month_label"]],
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#8c9aa3", line_width=1, row=1, col=2)
    fig.update_xaxes(
        title_text="月份",
        tickformat="%m月",
        hoverformat=SEASONAL_HOVER_FORMAT,
        dtick="M1",
        showgrid=True,
        gridcolor="#e5ebef",
        row=1,
        col=1,
    )
    _apply_continuous_contract_axis(fig, curve_asof, row=1, col=2)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#e5ebef", zeroline=False, row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=""),
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=54, r=36, t=92, b=68),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12)),
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, sans-serif", size=12, color="#27323a"),
    )
    fig.update_annotations(font=dict(size=15, color="#1f2a33"))
    return fig


def _weekly_combo_short_name(name: str) -> str:
    if "EBOB" in name or "欧洲" in name:
        return "EBOB - NWE石脑油"
    if "新加坡" in name or "MOPJ" in name:
        return "新加坡92汽油纸货 - MOPJ"
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
    curve_plot, curve_asof = _complete_contract_curve(curve, contract_column="合约")
    _apply_continuous_contract_axis(fig, curve_asof, row=row, col=col)
    if "价差" not in curve_plot.columns or not curve_plot["价差"].notna().any():
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
    fig.add_trace(
        go.Scatter(
            x=curve_plot["month_num"],
            y=curve_plot["价差"],
            mode="lines+markers",
            name="当前逐月价差",
            showlegend=False,
            line=dict(color="#6070ff", width=2.6),
            marker=dict(size=7),
            connectgaps=False,
            customdata=curve_plot[["contract_month", "natural_month_label"]],
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8c9aa3", line_width=1, row=row, col=col)
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
        margin=dict(l=54, r=36, t=92, b=68),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12)),
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, sans-serif", size=12, color="#27323a"),
    )
    fig.update_annotations(font=dict(size=15, color="#1f2a33"))
    return fig


def _render_weekly_panel_grid(
    df: pd.DataFrame,
    unit_mode: str,
    unit_factors: dict[str, float],
    *,
    panel_builder,
    title: str,
    filename: str,
    download_key: str,
    zero_line: bool,
) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide(df)
    panels = _weekly_convert_panels(panel_builder(catalog, wide), unit_mode, unit_factors)
    unit = "USD/mt" if unit_mode == "mt" else "USD/bbl"
    fig = _weekly_seasonality_figure(
        panels,
        title=title,
        years=5,
        y_title=unit,
        zero_line=zero_line,
    )
    _weekly_chart(fig, filename, download_key)


def _render_weekly_freight(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_freight_panels,
        title="运费路线季节性走势",
        filename="weekly_freight_seasonality",
        download_key="weekly_freight_png",
        zero_line=False,
    )


def _render_weekly_cracks(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_crack_panels,
        title="石脑油裂解价差季节性走势",
        filename="weekly_naphtha_cracks",
        download_key="weekly_cracks_png",
        zero_line=True,
    )


def _render_weekly_naphtha_prices(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_naphtha_price_panels,
        title="石脑油价格与近端结构季节性",
        filename="weekly_naphtha_prices",
        download_key="weekly_naphtha_prices_png",
        zero_line=False,
    )


def _render_weekly_product_cracks(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_product_crack_panels,
        title="主要成品油裂解价差季节性",
        filename="weekly_product_cracks",
        download_key="weekly_product_cracks_png",
        zero_line=True,
    )


def _render_weekly_propane(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_propane_panels,
        title="丙烷 / LPG 价格与近端结构季节性",
        filename="weekly_propane_structure",
        download_key="weekly_propane_png",
        zero_line=False,
    )


def _render_weekly_margins(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    _render_weekly_panel_grid(
        df,
        unit_mode,
        unit_factors,
        panel_builder=_weekly_margin_panels,
        title="主要炼厂利润季节性",
        filename="weekly_refining_margins",
        download_key="weekly_margins_png",
        zero_line=True,
    )


def _diesel_naphtha_definitions() -> dict[str, dict[str, object]]:
    return {
        "欧洲 LSGO - MOPJ": {
            "left": {"sector": "Diesel", "product": "欧洲低硫柴油/LSGO", "region": "欧洲"},
            "right": {"sector": "Naphtha", "product": "日本CFR石脑油", "region": "日本/东北亚"},
        },
        "新加坡 10ppm柴油 - MOPJ": {
            "left": {"sector": "Diesel", "product": "新加坡10ppm柴油", "region": "新加坡"},
            "right": {"sector": "Naphtha", "product": "日本CFR石脑油", "region": "日本/东北亚"},
        },
    }


def _render_weekly_combo(
    df: pd.DataFrame,
    unit_mode: str,
    unit_factors: dict[str, float],
    *,
    combo_definitions: dict[str, dict[str, object]] | None = None,
    filename: str = "weekly_gasoline_naphtha_spread",
    download_key: str = "weekly_combo_png",
) -> None:
    mode_label = st.radio("组合口径", list(TERM_MODE_OPTIONS), horizontal=True, key="weekly_combo_mode")
    combo_mode = TERM_MODE_OPTIONS[mode_label]
    unit = "USD/mt" if unit_mode == "mt" else "USD/bbl"
    st.caption(f"周报组合先将每条腿独立换算至 {unit}，再在同日相减；桶吨系数来自左侧可编辑参数。")
    combo_df = _filter_term_view(df, combo_mode, CALENDAR_MONTHS)
    catalog = catalog_with_labels(combo_df)
    wide = long_to_wide(combo_df, normalized=False)
    combos = combo_definitions or _combo_definitions()
    items: list[dict[str, object]] = []
    csv_frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name, combo in combos.items():
        curve, series_by_month = _monthly_combo_frame(
            wide,
            catalog,
            combo,
            unit_mode=unit_mode,
            unit_factors=unit_factors,
        )
        short_name = _weekly_combo_short_name(name) if combo_definitions is None else name
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
    fig = _weekly_combo_quad_figure(items, unit=unit)
    _weekly_chart(fig, filename, download_key)
    if csv_frames:
        _download_csv("下载组合数据 CSV", pd.concat(csv_frames, axis=1), f"{filename}.csv", f"{download_key}_csv")


def render_weekly_report(
    df: pd.DataFrame,
    default_unit_mode: str,
    unit_factors: dict[str, float],
) -> None:
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
    controls = st.columns([1.45, 0.55])
    with controls[0]:
        section = st.selectbox(
            "周报图表",
            [
                "运费季节性四宫格",
                "石脑油价格与结构",
                "石脑油裂解四宫格",
                "主要成品油裂解",
                "丙烷 / LPG 价格与结构",
                "柴油-石脑油价差",
                "汽油-石脑油价差",
                "主要炼厂利润",
            ],
        )
    with controls[1]:
        weekly_unit_mode, weekly_unit = _spread_unit_selector(
            "weekly_report_unit",
            default_unit_mode,
            label="出图单位",
            allow_native=False,
        )
    if weekly_unit_mode == "mt":
        st.caption(
            f"当前周报按 {weekly_unit} 出图：原油运费使用原油 {unit_factors['crude']:.2f} bbl/mt，"
            f"成品油运费使用 Gasoil {unit_factors['gasoil']:.2f} bbl/mt，石脑油裂解使用 {unit_factors['naphtha']:.2f} bbl/mt。"
        )
    if section == "运费季节性四宫格":
        _render_weekly_freight(continuous_df, weekly_unit_mode, unit_factors)
    elif section == "石脑油价格与结构":
        _render_weekly_naphtha_prices(continuous_df, weekly_unit_mode, unit_factors)
    elif section == "石脑油裂解四宫格":
        _render_weekly_cracks(continuous_df, weekly_unit_mode, unit_factors)
    elif section == "主要成品油裂解":
        _render_weekly_product_cracks(continuous_df, weekly_unit_mode, unit_factors)
    elif section == "丙烷 / LPG 价格与结构":
        _render_weekly_propane(continuous_df, weekly_unit_mode, unit_factors)
    elif section == "柴油-石脑油价差":
        _render_weekly_combo(
            df,
            weekly_unit_mode,
            unit_factors,
            combo_definitions=_diesel_naphtha_definitions(),
            filename="weekly_diesel_naphtha_spread",
            download_key="weekly_diesel_naphtha_png",
        )
    elif section == "汽油-石脑油价差":
        _render_weekly_combo(df, weekly_unit_mode, unit_factors)
    else:
        _render_weekly_margins(continuous_df, weekly_unit_mode, unit_factors)


def render_forward_curve(df: pd.DataFrame, unit_mode: str, unit_factors: dict[str, float]) -> None:
    catalog = _view_catalog(df)
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
    cols = st.columns(4)
    sector_label_map = {_sector_label(value): value for value in sector_options}
    selected_sector_label = cols[0].selectbox("板块", list(sector_label_map), key="curve_sector")
    sector = sector_label_map[selected_sector_label]
    sector_groups = groups[groups["sector"] == sector]

    product_options = sorted(sector_groups["product"].dropna().astype(str).unique())
    product = cols[1].selectbox("品种", product_options, key="curve_product")
    product_groups = sector_groups[sector_groups["product"] == product]

    region_options = sorted(product_groups["region"].dropna().astype(str).unique())
    region = cols[2].selectbox("地区", region_options, key="curve_region")
    family_rows = product_groups[product_groups["region"] == region].sort_values("quote")
    family_options = {
        f"{row['quote']} ({int(row['count'])})": str(row["family_key"])
        for _, row in family_rows.iterrows()
    }
    family_label = cols[3].selectbox("报价", list(family_options), key="curve_family")
    family_key = family_options[family_label]
    selected = f"{selected_sector_label} / {product} / {region} / {family_label}"
    curve_hist = build_curve_history(df, sector, product, region, family_key=family_key, normalized=False)
    if curve_hist.empty:
        st.warning("当前曲线组没有可用数据。")
        return
    contract_meta = curve_contract_catalog(catalog, sector, product, region, family_key)
    meta = contract_meta.iloc[0] if not contract_meta.empty else pd.Series({"sector": sector, "product": product, "region": region})
    curve_hist["value"], display_unit, display_conversion = convert_quote_values(
        curve_hist["value"], meta, unit_mode, unit_factors
    )
    curve_plot = curve_hist.copy()
    curve_plot["快照"] = curve_plot["snapshot"].map(SNAPSHOT_CN).fillna(curve_plot["snapshot"])
    current_snapshot = curve_plot[curve_plot["snapshot"].eq("Current")]
    forward_asof = _curve_asof(current_snapshot)
    if forward_asof is None:
        forward_asof = _curve_asof(curve_plot)
    fig = go.Figure()
    snapshot_style = {
        "Current": (ACCENT, 3.2, "solid"),
        "7D ago": (AMBER, 2.0, "dash"),
        "30D ago": (NEUTRAL, 1.7, "dot"),
        "90D ago": ("#9aa6ad", 1.3, "dot"),
    }
    for snapshot_name in ["90D ago", "30D ago", "7D ago", "Current"]:
        snapshot_source = curve_plot[curve_plot["snapshot"] == snapshot_name].sort_values("month_num")
        if snapshot_source.empty:
            continue
        snapshot_frame, _ = _complete_contract_curve(snapshot_source, contract_column="contract_month")
        color, width, dash = snapshot_style[snapshot_name]
        fig.add_trace(
            go.Scatter(
                x=snapshot_frame["month_num"],
                y=snapshot_frame["value"],
                mode="lines+markers",
                name=SNAPSHOT_CN.get(snapshot_name, snapshot_name),
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=7 if snapshot_name == "Current" else 5),
                connectgaps=False,
                customdata=snapshot_frame[["contract_month", "natural_month_label"]],
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    _apply_continuous_contract_axis(fig, forward_asof)
    fig.update_yaxes(title=f"数值（{display_unit or '未标注'}）")
    _apply_fig_layout(fig, f"{selected} 远期曲线")
    fig.update_layout(margin=dict(l=36, r=24, t=54, b=72))
    st.plotly_chart(fig, width="stretch")
    st.caption(_contract_month_mapping_caption(forward_asof))
    current = build_forward_curve(df, sector, product, region, family_key=family_key, normalized=False)
    if not current.empty:
        current["value"], _, _ = convert_quote_values(current["value"], meta, unit_mode, unit_factors)
    spreads = forward_curve_spreads(current)
    cols = st.columns(3)
    for idx, (name, value) in enumerate(spreads.items()):
        cols[idx].metric(name, _fmt(value))
    st.caption(f"展示单位：{display_unit or '未标注'}；{display_conversion}")
    pivot = curve_hist.pivot_table(index=["month_num", "contract_month"], columns="snapshot", values="value", aggfunc="last").reset_index()
    pivot = _continuous_month_mapping(forward_asof)[["month_num", "contract_month"]].merge(
        pivot.drop(columns="contract_month", errors="ignore"),
        on="month_num",
        how="left",
    )
    delta_rows = []
    if "Current" in pivot.columns:
        for comparator, label in [("7D ago", "较1周前"), ("30D ago", "较1个月前")]:
            if comparator not in pivot.columns:
                continue
            for _, row in pivot.iterrows():
                delta_rows.append(
                    {
                        "month_num": row["month_num"],
                        "contract_month": row["contract_month"],
                        "比较": label,
                        "变化": row["Current"] - row[comparator],
                    }
                )
    delta = pd.DataFrame(delta_rows).dropna(subset=["变化"]) if delta_rows else pd.DataFrame()
    if not delta.empty:
        fig_delta = px.bar(
            delta,
            x="month_num",
            y="变化",
            color="比较",
            barmode="group",
            color_discrete_map={"较1周前": ACCENT, "较1个月前": AMBER},
            title="曲线逐月变化",
            template=_plot_template(),
        )
        fig_delta.add_hline(y=0, line_color=NEUTRAL, line_width=1)
        _apply_continuous_contract_axis(fig_delta, forward_asof)
        fig_delta.update_yaxes(title="当前 - 历史快照")
        _apply_fig_layout(fig_delta, "曲线逐月变化")
        fig_delta.update_layout(margin=dict(l=36, r=24, t=54, b=72))
        st.plotly_chart(fig_delta, width="stretch")
    _download_csv("下载曲线历史 CSV", curve_hist, "nap_forward_curve.csv", "download_curve")


def render_vol_risk(df: pd.DataFrame) -> None:
    catalog = _view_catalog(df)
    wide = _view_wide(df)
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
        st.plotly_chart(fig, width="stretch")
    with right:
        dd = drawdown_series(series)
        fig_dd = px.area(dd.rename("回撤"), title="回撤", template=_plot_template())
        fig_dd.update_yaxes(title="回撤")
        _apply_fig_layout(fig_dd, "回撤")
        st.plotly_chart(fig_dd, width="stretch")
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
    catalog = _view_catalog(df)
    freight = catalog[catalog["sector"] == "Freight"]
    if freight.empty:
        st.info("没有加载到运费路线。")
        return
    wide = _view_wide(df)
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
        st.plotly_chart(fig, width="stretch")
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
    st.plotly_chart(fig, width="stretch")
    _download_csv("下载套利序列 CSV", aligned, "nap_freight_adjusted_arbitrage.csv", "download_arbitrage")


def _build_data_health(df: pd.DataFrame) -> dict[str, object]:
    base = df
    if "term_type" in base.columns:
        base = base[base["term_type"].astype("string").fillna("continuous").ne("calendar")]
    catalog = catalog_with_labels(base)
    dates = pd.to_datetime(base["date"], errors="coerce")
    latest = dates.max()
    today = pd.Timestamp.now().normalize()
    future_mask = dates.dt.normalize().gt(today)
    future_dates = int(dates[future_mask].dt.normalize().nunique())
    future_rows = int(future_mask.sum())
    duplicate_rows = int(base.duplicated(["date", "series_id"]).sum())

    series_stats = (
        base.groupby("series_id", observed=True)
        .agg(起始日期=("date", "min"), 最新日期=("date", "max"), 观测数=("value", "count"))
        .reset_index()
    )
    meta_columns = ["series_id", "display_name", "sheet", "sector", "product", "region", "unit_native", "ric"]
    series_stats = series_stats.merge(catalog[meta_columns], on="series_id", how="left")
    series_stats["滞后天数"] = (latest - pd.to_datetime(series_stats["最新日期"], errors="coerce")).dt.days
    series_stats["板块"] = series_stats["sector"].map(_sector_label)
    unit_text = series_stats["unit_native"].astype("string").fillna("").str.strip()
    ric_text = series_stats["ric"].astype("string").fillna("").str.strip()
    series_stats["单位状态"] = np.where(unit_text.eq(""), "缺失", "完整")
    series_stats["RIC状态"] = np.where(ric_text.eq(""), "缺失", "完整")

    sheet_stats = (
        base.groupby("sheet", observed=True)
        .agg(
            序列数=("series_id", "nunique"),
            数据行数=("series_id", "size"),
            起始日期=("date", "min"),
            最新日期=("date", "max"),
        )
        .reset_index()
        .rename(columns={"sheet": "工作表"})
    )
    sheet_stats["距全库最新天数"] = (latest - pd.to_datetime(sheet_stats["最新日期"], errors="coerce")).dt.days
    sheet_stats = sheet_stats.sort_values(["距全库最新天数", "工作表"])

    sheet_counts = catalog.groupby("sheet", observed=True)["series_id"].nunique()
    checks = pd.DataFrame(
        [
            {"检查项": "连续月原始序列", "当前值": int(catalog["series_id"].nunique()), "验收线": ">= 500"},
            {"检查项": "Crude 工作表序列", "当前值": int(sheet_counts.get("Crude", 0)), "验收线": ">= 150"},
            {"检查项": "Crk 工作表序列", "当前值": int(sheet_counts.get("Crk", 0)), "验收线": ">= 100"},
            {"检查项": "重复(date, series_id)", "当前值": duplicate_rows, "验收线": "= 0"},
            {"检查项": "未来日期", "当前值": future_dates, "验收线": "= 0"},
        ]
    )
    checks["结果"] = [
        "通过" if checks.iloc[0]["当前值"] >= 500 else "关注",
        "通过" if checks.iloc[1]["当前值"] >= 150 else "关注",
        "通过" if checks.iloc[2]["当前值"] >= 100 else "关注",
        "通过" if duplicate_rows == 0 else "异常",
        "通过" if future_dates == 0 else "异常",
    ]
    return {
        "latest": latest,
        "catalog": catalog,
        "series": series_stats,
        "sheets": sheet_stats,
        "checks": checks,
        "duplicates": duplicate_rows,
        "future_dates": future_dates,
        "future_rows": future_rows,
    }


def render_data_health(df: pd.DataFrame) -> None:
    health = _cached_view(df, "data_health", lambda: _build_data_health(df))
    catalog = health["catalog"]
    series_stats = health["series"]
    stale = series_stats[pd.to_numeric(series_stats["滞后天数"], errors="coerce") > 7]
    unit_missing = int(series_stats["单位状态"].eq("缺失").sum())
    latest = health["latest"]

    st.markdown(
        '<div class="nap-report-strip"><div><strong>数据健康</strong><em>检查最新日期、工作表覆盖、重复键、单位与 RIC 完整度。这里默认检查连续月原始序列，避免自然月派生数据重复计数。</em></div></div>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(5)
    metrics[0].metric("连续月序列", f"{len(catalog):,}")
    metrics[1].metric("最新交易日", latest.strftime("%Y-%m-%d") if pd.notna(latest) else "-")
    metrics[2].metric("未来日期", f"{int(health['future_dates']):,}")
    metrics[3].metric("滞后超过7天", f"{len(stale):,}")
    metrics[4].metric("缺单位 / 重复键", f"{unit_missing:,} / {int(health['duplicates']):,}")

    if int(health["future_dates"]) > 0:
        st.error(
            f"发现 {int(health['future_dates'])} 个未来交易日、{int(health['future_rows']):,} 行记录。"
            "这些记录会影响最新值、Z-score、结构与风险结果，请先核对日期来源。"
        )

    left, right = st.columns([0.8, 1.2])
    with left:
        st.markdown("#### 验收检查")
        _safe_dataframe(health["checks"], use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 工作表更新状态")
        _safe_dataframe(health["sheets"], use_container_width=True, hide_index=True)

    st.markdown("#### 需关注序列")
    attention = series_stats[
        series_stats["滞后天数"].gt(7) | series_stats["单位状态"].eq("缺失") | series_stats["RIC状态"].eq("缺失")
    ].copy()
    attention = attention.sort_values(["滞后天数", "sheet", "display_name"], ascending=[False, True, True])
    attention = attention.rename(
        columns={"display_name": "序列", "sheet": "工作表", "product": "品种", "region": "地区", "ric": "RIC"}
    )[["板块", "品种", "地区", "序列", "RIC", "最新日期", "滞后天数", "观测数", "单位状态", "RIC状态", "工作表"]]
    if attention.empty:
        st.success("未发现滞后、单位缺失或 RIC 缺失的序列。")
    else:
        _safe_dataframe(attention.head(300), use_container_width=True, hide_index=True)
        _download_csv("下载需关注序列 CSV", attention, "nap_data_health_attention.csv", "download_data_health")


def render_glossary(df: pd.DataFrame, explanations: dict) -> None:
    catalog = _view_catalog(df)
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
    unit_mode = str(controls.get("display_unit_mode", "regional"))
    unit_factors = dict(controls.get("unit_factors", DEFAULT_BBL_PER_MT))
    raw_df.attrs["nap_view_key"] = f"{controls['workbook_signature']}|raw|{controls['refresh_token']}"
    df = _filter_term_view(
        raw_df,
        str(controls.get("term_mode", "continuous")),
        list(controls.get("calendar_months", CALENDAR_MONTHS)),  # type: ignore[arg-type]
    )
    month_key = ",".join(str(month) for month in controls.get("calendar_months", CALENDAR_MONTHS))
    df.attrs["nap_view_key"] = (
        f"{controls['workbook_signature']}|{controls.get('term_mode', 'continuous')}|{month_key}|{controls['refresh_token']}"
    )
    if df.empty:
        st.warning("当前查看模式下没有可用数据，请切换连续月/自然月模式，或重新解析当前 Excel。")
        return

    _render_topbar(df, str(controls["workbook_path"]))
    explanations = _load_yaml(default_explanations_path())
    if page == "market":
        render_market_map(df, unit_mode, unit_factors)
    elif page == "detail":
        detail_df = _append_continuous_crude(df, raw_df, str(controls.get("term_mode", "continuous")))
        render_series_detail(detail_df, explanations, unit_mode, unit_factors)
    elif page == "seasonality":
        season_df = _append_continuous_crude(df, raw_df, str(controls.get("term_mode", "continuous")))
        render_seasonality(season_df, unit_mode, unit_factors)
    elif page == "relationship":
        render_relationship_lab(df, unit_mode, unit_factors)
    elif page == "combos":
        render_combo_spreads(df, unit_mode, unit_factors)
    elif page == "weekly":
        render_weekly_report(raw_df, unit_mode, unit_factors)
    elif page == "curve":
        render_forward_curve(df, unit_mode, unit_factors)
    elif page == "risk":
        render_vol_risk(df)
    elif page == "freight":
        render_freight_arbitrage(df)
    elif page == "glossary":
        render_glossary(df, explanations)
    elif page == "health":
        render_data_health(raw_df)


def main() -> None:
    run_nap_dashboard()


if __name__ == "__main__":
    main()
