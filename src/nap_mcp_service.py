from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .nap_adapter import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_NAP_WORKBOOK,
    default_cache_path,
    default_catalog_path,
    default_explanations_path,
    load_nap_timeseries,
)
from .nap_analytics import (
    available_curve_groups,
    build_forward_curve,
    build_market_snapshot,
    catalog_with_labels,
    curve_contract_catalog,
    forward_curve_spreads,
    long_to_wide,
    relationship_package,
)
from .risk_engine import percentile_of_value, summarize_risk_metrics, zscore_of_value
from .seasonal_engine import monthly_box_frame, seasonal_stats


ValueBasis = Literal["native", "normalized"]
Frequency = Literal["daily", "weekly", "monthly"]

MAX_SERIES_PER_QUERY = 20
MAX_POINTS_HARD_LIMIT = 20_000
DEFAULT_POINTS_LIMIT = 5_000
MAX_SEARCH_LIMIT = 500


class NaphthaMCPError(RuntimeError):
    """Base error with a message safe to return through MCP."""


class NaphthaInputError(NaphthaMCPError):
    """The caller supplied an invalid or ambiguous query."""


class NaphthaDataUnavailable(NaphthaMCPError):
    """The configured workbook/cache cannot currently provide data."""


METRIC_DICTIONARY: dict[str, dict[str, Any]] = {
    "value": {
        "label_zh": "Reuters原始值",
        "definition": "Excel 中 Reuters/LSEG 历史价格公式已回填的原始报价，不做桶吨或加仑换算。",
        "unit": "unit_native",
        "grain": "交易日 × series_id",
    },
    "value_normalized": {
        "label_zh": "标准化值",
        "definition": "供跨品种价差使用的换算值；汽油/取暖油加仑价转桶价，部分 USD/mt 物理货按目录系数转 USD/bbl。",
        "unit": "unit_normalized（若无转换则等于 unit_native）",
        "grain": "交易日 × series_id",
        "caveat": "桶吨系数是研究假设，不改变 Reuters 原始值。",
    },
    "latest": {
        "label_zh": "最新值",
        "definition": "该序列最后一个非空观测值；不同序列的 latest_date 可能不同。",
    },
    "chg_1d": {"label_zh": "1日变化", "definition": "当前值减前1个有效交易日观测值，绝对变化而非百分比。"},
    "chg_5d": {"label_zh": "5日变化", "definition": "当前值减前5个有效交易日观测值，绝对变化而非百分比。"},
    "chg_20d": {"label_zh": "20日变化", "definition": "当前值减前20个有效交易日观测值，绝对变化而非百分比。"},
    "log_return": {
        "label_zh": "对数收益",
        "definition": "ln(Pt/Pt-1)；非正价格会被视为不可计算。",
        "unit": "decimal",
    },
    "z_60d": {
        "label_zh": "60日Z-score",
        "definition": "(最新值 - 最近60个有效观测均值) / 最近60个有效观测总体标准差。",
    },
    "pct_250d": {
        "label_zh": "250日历史分位",
        "definition": "最近最多250个有效观测中，小于等于最新值的观测占比。",
        "unit": "0-100",
    },
    "vol_20d": {
        "label_zh": "20日年化波动率",
        "definition": "最近20个对数收益的总体标准差 × sqrt(252)；全序列无有效对数收益时退回一阶差分。",
        "unit": "decimal/年化；差分类序列保留原报价量纲",
    },
    "m1_m2": {
        "label_zh": "M1-M2月差",
        "definition": "同一连续月曲线、同一有效日期的 M1 减 M2。正值为 backwardation，负值为 contango。",
    },
    "m1_m3": {"label_zh": "M1-M3月差", "definition": "同一连续月曲线、同一有效日期的 M1 减 M3。"},
    "m1_m6": {"label_zh": "M1-M6月差", "definition": "同一连续月曲线、同一有效日期的 M1 减 M6。"},
    "rolling_correlation": {
        "label_zh": "滚动相关",
        "definition": "两条对齐序列在指定窗口内的 Pearson 相关系数。",
    },
    "rolling_beta": {
        "label_zh": "滚动Beta",
        "definition": "基于对数收益的滚动协方差除以基准序列滚动方差。",
    },
    "seasonal_percentile": {
        "label_zh": "同日季节性分位",
        "definition": "最新月日相对于最近指定年份同一月日插值观测的经验分位。",
        "unit": "0-100",
    },
    "seasonal_deviation": {
        "label_zh": "同日季节性偏离",
        "definition": "最新值减最近指定年份同一月日观测均值。",
    },
}


ABOUT: dict[str, Any] = {
    "name": "Reuters NAP Dashboard MCP",
    "purpose": "把 Reuters NAP 本地研究工作簿和 Dashboard 分析层暴露为可验证、带来源与新鲜度的 MCP 查询接口。",
    "architecture": [
        "上游 Reuters/LSEG Excel 公式在已登录的 Workspace/Excel 会话中回填工作簿。",
        "适配器读取本地 xlsx，将横向公式区转换为标准日度长表，并派生连续月对应的自然月序列。",
        "工作簿签名由绝对路径、文件大小、修改时间和解析器版本组成；签名变化时重建独立缓存。",
        "MCP 惰性加载同一数据层，查询期间不抓取 Streamlit 会话状态，也不直接访问 Reuters 网络。",
    ],
    "upstream_authentication": "MCP 本身无 Reuters 凭据；上游数据刷新依赖用户已有的 LSEG Workspace/Reuters Excel 在线认证会话。",
    "network_dependency": "stdio MCP 与本地查询无需网络。只有上游 Excel 公式刷新依赖 LSEG/Reuters 网络。",
    "refresh_semantics": "每次工具调用检查工作簿签名；文件大小或修改时间变化会使进程内数据失效并重新解析。MCP 不负责触发 Reuters 公式刷新。",
    "default_workbook": str(DEFAULT_NAP_WORKBOOK),
    "environment_overrides": {
        "NAP_MCP_WORKBOOK": "工作簿完整路径",
        "NAP_MCP_CACHE": "可选固定 parquet 缓存路径；未设置时使用 Dashboard 的签名缓存规则",
        "NAP_MCP_CATALOG": "序列目录 YAML 路径",
        "NAP_MCP_EXPLANATIONS": "价格解释 YAML 路径",
    },
}


def _utc_mtime(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _timestamp(value: Any, *, date_only: bool = False) -> str | None:
    if value is None or value is pd.NaT:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d") if date_only else parsed.isoformat()


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _timestamp(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json_safe(frame.to_dict(orient="records"))


def workbook_signature(path: str | Path) -> str:
    workbook = Path(path)
    if not workbook.exists():
        return "missing"
    stat = workbook.stat()
    try:
        resolved = str(workbook.resolve())
    except OSError:
        resolved = str(workbook)
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|{CACHE_SCHEMA_VERSION}"


def signature_cache_path(base_cache_path: str | Path, signature: str) -> Path:
    base = Path(base_cache_path)
    token = hashlib.blake2b(signature.encode("utf-8"), digest_size=6).hexdigest()
    return base.with_name(f"{base.stem}_{token}{base.suffix}")


def existing_cache_file(cache_path: Path) -> Path | None:
    for candidate in (cache_path, cache_path.with_suffix(cache_path.suffix + ".pkl")):
        if candidate.exists():
            return candidate
    return None


def _compact_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy(deep=False)
    for column in (
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
    ):
        if column in out.columns:
            out[column] = out[column].astype("category")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _parse_date(value: str | None, field: str) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise NaphthaInputError(f"{field} must be an ISO date such as 2026-08-07") from exc
    if pd.isna(parsed):
        raise NaphthaInputError(f"{field} must be an ISO date such as 2026-08-07")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _validate_range(start_date: str | None, end_date: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise NaphthaInputError("start_date must be on or before end_date")
    return start, end


def _validate_limit(value: int, *, maximum: int, field: str = "limit") -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise NaphthaInputError(f"{field} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise NaphthaInputError(f"{field} must be between 1 and {maximum}")
    return parsed


def _filter_text(frame: pd.DataFrame, query: str) -> pd.Series:
    if not query.strip():
        return pd.Series(True, index=frame.index)
    columns = [
        "series_id",
        "display_name",
        "short_name",
        "name_native",
        "ric",
        "sector",
        "product",
        "region",
        "sheet",
        "label",
        "quote",
        "family_key",
        "curve_mode",
    ]
    haystack = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        if column in frame.columns:
            haystack = haystack.str.cat(frame[column].astype("string").fillna(""), sep=" ")
    tokens = [token.lower() for token in query.split() if token.strip()]
    lowered = haystack.str.lower()
    mask = pd.Series(True, index=frame.index)
    for token in tokens:
        mask &= lowered.str.contains(token, regex=False, na=False)
    return mask


def _filter_catalog(
    catalog: pd.DataFrame,
    *,
    query: str = "",
    sector: str | None = None,
    product: str | None = None,
    region: str | None = None,
    term_type: str | None = None,
    contract_month: str | None = None,
    calendar_month: int | None = None,
) -> pd.DataFrame:
    frame = catalog.copy()
    frame = frame[_filter_text(frame, query)]
    for column, expected in (
        ("sector", sector),
        ("product", product),
        ("region", region),
        ("term_type", term_type),
        ("contract_month", contract_month),
    ):
        if expected not in (None, ""):
            frame = frame[frame[column].astype("string").str.casefold().eq(str(expected).casefold())]
    if calendar_month is not None:
        month = int(calendar_month)
        if month < 1 or month > 12:
            raise NaphthaInputError("calendar_month must be between 1 and 12")
        frame = frame[pd.to_numeric(frame["calendar_month"], errors="coerce").eq(month)]
    return frame


def _aggregate_last(frame: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    if frame.empty or frequency == "daily":
        return frame.sort_values(["series_id", "date"])
    out = frame.sort_values(["series_id", "date"]).copy()
    period = "W-FRI" if frequency == "weekly" else "M"
    out["_period"] = out["date"].dt.to_period(period)
    out = out.groupby(["series_id", "_period"], observed=True, sort=True).tail(1)
    return out.drop(columns="_period").sort_values(["series_id", "date"])


def _series_summary(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna().sort_index().astype(float)
    if clean.empty:
        return {}
    latest = float(clean.iloc[-1])
    return json_safe(
        {
            "latest_date": clean.index[-1],
            "latest": latest,
            "chg_1d": float(clean.diff(1).iloc[-1]) if len(clean) > 1 else np.nan,
            "chg_5d": float(clean.diff(5).iloc[-1]) if len(clean) > 5 else np.nan,
            "chg_20d": float(clean.diff(20).iloc[-1]) if len(clean) > 20 else np.nan,
            "z_60d": zscore_of_value(clean, latest, 60),
            "pct_250d": percentile_of_value(clean, latest, 250),
            "observations": int(len(clean)),
        }
    )


class NaphthaDataStore:
    """Lazy, file-signature-aware access to the Dashboard's normalized data layer."""

    def __init__(
        self,
        workbook_path: str | Path = DEFAULT_NAP_WORKBOOK,
        cache_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        explanations_path: str | Path | None = None,
        *,
        static_frame: pd.DataFrame | None = None,
    ) -> None:
        self.workbook_path = Path(workbook_path)
        self.base_cache_path = Path(cache_path) if cache_path else default_cache_path()
        self.explicit_cache_path = cache_path is not None
        self.catalog_path = Path(catalog_path) if catalog_path else default_catalog_path()
        self.explanations_path = Path(explanations_path) if explanations_path else default_explanations_path()
        self._lock = threading.RLock()
        self._frame: pd.DataFrame | None = _compact_frame(static_frame) if static_frame is not None else None
        self._static = static_frame is not None
        self._load_key: str | None = "static" if self._static else None
        self._loaded_at_utc: str | None = datetime.now(timezone.utc).isoformat() if self._static else None
        self._loaded_from: str | None = "injected_frame" if self._static else None
        self._active_cache_path: Path | None = None
        self._derived: dict[str, Any] = {}

    @classmethod
    def from_environment(cls) -> "NaphthaDataStore":
        return cls(
            workbook_path=os.environ.get("NAP_MCP_WORKBOOK", str(DEFAULT_NAP_WORKBOOK)),
            cache_path=os.environ.get("NAP_MCP_CACHE") or None,
            catalog_path=os.environ.get("NAP_MCP_CATALOG") or None,
            explanations_path=os.environ.get("NAP_MCP_EXPLANATIONS") or None,
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "NaphthaDataStore":
        return cls(static_frame=frame)

    def _catalog_mtime_key(self) -> str:
        if not self.catalog_path.exists():
            return "missing"
        stat = self.catalog_path.stat()
        return f"{stat.st_size}|{stat.st_mtime_ns}"

    def _current_key(self) -> str:
        return f"{workbook_signature(self.workbook_path)}|catalog={self._catalog_mtime_key()}"

    def _cache_for_signature(self, signature: str) -> Path:
        if self.explicit_cache_path:
            return self.base_cache_path
        return signature_cache_path(self.base_cache_path, signature)

    def _bootstrap_cache(self, target: Path) -> None:
        if existing_cache_file(target) is not None or not self.workbook_path.exists():
            return
        source = existing_cache_file(self.base_cache_path)
        if source is None or source.stat().st_mtime < self.workbook_path.stat().st_mtime:
            return
        destination = target.with_suffix(target.suffix + ".pkl") if source.name.endswith(".pkl") else target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def load(self, *, force_reparse: bool = False) -> pd.DataFrame:
        if self._static:
            assert self._frame is not None
            return self._frame
        with self._lock:
            key = self._current_key()
            if not force_reparse and self._frame is not None and self._load_key == key:
                return self._frame

            signature = workbook_signature(self.workbook_path)
            cache = self._cache_for_signature(signature)
            self._bootstrap_cache(cache)
            cache_before = existing_cache_file(cache)
            workbook_exists = self.workbook_path.exists()
            cache_fresh = bool(
                cache_before
                and (
                    not workbook_exists
                    or cache_before.stat().st_mtime >= self.workbook_path.stat().st_mtime
                )
            )
            if not workbook_exists and cache_before is None:
                raise NaphthaDataUnavailable(
                    f"Configured NAP workbook does not exist and no cache is available: {self.workbook_path}"
                )

            try:
                frame = load_nap_timeseries(
                    workbook_path=self.workbook_path,
                    cache_path=cache,
                    catalog_path=self.catalog_path,
                    refresh=force_reparse,
                    generate_catalog_file=False,
                )
            except FileNotFoundError as exc:
                raise NaphthaDataUnavailable(str(exc)) from exc
            except Exception as exc:
                raise NaphthaDataUnavailable(f"Unable to load NAP data: {exc}") from exc
            if frame.empty:
                raise NaphthaDataUnavailable("NAP workbook/cache loaded but produced no usable series")

            self._frame = _compact_frame(frame)
            self._load_key = self._current_key()
            self._loaded_at_utc = datetime.now(timezone.utc).isoformat()
            self._loaded_from = "cache" if cache_fresh and not force_reparse else "workbook"
            self._active_cache_path = cache
            self._derived.clear()
            return self._frame

    def catalog(self) -> pd.DataFrame:
        self.load()
        if "catalog" not in self._derived:
            assert self._frame is not None
            self._derived["catalog"] = catalog_with_labels(self._frame)
        return self._derived["catalog"]

    def wide(self, value_basis: ValueBasis = "normalized") -> pd.DataFrame:
        self.load()
        if value_basis not in ("native", "normalized"):
            raise NaphthaInputError("value_basis must be 'native' or 'normalized'")
        key = f"wide:{value_basis}"
        if key not in self._derived:
            assert self._frame is not None
            self._derived[key] = long_to_wide(self._frame, normalized=value_basis == "normalized")
        return self._derived[key]

    def _load_explanations(self) -> dict[str, Any]:
        key = f"explanations:{_utc_mtime(self.explanations_path)}"
        if key in self._derived:
            return self._derived[key]
        if not self.explanations_path.exists():
            data: dict[str, Any] = {}
        else:
            with self.explanations_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        self._derived = {k: v for k, v in self._derived.items() if not k.startswith("explanations:")}
        self._derived[key] = data
        return data

    def _explanation(self, row: Mapping[str, Any]) -> dict[str, Any]:
        data = self._load_explanations()
        series_id = str(row.get("series_id") or "")
        if series_id in data.get("series", {}):
            return dict(data["series"][series_id])
        defaults = data.get("defaults", {})
        return dict(defaults.get(str(row.get("product") or "")) or defaults.get(str(row.get("sector") or "")) or defaults.get("generic", {}))

    def source_meta(self) -> dict[str, Any]:
        frame = self.load()
        latest = pd.to_datetime(frame["date"], errors="coerce").max()
        earliest = pd.to_datetime(frame["date"], errors="coerce").min()
        cache_file = existing_cache_file(self._active_cache_path) if self._active_cache_path else None
        return json_safe(
            {
                "source_id": "reuters_lseg_excel_workbook",
                "source_transport": "local xlsx values populated by Reuters/LSEG Excel formulas",
                "workbook_path": str(self.workbook_path) if not self._static else None,
                "workbook_mtime_utc": _utc_mtime(self.workbook_path) if not self._static else None,
                "cache_path": str(cache_file) if cache_file else None,
                "cache_mtime_utc": _utc_mtime(cache_file),
                "loaded_from": self._loaded_from,
                "loaded_at_utc": self._loaded_at_utc,
                "earliest_trade_date": _timestamp(earliest, date_only=True),
                "latest_trade_date": _timestamp(latest, date_only=True),
                "refresh_semantics": "file-signature invalidation; MCP does not refresh Reuters formulas",
            }
        )

    def response(self, data: Any, **meta: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "data": json_safe(data),
            "meta": {**self.source_meta(), **json_safe(meta)},
        }

    def data_status(self, *, force_reparse: bool = False) -> dict[str, Any]:
        frame = self.load(force_reparse=force_reparse)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        latest = dates.max()
        earliest = dates.min()
        today = pd.Timestamp.now().normalize()
        continuous = frame
        if "term_type" in continuous.columns:
            continuous = continuous[continuous["term_type"].astype("string").fillna("continuous").ne("calendar")]
        latest_by_series = continuous.groupby("series_id", observed=True)["date"].max()
        stale_vs_dataset = int(((latest - pd.to_datetime(latest_by_series, errors="coerce")).dt.days > 7).sum())
        future_mask = dates.dt.normalize().gt(today)
        duplicate_rows = int(frame.duplicated(["date", "series_id"]).sum())
        source_counts = (
            frame.groupby("source", observed=True)["series_id"].nunique().sort_values(ascending=False).to_dict()
            if "source" in frame.columns
            else {}
        )
        sheet_counts = frame.groupby("sheet", observed=True)["series_id"].nunique().sort_values(ascending=False).to_dict()
        issues: list[dict[str, Any]] = []
        if duplicate_rows:
            issues.append({"severity": "critical", "code": "duplicate_grain", "rows": duplicate_rows})
        if int(future_mask.sum()):
            issues.append(
                {
                    "severity": "high",
                    "code": "future_trade_dates",
                    "rows": int(future_mask.sum()),
                    "distinct_dates": int(dates[future_mask].dt.normalize().nunique()),
                }
            )
        if stale_vs_dataset:
            issues.append({"severity": "medium", "code": "series_lag_gt_7d_vs_dataset", "series": stale_vs_dataset})
        lag_days = int((today - latest.normalize()).days) if pd.notna(latest) else None
        return self.response(
            {
                "workbook": {
                    "path": str(self.workbook_path) if not self._static else None,
                    "exists": self.workbook_path.exists() if not self._static else None,
                    "size_bytes": self.workbook_path.stat().st_size if not self._static and self.workbook_path.exists() else None,
                    "mtime_utc": _utc_mtime(self.workbook_path) if not self._static else None,
                    "signature": workbook_signature(self.workbook_path) if not self._static else "static",
                },
                "coverage": {
                    "earliest_trade_date": _timestamp(earliest, date_only=True),
                    "latest_trade_date": _timestamp(latest, date_only=True),
                    "calendar_lag_days": lag_days,
                    "rows": int(len(frame)),
                    "series": int(frame["series_id"].nunique()),
                    "continuous_series": int(continuous["series_id"].nunique()),
                    "calendar_derived_series": int(
                        frame.loc[frame["term_type"].astype("string").eq("calendar"), "series_id"].nunique()
                    )
                    if "term_type" in frame.columns
                    else 0,
                },
                "quality": {
                    "duplicate_date_series_rows": duplicate_rows,
                    "future_rows": int(future_mask.sum()),
                    "future_dates": int(dates[future_mask].dt.normalize().nunique()),
                    "series_lag_gt_7d_vs_dataset_latest": stale_vs_dataset,
                    "issues": issues,
                },
                "source_series_counts": source_counts,
                "sheet_series_counts": sheet_counts,
                "network_used_by_mcp": False,
                "upstream_authentication": ABOUT["upstream_authentication"],
            },
            forced_reparse=force_reparse,
        )

    def search_series(
        self,
        *,
        query: str = "",
        sector: str | None = None,
        product: str | None = None,
        region: str | None = None,
        term_type: str | None = None,
        contract_month: str | None = None,
        calendar_month: int | None = None,
        include_explanations: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        limit = _validate_limit(limit, maximum=MAX_SEARCH_LIMIT)
        matched = _filter_catalog(
            self.catalog(),
            query=query,
            sector=sector,
            product=product,
            region=region,
            term_type=term_type,
            contract_month=contract_month,
            calendar_month=calendar_month,
        ).sort_values(["sector", "product", "region", "term_type", "contract_month", "display_name"])
        total = int(len(matched))
        selected = matched.head(limit).copy()
        columns = [
            "series_id",
            "label",
            "display_name",
            "short_name",
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
            "is_derived",
            "source",
        ]
        result = records(selected[[column for column in columns if column in selected.columns]])
        if include_explanations:
            for item in result:
                item["explanation"] = self._explanation(item)
        return self.response(
            result,
            query={
                "text": query,
                "sector": sector,
                "product": product,
                "region": region,
                "term_type": term_type,
                "contract_month": contract_month,
                "calendar_month": calendar_month,
            },
            total_matches=total,
            returned=len(result),
            truncated=total > len(result),
        )

    def series_detail(self, series_id: str) -> dict[str, Any]:
        matched = self.catalog()[self.catalog()["series_id"].astype(str).eq(str(series_id))]
        if matched.empty:
            raise NaphthaInputError(f"Unknown series_id: {series_id}. Call search_series first.")
        row = records(matched.head(1))[0]
        frame = self.load()
        values = frame[frame["series_id"].astype(str).eq(str(series_id))]
        native = values.set_index("date")["value"].dropna().sort_index()
        normalized_col = "value_normalized" if "value_normalized" in values.columns else "value"
        normalized = values.set_index("date")[normalized_col].dropna().sort_index()
        row["coverage"] = {
            "first_date": _timestamp(values["date"].min(), date_only=True),
            "latest_date": _timestamp(values["date"].max(), date_only=True),
            "observations": int(values["value"].count()),
        }
        row["latest_native"] = _series_summary(native)
        row["latest_normalized"] = _series_summary(normalized)
        row["explanation"] = self._explanation(row)
        return self.response(row)

    def _validated_series_ids(self, series_ids: Sequence[str]) -> list[str]:
        ids = [str(value).strip() for value in series_ids if str(value).strip()]
        if not ids:
            raise NaphthaInputError("series_ids must contain at least one series_id")
        if len(ids) > MAX_SERIES_PER_QUERY:
            raise NaphthaInputError(f"At most {MAX_SERIES_PER_QUERY} series may be queried at once")
        if len(ids) != len(set(ids)):
            raise NaphthaInputError("series_ids must not contain duplicates")
        known = set(self.catalog()["series_id"].astype(str))
        unknown = [series_id for series_id in ids if series_id not in known]
        if unknown:
            raise NaphthaInputError(f"Unknown series_id(s): {', '.join(unknown)}. Call search_series first.")
        return ids

    def _time_slice(
        self,
        series_ids: Sequence[str],
        *,
        start_date: str | None,
        end_date: str | None,
        value_basis: ValueBasis,
        default_days: int = 365,
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        if value_basis not in ("native", "normalized"):
            raise NaphthaInputError("value_basis must be 'native' or 'normalized'")
        start, end = _validate_range(start_date, end_date)
        frame = self.load()
        out = frame[frame["series_id"].astype(str).isin(series_ids)].copy()
        if out.empty:
            raise NaphthaDataUnavailable("Selected series have no observations")
        available_end = pd.to_datetime(out["date"], errors="coerce").max().normalize()
        effective_end = end or available_end
        effective_start = start or (effective_end - pd.Timedelta(days=default_days))
        out = out[(out["date"] >= effective_start) & (out["date"] <= effective_end)]
        if out.empty:
            raise NaphthaDataUnavailable(
                f"No observations in requested range {effective_start:%Y-%m-%d} to {effective_end:%Y-%m-%d}"
            )
        value_column = "value_normalized" if value_basis == "normalized" and "value_normalized" in out.columns else "value"
        out["query_value"] = pd.to_numeric(out[value_column], errors="coerce")
        out["query_unit"] = (
            out["unit_normalized"].astype("string").fillna("")
            if value_basis == "normalized" and "unit_normalized" in out.columns
            else out["unit_native"].astype("string").fillna("")
        )
        return out.dropna(subset=["date", "query_value"]), effective_start, effective_end

    def get_timeseries(
        self,
        series_ids: Sequence[str],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        value_basis: ValueBasis = "normalized",
        frequency: Frequency = "daily",
        max_points: int = DEFAULT_POINTS_LIMIT,
    ) -> dict[str, Any]:
        ids = self._validated_series_ids(series_ids)
        if frequency not in ("daily", "weekly", "monthly"):
            raise NaphthaInputError("frequency must be daily, weekly, or monthly")
        max_points = _validate_limit(max_points, maximum=MAX_POINTS_HARD_LIMIT, field="max_points")
        frame, effective_start, effective_end = self._time_slice(
            ids,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
        )
        daily_frame = frame
        output_frame = _aggregate_last(frame, frequency)
        if len(output_frame) > max_points:
            raise NaphthaInputError(
                f"Query returns {len(output_frame):,} points, above max_points={max_points:,}; shorten the date range or use weekly/monthly frequency"
            )
        catalog = self.catalog().set_index("series_id")
        output = output_frame[
            ["date", "series_id", "display_name", "query_value", "query_unit", "source", "is_derived"]
        ].rename(columns={"query_value": "value", "query_unit": "unit"})
        summaries: dict[str, Any] = {}
        for series_id in ids:
            item = daily_frame[daily_frame["series_id"].astype(str).eq(series_id)].set_index("date")["query_value"]
            summaries[series_id] = {
                "display_name": str(catalog.loc[series_id].get("display_name", series_id)),
                **_series_summary(item),
            }
        return self.response(
            records(output),
            series=summaries,
            value_basis=value_basis,
            frequency=frequency,
            requested_start=start_date,
            requested_end=end_date,
            effective_start=_timestamp(effective_start, date_only=True),
            effective_end=_timestamp(effective_end, date_only=True),
            points=int(len(output)),
        )

    def market_snapshot(
        self,
        *,
        query: str = "",
        sector: str | None = None,
        product: str | None = None,
        region: str | None = None,
        term_type: str | None = None,
        contract_month: str | None = None,
        calendar_month: int | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        limit = _validate_limit(limit, maximum=MAX_SEARCH_LIMIT)
        if "market_snapshot" not in self._derived:
            frame = self.load()
            self._derived["market_snapshot"] = build_market_snapshot(
                frame,
                wide=self.wide("normalized"),
                catalog=self.catalog(),
            )
        snapshot: pd.DataFrame = self._derived["market_snapshot"]
        matched_ids = set(
            _filter_catalog(
                self.catalog(),
                query=query,
                sector=sector,
                product=product,
                region=region,
                term_type=term_type,
                contract_month=contract_month,
                calendar_month=calendar_month,
            )["series_id"].astype(str)
        )
        selected = snapshot[snapshot["series_id"].astype(str).isin(matched_ids)].copy()
        selected = selected.sort_values(["sector", "product", "region", "contract_month", "display_name"])
        total = int(len(selected))
        selected = selected.head(limit)
        return self.response(
            records(selected),
            total_matches=total,
            returned=int(len(selected)),
            truncated=total > len(selected),
            metric_dictionary_uri="naphtha://metrics",
            value_basis="normalized",
        )

    def list_curve_groups(
        self,
        *,
        query: str = "",
        sector: str | None = None,
        product: str | None = None,
        region: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        limit = _validate_limit(limit, maximum=MAX_SEARCH_LIMIT)
        groups = available_curve_groups(self.catalog())
        if not groups.empty:
            groups = groups[_filter_text(groups, query)]
            for column, expected in (("sector", sector), ("product", product), ("region", region)):
                if expected not in (None, ""):
                    groups = groups[groups[column].astype("string").str.casefold().eq(str(expected).casefold())]
        total = int(len(groups))
        return self.response(
            records(groups.head(limit)),
            total_matches=total,
            returned=min(total, limit),
            truncated=total > limit,
        )

    def forward_curve(
        self,
        *,
        sector: str,
        product: str,
        region: str,
        family_key: str | None = None,
        as_of: str | None = None,
        value_basis: ValueBasis = "normalized",
    ) -> dict[str, Any]:
        if value_basis not in ("native", "normalized"):
            raise NaphthaInputError("value_basis must be 'native' or 'normalized'")
        groups = available_curve_groups(self.catalog())
        groups = groups[
            groups["sector"].astype("string").str.casefold().eq(str(sector).casefold())
            & groups["product"].astype("string").str.casefold().eq(str(product).casefold())
            & groups["region"].astype("string").str.casefold().eq(str(region).casefold())
        ]
        if family_key:
            groups = groups[groups["family_key"].astype(str).eq(str(family_key))]
        if groups.empty:
            raise NaphthaInputError("No forward-curve family matches those dimensions; call list_curve_groups first")
        if len(groups) > 1 and not family_key:
            choices = "; ".join(
                f"{row.quote} [{row.family_key}]" for row in groups[["quote", "family_key"]].itertuples(index=False)
            )
            raise NaphthaInputError(f"Multiple curve families match; provide family_key. Choices: {choices}")
        chosen = groups.iloc[0]
        asof = _parse_date(as_of, "as_of")
        frame = self.load()
        contracts = curve_contract_catalog(
            self.catalog(),
            str(chosen["sector"]),
            str(chosen["product"]),
            str(chosen["region"]),
            str(chosen["family_key"]),
        )
        curve_ids = contracts["series_id"].astype(str).tolist()
        curve_wide = self.wide(value_basis).reindex(columns=curve_ids)
        curve = build_forward_curve(
            frame,
            sector=str(chosen["sector"]),
            product=str(chosen["product"]),
            region=str(chosen["region"]),
            asof=asof,
            normalized=value_basis == "normalized",
            family_key=str(chosen["family_key"]),
            catalog=self.catalog(),
            wide=curve_wide,
        )
        if curve.empty:
            raise NaphthaDataUnavailable("Curve family exists but has no values on or before the requested as_of date")
        spreads = forward_curve_spreads(curve)
        return self.response(
            records(curve),
            curve_group=json_safe(chosen.to_dict()),
            requested_as_of=as_of,
            actual_as_of=_timestamp(curve["asof"].max(), date_only=True),
            spreads=spreads,
            value_basis=value_basis,
        )

    def calculate_spread(
        self,
        legs: Sequence[Mapping[str, Any]],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        value_basis: ValueBasis = "normalized",
        frequency: Frequency = "daily",
        max_points: int = DEFAULT_POINTS_LIMIT,
    ) -> dict[str, Any]:
        if not legs or len(legs) > 8:
            raise NaphthaInputError("legs must contain between 1 and 8 entries")
        parsed_legs: list[tuple[str, float]] = []
        for leg in legs:
            series_id = str(leg.get("series_id") or "").strip()
            try:
                weight = float(leg.get("weight", 1.0))
            except (TypeError, ValueError) as exc:
                raise NaphthaInputError(f"Invalid weight for {series_id or 'unnamed leg'}") from exc
            if not series_id or not math.isfinite(weight) or weight == 0:
                raise NaphthaInputError("Each spread leg needs a series_id and a finite non-zero weight")
            parsed_legs.append((series_id, weight))
        ids = self._validated_series_ids([series_id for series_id, _ in parsed_legs])
        if frequency not in ("daily", "weekly", "monthly"):
            raise NaphthaInputError("frequency must be daily, weekly, or monthly")
        max_points = _validate_limit(max_points, maximum=MAX_POINTS_HARD_LIMIT, field="max_points")
        frame, effective_start, effective_end = self._time_slice(
            ids,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
        )
        unit_by_series = (
            frame.groupby("series_id", observed=True)["query_unit"].agg(lambda values: next((str(v) for v in values if str(v)), ""))
        )
        units = {unit for unit in unit_by_series.astype(str) if unit}
        if len(units) > 1:
            raise NaphthaInputError(
                f"Spread legs have incompatible {value_basis} units: {sorted(units)}. Use normalized values or choose compatible series."
            )
        wide = frame.pivot_table(index="date", columns="series_id", values="query_value", aggfunc="last", observed=True)
        aligned = wide[ids].dropna()
        if aligned.empty:
            raise NaphthaDataUnavailable("Spread legs have no overlapping observations in the requested range")
        spread = pd.Series(0.0, index=aligned.index, name="value")
        for series_id, weight in parsed_legs:
            spread = spread + aligned[series_id].astype(float) * weight
        output = pd.DataFrame({"date": spread.index, "series_id": "custom_spread", "value": spread.values})
        output = _aggregate_last(output, frequency)
        if len(output) > max_points:
            raise NaphthaInputError(
                f"Spread returns {len(output):,} points, above max_points={max_points:,}; shorten the range or aggregate"
            )
        catalog = self.catalog().set_index("series_id")
        leg_meta = [
            {
                "series_id": series_id,
                "weight": weight,
                "display_name": str(catalog.loc[series_id].get("display_name", series_id)),
                "unit": str(unit_by_series.get(series_id, "")),
            }
            for series_id, weight in parsed_legs
        ]
        return self.response(
            records(output),
            legs=leg_meta,
            formula=" + ".join(f"({weight:g} × {series_id})" for series_id, weight in parsed_legs),
            unit=next(iter(units), ""),
            value_basis=value_basis,
            frequency=frequency,
            effective_start=_timestamp(effective_start, date_only=True),
            effective_end=_timestamp(effective_end, date_only=True),
            summary=_series_summary(spread),
            points=int(len(output)),
        )

    def analyze_series(
        self,
        series_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        value_basis: ValueBasis = "normalized",
        seasonal_years: int = 10,
    ) -> dict[str, Any]:
        ids = self._validated_series_ids([series_id])
        seasonal_years = _validate_limit(seasonal_years, maximum=20, field="seasonal_years")
        frame, effective_start, effective_end = self._time_slice(
            ids,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            default_days=seasonal_years * 366,
        )
        series = frame.set_index("date")["query_value"].dropna().sort_index().astype(float)
        if series.empty:
            raise NaphthaDataUnavailable("Series has no valid numeric observations in the requested range")
        risk = summarize_risk_metrics(series)
        seasonal = seasonal_stats(series, years=seasonal_years)
        monthly = monthly_box_frame(series, years=seasonal_years)
        monthly_summary = (
            monthly.groupby("month", observed=True)["value"]
            .agg(observations="count", mean="mean", median="median", min="min", max="max")
            .reset_index()
        )
        quantiles = monthly.groupby("month", observed=True)["value"].quantile([0.25, 0.75]).unstack()
        if not quantiles.empty:
            quantiles = quantiles.rename(columns={0.25: "p25", 0.75: "p75"}).reset_index()
            monthly_summary = monthly_summary.merge(quantiles, on="month", how="left")
        meta = self.catalog().set_index("series_id").loc[series_id]
        return self.response(
            {
                "series": {
                    "series_id": series_id,
                    "display_name": str(meta.get("display_name", series_id)),
                    "product": str(meta.get("product", "")),
                    "region": str(meta.get("region", "")),
                    "unit": str(frame["query_unit"].iloc[-1]),
                },
                "summary": _series_summary(series),
                "risk": risk,
                "seasonality": {**seasonal, "years": seasonal_years, "monthly": records(monthly_summary)},
            },
            value_basis=value_basis,
            effective_start=_timestamp(effective_start, date_only=True),
            effective_end=_timestamp(effective_end, date_only=True),
        )

    def compare_series(
        self,
        series_a: str,
        series_b: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        value_basis: ValueBasis = "normalized",
        window: int = 60,
    ) -> dict[str, Any]:
        ids = self._validated_series_ids([series_a, series_b])
        window = _validate_limit(window, maximum=500, field="window")
        if window < 10:
            raise NaphthaInputError("window must be at least 10 observations")
        frame, effective_start, effective_end = self._time_slice(
            ids,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            default_days=1825,
        )
        wide = frame.pivot_table(index="date", columns="series_id", values="query_value", aggfunc="last", observed=True)
        aligned = wide[ids].dropna()
        if len(aligned) < max(3, window):
            raise NaphthaDataUnavailable(
                f"Only {len(aligned)} overlapping observations; at least {window} are required for the requested rolling window"
            )
        package = relationship_package(aligned, series_a, series_b, window=window)
        lead_lag = package.get("lead_lag", pd.DataFrame())
        best_lag: dict[str, Any] | None = None
        if isinstance(lead_lag, pd.DataFrame) and not lead_lag.dropna(subset=["correlation"]).empty:
            valid = lead_lag.dropna(subset=["correlation"]).copy()
            idx = valid["correlation"].abs().idxmax()
            best_lag = json_safe(valid.loc[idx].to_dict())
        rolling_corr = package.get("rolling_corr", pd.Series(dtype=float))
        rolling_beta = package.get("rolling_beta", pd.Series(dtype=float))
        residual_z = package.get("residual_z", pd.Series(dtype=float))
        meta = self.catalog().set_index("series_id")
        result = {
            "series_a": {"series_id": series_a, "display_name": str(meta.loc[series_a].get("display_name", series_a))},
            "series_b": {"series_id": series_b, "display_name": str(meta.loc[series_b].get("display_name", series_b))},
            "observations": int(len(aligned)),
            "first_date": _timestamp(aligned.index.min(), date_only=True),
            "latest_date": _timestamp(aligned.index.max(), date_only=True),
            "level_correlation": float(aligned[series_a].corr(aligned[series_b])),
            "change_correlation": float(aligned[series_a].diff().corr(aligned[series_b].diff())),
            "rolling_correlation_latest": float(rolling_corr.dropna().iloc[-1]) if isinstance(rolling_corr, pd.Series) and not rolling_corr.dropna().empty else None,
            "rolling_beta_latest": float(rolling_beta.dropna().iloc[-1]) if isinstance(rolling_beta, pd.Series) and not rolling_beta.dropna().empty else None,
            "residual_z_latest": float(residual_z.dropna().iloc[-1]) if isinstance(residual_z, pd.Series) and not residual_z.dropna().empty else None,
            "strongest_lead_lag": best_lag,
            "window": window,
        }
        return self.response(
            result,
            value_basis=value_basis,
            effective_start=_timestamp(effective_start, date_only=True),
            effective_end=_timestamp(effective_end, date_only=True),
        )


def metric_dictionary() -> dict[str, Any]:
    return json_safe(METRIC_DICTIONARY)


def about() -> dict[str, Any]:
    return json_safe(ABOUT)
