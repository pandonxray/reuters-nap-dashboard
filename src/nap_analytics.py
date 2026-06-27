from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

try:
    from .nap_adapter import series_catalog_frame
    from .risk_engine import (
        drawdown_series,
        lead_lag_correlation,
        log_returns,
        percentile_of_value,
        price_change,
        realized_volatility,
        regression_residual_zscore,
        rolling_beta,
        rolling_correlation,
        zscore_of_value,
    )
except ImportError:
    from src.nap_adapter import series_catalog_frame
    from src.risk_engine import (
        drawdown_series,
        lead_lag_correlation,
        log_returns,
        percentile_of_value,
        price_change,
        realized_volatility,
        regression_residual_zscore,
        rolling_beta,
        rolling_correlation,
        zscore_of_value,
    )


MARKET_GROUPS = [
    "Crude",
    "Gasoline",
    "Naphtha",
    "Diesel",
    "Jet/Heating Oil",
    "Propane/LPG",
    "Fuel Oil",
    "Freight",
    "Cracks",
    "Margins",
]


def long_to_wide(df: pd.DataFrame, normalized: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    value_col = "value_normalized" if normalized and "value_normalized" in df.columns else "value"
    frame = df[["date", "series_id", value_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.pivot_table(index="date", columns="series_id", values=value_col, aggfunc="last").sort_index()


def catalog_with_labels(df: pd.DataFrame) -> pd.DataFrame:
    catalog = series_catalog_frame(df)
    if catalog.empty:
        return catalog
    catalog = catalog.copy()
    catalog["label"] = catalog.apply(
        lambda row: f"{row['display_name']} · {row['ric']}" if row.get("ric") else str(row["display_name"]),
        axis=1,
    )
    return catalog


def display_lookup(catalog: pd.DataFrame) -> dict[str, str]:
    if catalog.empty:
        return {}
    return dict(zip(catalog["series_id"], catalog["display_name"]))


def _latest_valid(series: pd.Series) -> tuple[pd.Timestamp | pd.NaT, float]:
    clean = series.dropna()
    if clean.empty:
        return pd.NaT, np.nan
    return clean.index[-1], float(clean.iloc[-1])


def structure_labels(catalog: pd.DataFrame, wide: pd.DataFrame) -> dict[str, str]:
    labels = {series_id: "flat/spot" for series_id in catalog.get("series_id", [])}
    if catalog.empty or wide.empty:
        return labels
    contract_catalog = catalog[catalog["contract_month"].astype(str).str.match(r"^M\d+$", na=False)].copy()
    if contract_catalog.empty:
        return labels
    for _, group in contract_catalog.groupby(["sector", "product", "region"], dropna=False):
        m1 = group[group["contract_month"] == "M1"]
        m2 = group[group["contract_month"] == "M2"]
        if m1.empty or m2.empty:
            continue
        m1_id = str(m1.iloc[0]["series_id"])
        m2_id = str(m2.iloc[0]["series_id"])
        if m1_id not in wide.columns or m2_id not in wide.columns:
            continue
        aligned = wide[[m1_id, m2_id]].dropna()
        if aligned.empty:
            continue
        spread = float(aligned.iloc[-1, 0] - aligned.iloc[-1, 1])
        label = "backwardation" if spread > 0 else "contango" if spread < 0 else "flat"
        for series_id in group["series_id"]:
            labels[str(series_id)] = label
    return labels


def build_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wide = long_to_wide(df, normalized=True)
    catalog = catalog_with_labels(df)
    structure = structure_labels(catalog, wide)
    meta = catalog.set_index("series_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for series_id in wide.columns:
        series = wide[series_id].dropna()
        if series.empty:
            continue
        latest_date, latest = _latest_valid(series)
        row_meta = meta.get(series_id, {})
        row = {
            "series_id": series_id,
            "display_name": row_meta.get("display_name", series_id),
            "sheet": row_meta.get("sheet", ""),
            "sector": row_meta.get("sector", ""),
            "product": row_meta.get("product", ""),
            "region": row_meta.get("region", ""),
            "contract_month": row_meta.get("contract_month", ""),
            "unit": row_meta.get("unit_normalized") or row_meta.get("unit_native", ""),
            "ric": row_meta.get("ric", ""),
            "latest_date": latest_date,
            "latest": latest,
            "chg_1d": float(price_change(series, 1).iloc[-1]) if len(series) > 1 else np.nan,
            "chg_5d": float(price_change(series, 5).iloc[-1]) if len(series) > 5 else np.nan,
            "chg_20d": float(price_change(series, 20).iloc[-1]) if len(series) > 20 else np.nan,
            "log_return": float(log_returns(series).iloc[-1]) if len(series) > 1 else np.nan,
            "z_60d": zscore_of_value(series, latest, 60),
            "pct_250d": percentile_of_value(series, latest, 250),
            "vol_20d": float(realized_volatility(series, 20).iloc[-1]) if len(series) > 20 else np.nan,
            "structure": structure.get(series_id, "flat/spot"),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sector_order = {sector: i for i, sector in enumerate(MARKET_GROUPS)}
    out["sector_order"] = out["sector"].map(sector_order).fillna(999)
    return out.sort_values(["sector_order", "sector", "product", "region", "contract_month", "display_name"]).drop(columns="sector_order")


def available_curve_groups(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame(columns=["sector", "product", "region", "count"])
    curves = catalog[catalog["contract_month"].astype(str).str.match(r"^M\d+$", na=False)].copy()
    if curves.empty:
        return pd.DataFrame(columns=["sector", "product", "region", "count"])
    return (
        curves.groupby(["sector", "product", "region"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["sector", "product", "region"])
    )


def _nearest_asof(index: pd.DatetimeIndex, asof: pd.Timestamp) -> pd.Timestamp | None:
    eligible = index[index <= asof]
    if eligible.empty:
        return None
    return eligible.max()


def build_forward_curve(
    df: pd.DataFrame,
    sector: str,
    product: str,
    region: str,
    asof: pd.Timestamp | None = None,
    normalized: bool = True,
) -> pd.DataFrame:
    catalog = catalog_with_labels(df)
    if catalog.empty:
        return pd.DataFrame()
    curves = catalog[
        (catalog["sector"] == sector)
        & (catalog["product"] == product)
        & (catalog["region"] == region)
        & catalog["contract_month"].astype(str).str.match(r"^M\d+$", na=False)
    ].copy()
    if curves.empty:
        return pd.DataFrame()
    curves["month_num"] = curves["contract_month"].str.extract(r"(\d+)").astype(int)
    curves = curves.sort_values("month_num")
    wide = long_to_wide(df[df["series_id"].isin(curves["series_id"])], normalized=normalized)
    if wide.empty:
        return pd.DataFrame()
    selected_asof = _nearest_asof(wide.index, asof or wide.index.max())
    if selected_asof is None:
        return pd.DataFrame()
    rows = []
    for _, row in curves.iterrows():
        value = wide.at[selected_asof, row["series_id"]] if row["series_id"] in wide.columns else np.nan
        rows.append(
            {
                "asof": selected_asof,
                "contract_month": row["contract_month"],
                "month_num": int(row["month_num"]),
                "series_id": row["series_id"],
                "display_name": row["display_name"],
                "value": value,
                "unit": row.get("unit_normalized") or row.get("unit_native", ""),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["value"])


def build_curve_history(
    df: pd.DataFrame,
    sector: str,
    product: str,
    region: str,
    offsets: Iterable[int] = (0, 7, 30, 90),
) -> pd.DataFrame:
    wide = long_to_wide(df, normalized=True)
    if wide.empty:
        return pd.DataFrame()
    latest = wide.index.max()
    curves = []
    for offset in offsets:
        curve = build_forward_curve(df, sector, product, region, latest - pd.Timedelta(days=int(offset)))
        if curve.empty:
            continue
        label = "Current" if offset == 0 else f"{offset}D ago"
        curve["snapshot"] = label
        curves.append(curve)
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def forward_curve_spreads(curve: pd.DataFrame) -> dict[str, float]:
    if curve.empty:
        return {"M1-M2": np.nan, "M1-M3": np.nan, "M1-M6": np.nan}
    values = curve.sort_values("month_num").drop_duplicates("contract_month").set_index("contract_month")["value"]
    return {
        "M1-M2": float(values.get("M1", np.nan) - values.get("M2", np.nan)),
        "M1-M3": float(values.get("M1", np.nan) - values.get("M3", np.nan)),
        "M1-M6": float(values.get("M1", np.nan) - values.get("M6", np.nan)),
    }


def build_spread_series(wide: pd.DataFrame, legs: list[tuple[str, float]]) -> pd.Series:
    if wide.empty or not legs:
        return pd.Series(dtype=float)
    out = pd.Series(0.0, index=wide.index, dtype=float)
    required = []
    for series_id, weight in legs:
        if series_id not in wide.columns:
            continue
        out = out.add(wide[series_id].astype(float).multiply(float(weight)), fill_value=np.nan)
        required.append(series_id)
    if not required:
        return pd.Series(dtype=float)
    return out.dropna()


def relationship_package(wide: pd.DataFrame, series_a: str, series_b: str, window: int = 60) -> dict[str, pd.DataFrame | pd.Series]:
    if series_a not in wide.columns or series_b not in wide.columns:
        return {}
    a = wide[series_a].dropna()
    b = wide[series_b].dropna()
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if aligned.empty:
        return {}
    return {
        "aligned": aligned,
        "rolling_corr": rolling_correlation(aligned["a"], aligned["b"], window),
        "rolling_beta": rolling_beta(aligned["a"], aligned["b"], window),
        "lead_lag": lead_lag_correlation(aligned["a"], aligned["b"], max_lag=20),
        "residual_z": regression_residual_zscore(aligned["a"], aligned["b"], regression_window=max(window, 30), z_window=min(window, 60)),
        "drawdown_a": drawdown_series(aligned["a"]),
        "drawdown_b": drawdown_series(aligned["b"]),
    }
