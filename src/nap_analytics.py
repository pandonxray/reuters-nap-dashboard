from __future__ import annotations

from collections.abc import Iterable

import re
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


def _make_unique(values: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw in values:
        value = str(raw or "").strip() or "Unnamed"
        count = seen.get(value, 0) + 1
        seen[value] = count
        unique.append(value if count == 1 else f"{value} #{count}")
    return unique


def _row_text(row: pd.Series) -> str:
    parts = [
        row.get("display_name"),
        row.get("short_name"),
        row.get("name_native"),
        row.get("ric"),
        row.get("product"),
        row.get("region"),
        row.get("sheet"),
    ]
    return " ".join(str(part) for part in parts if pd.notna(part)).lower()


def _has_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token.lower() in text for token in tokens)


def _refine_product(row: pd.Series) -> str:
    sector = str(row.get("sector") or "").strip()
    product = str(row.get("product") or "").strip()
    text = _row_text(row)
    ric = str(row.get("ric") or "").strip().upper()

    if sector == "Crude":
        if "murban" in text or ric.startswith("MRBN"):
            return "Murban原油"
        if re.search(r"\bsc\b", text) or ric.startswith("ISC"):
            return "SC原油"
        if "bdss" in text or "dubsw-brtsw" in text:
            return "Dubai-Brent价差"
        if "brent cfd" in text or "brt-" in text:
            return "Brent CFD"
        if "dfl" in text:
            return "Brent DFL"
        if any(token in text for token in ["cushing", "wts"]) or ric.startswith("WTC") or ric.startswith("WTS"):
            return "WTI/Cushing体系"
        if any(token in text for token in ["lls", "mars"]) or ric.startswith(("LLS", "MRS")):
            return "美湾现货原油"
        if product and product != "Crude":
            return product
        return "其他原油"

    if sector == "Gasoline":
        if "rbob" in text or ric.startswith("RB"):
            return "RBOB汽油"
        if "ebob" in text or "nwe" in text:
            return "欧洲EBOB汽油"
        if "97" in text:
            return "新加坡97汽油"
        if "95" in text:
            return "新加坡95汽油"
        if "92" in text or "mog92" in text or "gl92" in text:
            return "新加坡92汽油"
        return "其他汽油"

    if sector == "Naphtha":
        if "mopj" in text or "nacfrjpck" in text or "裂差" in text:
            return "MOPJ裂解/价差"
        if re.search(r"\bew\b", text) or "napjpew" in text:
            return "东西石脑油价差"
        if "dif" in text or "贴水" in text:
            return "石脑油贴水"
        if "japan" in text or "tyo" in text or "cfr" in text or "nacfrjp" in text:
            return "日本CFR石脑油"
        return "石脑油纸货"

    if sector == "Jet/Heating Oil":
        if "ho " in text or ric.startswith("HO"):
            return "NYMEX取暖油"
        if "usg" in text or "usgc" in text:
            return "美国湾航煤"
        if "sin" in text or "singapore" in text:
            return "新加坡航煤"
        if "nwe" in text or "europe" in text or "new" in text:
            return "欧洲航煤"
        return "其他航煤/取暖油"

    if sector == "Diesel":
        if "lgo" in text or ric.startswith("LGO"):
            return "欧洲低硫柴油/LSGO"
        if "10ppm" in text or "go10" in text:
            return "新加坡10ppm柴油"
        return "其他柴油"

    if sector == "Fuel Oil":
        if "vlsfo" in text or "低硫" in text:
            return "低硫燃料油/VLSFO"
        if "hsfo" in text or "fo380" in text or "高硫" in text or "hfo" in text:
            return "高硫燃料油/HSFO"
        return "其他燃料油"

    if sector == "Cracks":
        if "92ron" in text or "mog92sgck" in text:
            return "新加坡92汽油裂解"
        if "sin go" in text or "go10brtck" in text:
            return "新加坡柴油裂解"
        if "fo380" in text or "高硫" in text:
            return "新加坡高硫裂解"
        if "ebob" in text:
            return "欧洲汽油裂解"
        if "jetfcnwe" in text:
            return "欧洲航煤裂解"
        if "jetsg" in text or "singapore jet" in text:
            return "新加坡航煤裂解"
        if "rbob" in text or ric.startswith("RB"):
            return "RBOB-WTI裂解"
        if "ho" in text or ric.startswith("HO"):
            return "HO-WTI裂解"
        if "lgo" in text or ric.startswith("LGO"):
            return "LSGO-Brent裂解"
        return "其他裂解价差"

    if sector == "Margins":
        if "coking" in text or "cok" in ric:
            return "Coking炼厂利润"
        if "topping" in text or "top" in ric:
            return "Topping炼厂利润"
        if "cracking" in text or "crack" in text or "crk" in ric or "ref" in ric:
            return "Cracking炼厂利润"
        return "综合炼厂利润"

    if sector == "Freight":
        if ric.startswith("TC-"):
            return "成品油轮运费"
        if ric.startswith("TD-"):
            return "原油轮运费"
        return "运费路线"

    if sector == "Propane/LPG":
        if "fei" in text:
            return "FEI丙烷"
        if re.search(r"\bcp\b", text):
            return "沙特CP"
        if "mb" in text or "mont belvieu" in text:
            return "Mont Belvieu"
        if "nwe" in text or "nwem" in text or "西北欧" in text:
            return "西北欧LPG"
        if "tyo" in text or "东北亚" in text or "japan" in text:
            return "东北亚丙烷现货"
        return "丙烷/LPG"

    if sector == "LNG":
        if "henry" in text or ric.startswith("A7Q") or "美国" in text:
            return "美国天然气"
        return "LNG现货/纸货"

    return product or sector or "未分类"


def _route_region(ric: str) -> str:
    route_map = {
        "TC-AMS-NYC": "欧洲-美国东岸",
        "TC-FJR-LAV": "中东-地中海",
        "TC-HOU-AMS": "美国湾-欧洲",
        "TC-JGA-LAV": "中东-地中海",
        "TC-LAV-SIN": "地中海-新加坡",
        "TC-SIN1-NGB": "新加坡-中国",
        "TC-YOS-SIN": "韩国-新加坡",
        "TD-BON-HOU": "西非-美国湾",
        "TD-BON-NGB": "西非-中国",
        "TD-BON-RDM": "西非-欧洲",
        "TD-BSR-LAV": "中东-地中海",
        "TD-CRP-MLF": "美国湾-欧洲",
        "TD-HOU-RDM": "美国湾-欧洲",
        "TD-HPT-TAO": "中国沿海",
        "TD-LPP-SIN": "地中海-新加坡",
        "TD-RTA-NGB": "中东-中国",
        "TD-SSO-NGB": "中东-中国",
    }
    return route_map.get(ric.upper(), "")


def _refine_region(row: pd.Series, refined_product: str) -> str:
    sector = str(row.get("sector") or "").strip()
    region = str(row.get("region") or "").strip()
    text = _row_text(row)
    ric = str(row.get("ric") or "").strip().upper()

    if sector == "Crude":
        if refined_product in {"WTI", "WTI/Cushing体系", "美湾现货原油"}:
            return "美国"
        if refined_product in {"Brent", "Brent CFD", "Brent DFL"}:
            return "北海/欧洲"
        if refined_product in {"Dubai", "Murban原油"}:
            return "中东"
        if refined_product == "SC原油":
            return "中国"
        if refined_product == "Dubai-Brent价差":
            return "中东-欧洲"

    if sector == "Gasoline":
        if refined_product == "RBOB汽油":
            return "美国"
        if refined_product == "欧洲EBOB汽油":
            return "欧洲"
        if refined_product.startswith("新加坡"):
            return "新加坡"

    if sector == "Naphtha":
        if refined_product in {"日本CFR石脑油", "MOPJ裂解/价差", "石脑油贴水"}:
            return "日本/东北亚"
        if refined_product == "东西石脑油价差":
            return "欧洲-亚洲"

    if sector == "Diesel":
        if refined_product.startswith("欧洲"):
            return "欧洲"
        if refined_product.startswith("新加坡"):
            return "新加坡"

    if sector == "Jet/Heating Oil":
        if refined_product == "NYMEX取暖油":
            return "美国"
        if refined_product == "美国湾航煤":
            return "美国湾"
        if refined_product == "欧洲航煤":
            return "欧洲"
        if refined_product == "新加坡航煤":
            return "新加坡"

    if sector == "Propane/LPG":
        if refined_product == "FEI丙烷":
            return "亚洲"
        if refined_product == "西北欧LPG":
            return "西北欧"
        if refined_product == "东北亚丙烷现货":
            return "东北亚"
        if refined_product == "沙特CP":
            return "中东"
        if refined_product == "Mont Belvieu":
            return "美国"

    if sector == "Fuel Oil":
        if _has_any(text, ["singapore", "新加坡", "sgsw"]):
            return "新加坡"
        if _has_any(text, ["europe", "欧洲", "nwe", "ara", "hfofaraa"]):
            return "欧洲"

    if sector == "Cracks":
        if refined_product.startswith("新加坡"):
            return "新加坡"
        if refined_product in {"RBOB-WTI裂解", "HO-WTI裂解"}:
            return "美国"
        if refined_product in {"LSGO-Brent裂解", "欧洲汽油裂解", "欧洲航煤裂解"}:
            return "欧洲"

    if sector == "Margins":
        if _has_any(text, ["singapore", "新加坡", "sgm", "sgcs"]):
            return "新加坡"
        if _has_any(text, ["med", "地中海"]):
            return "地中海"
        if _has_any(text, ["nwe", "rot", "欧洲", "西北欧"]):
            return "西北欧"
        if _has_any(text, ["usg", "wti-usg", "美国"]):
            return "美国湾"

    if sector == "Freight":
        route_region = _route_region(ric)
        if route_region:
            return route_region

    if sector == "LNG":
        if refined_product == "美国天然气":
            return "美国"

    cn_region = {
        "Global": "全球",
        "US": "美国",
        "Europe": "欧洲",
        "Singapore": "新加坡",
        "China": "中国",
        "Japan": "日本",
        "Korea": "韩国",
        "Middle East": "中东",
        "Mediterranean": "地中海",
    }
    return cn_region.get(region, region or "未标注")


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
    catalog["product"] = catalog.apply(_refine_product, axis=1)
    catalog["region"] = catalog.apply(lambda row: _refine_region(row, str(row.get("product") or "")), axis=1)
    labels = []
    for _, row in catalog.iterrows():
        parts = [
            row.get("sector"),
            row.get("product"),
            row.get("region"),
            row.get("contract_month"),
        ]
        context = " / ".join(str(part) for part in parts if pd.notna(part) and str(part).strip())
        name = str(row.get("display_name") or row.get("series_id") or "").strip()
        ric = str(row.get("ric") or "").strip()
        label = name
        if context:
            label = f"{context} | {label}"
        if ric:
            label = f"{label} · {ric}"
        labels.append(label)
    catalog["label"] = _make_unique(labels)
    return catalog


def display_lookup(catalog: pd.DataFrame) -> dict[str, str]:
    if catalog.empty:
        return {}
    names = []
    for _, row in catalog.iterrows():
        parts = [
            row.get("display_name"),
            row.get("contract_month"),
            row.get("region"),
            row.get("ric"),
        ]
        names.append(" · ".join(str(part) for part in parts if pd.notna(part) and str(part).strip()))
    return dict(zip(catalog["series_id"], _make_unique(names)))


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
