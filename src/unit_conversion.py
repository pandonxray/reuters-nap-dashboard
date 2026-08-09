from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


DISPLAY_UNIT_OPTIONS = {
    "地区默认（美国 USD/bbl，其他地区 USD/mt）": "regional",
    "原始报价单位": "native",
    "物理货统一 USD/bbl": "bbl",
    "物理货统一 USD/mt": "mt",
}

# Barrels per metric ton. These are display parameters, not calculation
# constants: users can override them in the sidebar when a different density
# convention is required for a specific workflow.
DEFAULT_BBL_PER_MT = {
    "crude": 7.33,
    "naphtha": 8.90,
    "gasoline": 8.33,
    "jet": 7.88,
    "gasoil": 7.45,
    "fuel_oil": 6.35,
    "propane": 12.40,
}

FACTOR_LABELS = {
    "crude": "原油",
    "naphtha": "石脑油",
    "gasoline": "汽油",
    "jet": "航煤",
    "gasoil": "柴油 / Gasoil",
    "fuel_oil": "燃料油",
    "propane": "丙烷 / LPG",
}

NON_PHYSICAL_SECTORS = {"Cracks", "Margins", "Freight", "LNG"}


def is_us_region(region: Any) -> bool:
    text = str(region or "").strip().casefold()
    return text in {"us", "usa", "united states", "美国", "美国湾"} or "usgc" in text


def is_us_quote(meta: Mapping[str, Any]) -> bool:
    if is_us_region(meta.get("region")):
        return True
    ric = str(meta.get("ric") or "").strip().upper()
    return ric.startswith(("RB", "HO", "JETFUSGC", "CAL_RB", "CAL_HO", "CAL_JETFUSGC"))


def product_factor_key(meta: Mapping[str, Any]) -> str | None:
    sector = str(meta.get("sector") or "").casefold()
    text = " ".join(
        str(meta.get(key) or "")
        for key in ("sector", "product", "display_name", "ric")
    ).casefold()
    if sector == "crude":
        return "crude"
    if any(token in text for token in ("naphtha", "mopj", "石脑油")):
        return "naphtha"
    if any(token in text for token in ("jet", "航煤", "煤油", "kerosene")):
        return "jet"
    if any(token in text for token in ("gasoil", "lsgo", "diesel", "heating oil", "柴油")):
        return "gasoil"
    if any(token in text for token in ("gasoline", "ebob", "rbob", "汽油")):
        return "gasoline"
    if any(token in text for token in ("fuel oil", "vlsfo", "hsfo", "燃料油")):
        return "fuel_oil"
    if any(token in text for token in ("propane", "lpg", "丙烷", "液化气")):
        return "propane"
    if any(token in text for token in ("crude", "brent", "wti", "dubai", "原油")):
        return "crude"
    return None


def _canonical_unit(unit: Any) -> str:
    text = str(unit or "").strip().casefold().replace(" ", "")
    aliases = {
        "usd/bbl": "USD/bbl",
        "$/bbl": "USD/bbl",
        "usd/mt": "USD/mt",
        "$/mt": "USD/mt",
        "usd/gal": "USD/gal",
        "$/gal": "USD/gal",
        "usc/gal": "USC/gal",
        "cent/gal": "USC/gal",
        "cents/gal": "USC/gal",
    }
    return aliases.get(text, str(unit or "").strip())


def target_display_unit(meta: Mapping[str, Any], mode: str) -> str:
    native = _canonical_unit(meta.get("unit_native"))
    sector = str(meta.get("sector") or "")
    if mode == "native" or sector in NON_PHYSICAL_SECTORS:
        return native
    if mode == "regional":
        return "USD/bbl" if is_us_quote(meta) else "USD/mt"
    if mode == "bbl":
        return "USD/bbl"
    if mode == "mt":
        return "USD/mt"
    return native


def convert_quote_values(
    values: pd.Series,
    meta: Mapping[str, Any],
    mode: str,
    factors: Mapping[str, float] | None = None,
) -> tuple[pd.Series, str, str]:
    """Convert raw quote values to the requested display unit.

    Returns converted values, the actual displayed unit, and a concise formula.
    Unsupported or non-physical series are preserved in their native unit.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    native = _canonical_unit(meta.get("unit_native"))
    target = target_display_unit(meta, mode)
    if not native or not target or native == target:
        return numeric, native or target, "原始报价，无需换算。"

    factor_key = product_factor_key(meta)
    factor_map = dict(DEFAULT_BBL_PER_MT)
    factor_map.update({key: float(value) for key, value in (factors or {}).items()})
    barrels_per_mt = factor_map.get(factor_key or "")

    if target == "USD/bbl":
        if native == "USD/gal":
            return numeric * 42.0, target, "USD/gal × 42 = USD/bbl。"
        if native == "USC/gal":
            return numeric * 0.42, target, "美分/gal × 0.42 = USD/bbl。"
        if native == "USD/mt" and barrels_per_mt:
            return numeric / barrels_per_mt, target, f"USD/mt ÷ {barrels_per_mt:g} = USD/bbl。"
    elif target == "USD/mt" and barrels_per_mt:
        if native == "USD/bbl":
            return numeric * barrels_per_mt, target, f"USD/bbl × {barrels_per_mt:g} = USD/mt。"
        if native == "USD/gal":
            return numeric * 42.0 * barrels_per_mt, target, f"USD/gal × 42 × {barrels_per_mt:g} = USD/mt。"
        if native == "USC/gal":
            return numeric * 0.42 * barrels_per_mt, target, f"美分/gal × 0.42 × {barrels_per_mt:g} = USD/mt。"

    return numeric, native, f"{native or '未标注单位'} 暂无可靠的 {target} 换算，保留原始报价。"


def convert_frame_rows(
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    series_column: str,
    value_columns: list[str],
    mode: str,
    factors: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    metadata = catalog.set_index("series_id").to_dict(orient="index") if not catalog.empty else {}
    for index, row in out.iterrows():
        meta = metadata.get(str(row.get(series_column)), row.to_dict())
        existing = [column for column in value_columns if column in out.columns]
        converted, unit, formula = convert_quote_values(out.loc[index, existing], meta, mode, factors)
        out.loc[index, existing] = converted.values
        out.loc[index, "unit"] = unit
        out.loc[index, "display_conversion"] = formula
    return out
