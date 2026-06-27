from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from .nap_analytics import build_spread_series
except ImportError:
    from src.nap_analytics import build_spread_series


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "nap_formula_registry.yaml"


def load_formula_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    registry_path = Path(path) if path else default_registry_path()
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("formulas", []))


def _contains_all(value: str, keywords: list[str]) -> bool:
    text = str(value or "").lower()
    return all(str(keyword).lower() in text for keyword in keywords)


def resolve_leg(catalog: pd.DataFrame, selector: dict[str, Any]) -> str | None:
    if catalog.empty:
        return None
    frame = catalog.copy()
    for col in ["series_id", "display_name", "sheet", "sector", "product", "region", "contract_month", "ric"]:
        value = selector.get(col)
        if value in (None, "") or col not in frame.columns:
            continue
        frame = frame[frame[col].astype(str).str.lower() == str(value).lower()]
    contains = selector.get("contains") or []
    if isinstance(contains, str):
        contains = [contains]
    if contains:
        haystack = (
            frame.get("display_name", pd.Series("", index=frame.index)).astype(str)
            + " "
            + frame.get("ric", pd.Series("", index=frame.index)).astype(str)
            + " "
            + frame.get("product", pd.Series("", index=frame.index)).astype(str)
        )
        frame = frame[haystack.map(lambda value: _contains_all(value, contains))]
    if frame.empty:
        return None
    preferred = selector.get("prefer_contract_month")
    if preferred and "contract_month" in frame.columns:
        matched = frame[frame["contract_month"].astype(str).str.lower() == str(preferred).lower()]
        if not matched.empty:
            frame = matched
    return str(frame.sort_values(["contract_month", "display_name"]).iloc[0]["series_id"])


def evaluate_registry_formula(
    wide: pd.DataFrame,
    catalog: pd.DataFrame,
    formula: dict[str, Any],
) -> pd.Series:
    legs: list[tuple[str, float]] = []
    for leg in formula.get("legs", []):
        selector = dict(leg.get("selector", {}))
        series_id = leg.get("series_id") or resolve_leg(catalog, selector)
        if not series_id:
            continue
        legs.append((str(series_id), float(leg.get("weight", 1.0))))
    spread = build_spread_series(wide, legs)
    spread.name = str(formula.get("name", "custom_formula"))
    return spread


def evaluate_all_registry_formulas(wide: pd.DataFrame, catalog: pd.DataFrame, registry_path: str | Path | None = None) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for formula in load_formula_registry(registry_path):
        series = evaluate_registry_formula(wide, catalog, formula)
        if not series.empty:
            columns[str(formula.get("name", series.name))] = series
    return pd.DataFrame(columns).sort_index() if columns else pd.DataFrame(index=wide.index)
