from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string

logger = logging.getLogger(__name__)

EXCEL_EPOCH = "1899-12-30"
GALLONS_PER_PETROLEUM_BARREL = 42.0
DEFAULT_NAP_WORKBOOK = Path(r"C:\Users\74100\Nutstore\1\油气-djx-\NAP-丙烯-坚果云\Nap.xlsx")
STANDARD_COLUMNS = [
    "date",
    "series_id",
    "display_name",
    "value",
    "sheet",
    "sector",
    "product",
    "region",
    "contract_month",
    "unit_native",
    "unit_normalized",
    "ric",
    "is_derived",
    "source",
]
EXTRA_COLUMNS = ["value_normalized", "name_native", "short_name"]
RELEVANT_SHEETS = [
    "Crude",
    "Gasoline",
    "Heating Oil&Jet fuel",
    "Diesel",
    "Nap",
    "LNG",
    "Crk",
    "Margin",
    "Propane",
    "Fuel oil",
    "Freight",
    "原油",
    "成品油(国内汽柴表)",
    "成品油(国外汽柴表)",
]

SECTOR_BY_SHEET = {
    "Crude": "Crude",
    "Gasoline": "Gasoline",
    "Heating Oil&Jet fuel": "Jet/Heating Oil",
    "Diesel": "Diesel",
    "Nap": "Naphtha",
    "LNG": "LNG",
    "Crk": "Cracks",
    "Margin": "Margins",
    "Propane": "Propane/LPG",
    "Fuel oil": "Fuel Oil",
    "Freight": "Freight",
    "原油": "Freight",
    "成品油(国内汽柴表)": "Freight",
    "成品油(国外汽柴表)": "Freight",
}

GENERIC_META_LABELS = {
    "序列",
    "数据模块",
    "数据简称",
    "数据名称",
    "路透代码",
    "timestamp",
    "voyage rate (bbl)",
}


@dataclass(frozen=True)
class ParsedSeries:
    sheet: str
    display_name: str
    short_name: str
    name_native: str
    ric: str
    source: str
    is_derived: bool
    date_col: int
    value_col: int
    header_row: int
    first_data_row: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cache_path() -> Path:
    return project_root() / "data" / "processed" / "nap_timeseries.parquet"


def default_catalog_path() -> Path:
    return project_root() / "config" / "nap_series_catalog.yaml"


def default_explanations_path() -> Path:
    return project_root() / "config" / "nap_explanations.yaml"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", _clean_text(value)).lower()


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=5).hexdigest()


def _slug(value: str) -> str:
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    ascii_text = re.sub(r"_+", "_", ascii_text)
    return ascii_text


def _make_series_id(sheet: str, display_name: str, ric: str, value_col: int) -> str:
    basis = ric or display_name or f"col_{value_col}"
    slug = _slug(f"{sheet}_{basis}")
    if not slug:
        slug = f"{_slug(sheet)}_{_hash_text(basis) or value_col}"
    suffix = _hash_text(f"{sheet}|{display_name}|{ric}|{value_col}")
    return f"{slug}_{suffix}"


def usd_per_gallon_to_usd_per_barrel(value: float | pd.Series) -> float | pd.Series:
    return value * GALLONS_PER_PETROLEUM_BARREL


def coerce_excel_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(values, errors="coerce")
    serial_mask = numeric.notna() & numeric.between(15000, 90000)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime(
            numeric.loc[serial_mask],
            unit="D",
            origin=EXCEL_EPOCH,
            errors="coerce",
        )
    return parsed


def _looks_like_date_series(values: pd.Series, min_points: int = 8) -> bool:
    return int(coerce_excel_dates(values).notna().sum()) >= min_points


def _looks_like_numeric_series(values: pd.Series, min_points: int = 8) -> bool:
    return int(pd.to_numeric(values, errors="coerce").notna().sum()) >= min_points


def _worksheet_to_frame(ws) -> pd.DataFrame:
    return pd.DataFrame(ws.iter_rows(values_only=True))


def _top_reuters_formulas(ws, max_rows: int = 8) -> list[tuple[int, int, str, str]]:
    formulas: list[tuple[int, int, str, str]] = []
    for row in ws.iter_rows(min_row=1, max_row=min(max_rows, ws.max_row)):
        for cell in row:
            value = cell.value
            if not isinstance(value, str) or not value.startswith("="):
                continue
            upper = value.upper()
            if "_XLL.RDP.HISTORICALPRICING" in upper:
                formulas.append((cell.row, cell.column, "RDP.HistoricalPricing", value))
            elif "_XLL.RHISTORY" in upper:
                formulas.append((cell.row, cell.column, "RHistory", value))
    return formulas


def _formula_output_anchor(formula: str) -> tuple[int, int] | None:
    match = re.search(r",\s*([A-Z]{1,3})(\d+)\s*\)\s*$", formula)
    if not match:
        return None
    return column_index_from_string(match.group(1)), int(match.group(2))


def _first_non_empty_above(df: pd.DataFrame, row0: int, col0: int) -> str:
    for row in range(max(row0 - 1, 0), -1, -1):
        value = _clean_text(df.iat[row, col0]) if col0 < df.shape[1] else ""
        if value and _norm(value) not in GENERIC_META_LABELS and not value.startswith("="):
            return value
    return ""


def _candidate_meta_rows(df: pd.DataFrame, max_left_col0: int) -> list[int]:
    rows: list[int] = []
    max_scan = min(df.shape[0], 1200)
    for row in range(max_scan):
        values = [_clean_text(df.iat[row, col]) for col in range(max(0, max_left_col0))]
        useful = [
            value
            for value in values
            if value
            and not value.startswith("=")
            and _norm(value) not in GENERIC_META_LABELS
            and not re.fullmatch(r"\d+(\.0)?", value)
        ]
        if useful:
            rows.append(row)
    return rows


def _looks_like_ric(value: str) -> bool:
    if not value or any("\u4e00" <= char <= "\u9fff" for char in value):
        return False
    if " " in value or len(value) < 3 or len(value) > 40:
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9.\-=/#]+", value))


def _extract_ric_from_formula(formula: str) -> list[str]:
    match = re.search(r'RHistory\("([^"]+)"', formula, flags=re.IGNORECASE)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(";") if part.strip()]


def _metadata_from_row(df: pd.DataFrame, row: int | None, max_left_col0: int, header: str) -> dict[str, str]:
    if row is None or row < 0 or row >= df.shape[0]:
        return {"ric": "", "short_name": "", "name_native": ""}
    cells = [_clean_text(df.iat[row, col]) for col in range(max(0, max_left_col0))]
    texts = [
        value
        for value in cells
        if value
        and not value.startswith("=")
        and _norm(value) not in GENERIC_META_LABELS
        and not re.fullmatch(r"\d+(\.0)?", value)
    ]
    ric_candidates = [value for value in texts if _looks_like_ric(value)]
    ric = ric_candidates[-1] if ric_candidates else ""
    text_candidates = [value for value in texts if value != ric]
    header_norm = _norm(header)
    exact = [value for value in text_candidates if _norm(value) == header_norm]
    short_name = exact[0] if exact else ""
    if not short_name:
        short_candidates = [value for value in text_candidates if re.search(r"[A-Za-z0-9]", value) and len(value) <= 40]
        short_name = short_candidates[-1] if short_candidates else ""
    native_candidates = [value for value in text_candidates if value != short_name]
    name_native = native_candidates[0] if native_candidates else (short_name or header)
    return {"ric": ric, "short_name": short_name, "name_native": name_native}


def _find_meta_row(df: pd.DataFrame, max_left_col0: int, header: str, pair_index: int) -> int | None:
    header_norm = _norm(header)
    rows = _candidate_meta_rows(df, max_left_col0)
    if header_norm:
        for row in rows:
            left_values = [_norm(df.iat[row, col]) for col in range(max(0, max_left_col0))]
            if header_norm in left_values:
                return row
        for row in rows:
            left_values = [_norm(df.iat[row, col]) for col in range(max(0, max_left_col0))]
            if any(header_norm and (header_norm in value or value in header_norm) for value in left_values if value):
                return row
    if pair_index < len(rows):
        return rows[pair_index]
    return None


def _find_pair_start_near(df: pd.DataFrame, anchor_row0: int, anchor_col0: int) -> tuple[int, int]:
    for offset in range(0, 8):
        col = anchor_col0 + offset
        if col + 1 >= df.shape[1]:
            break
        date_values = df.iloc[anchor_row0:, col]
        value_values = df.iloc[anchor_row0:, col + 1]
        if _looks_like_date_series(date_values) and _looks_like_numeric_series(value_values):
            return anchor_row0, col
    # Some formula anchors point at the header row rather than the first data row.
    for row_offset in range(1, 4):
        row = anchor_row0 + row_offset
        if row >= df.shape[0]:
            break
        for offset in range(0, 8):
            col = anchor_col0 + offset
            if col + 1 >= df.shape[1]:
                break
            if _looks_like_date_series(df.iloc[row:, col]) and _looks_like_numeric_series(df.iloc[row:, col + 1]):
                return row, col
    return anchor_row0, anchor_col0


def _infer_sector(sheet: str, display_name: str) -> str:
    return SECTOR_BY_SHEET.get(sheet, "Other")


def _infer_region(display_name: str, ric: str, sheet: str) -> str:
    text = f"{display_name} {ric} {sheet}".lower()
    if any(token in text for token in ["singapore", "新加坡", "sin"]):
        return "Singapore"
    if any(token in text for token in ["china", "中国", "宁波", "青岛", "tao"]):
        return "China"
    if any(token in text for token in ["japan", "日本", "tokyo", "tyo"]):
        return "Japan"
    if any(token in text for token in ["korea", "韩国", "yos"]):
        return "Korea"
    if any(token in text for token in ["usgc", "美湾", "hou", "美国", "wti", "nymex"]):
        return "US"
    if any(token in text for token in ["europe", "欧洲", "nwe", "rotterdam", "阿姆斯特丹", "france", "法国"]):
        return "Europe"
    if any(token in text for token in ["middle east", "中东", "ras tanura", "dubai"]):
        return "Middle East"
    if any(token in text for token in ["med", "地中海", "lavera"]):
        return "Mediterranean"
    return "Global"


def _infer_product(display_name: str, ric: str, sheet: str) -> str:
    text = f"{display_name} {ric} {sheet}".lower()
    if "wti" in text or ric.startswith("CL"):
        return "WTI"
    if "brent" in text or "brt" in text or ric.startswith("LCO"):
        return "Brent"
    if "dubai" in text:
        return "Dubai"
    if "rbob" in text or ric.startswith("RB"):
        return "RBOB"
    if re.search(r"\bho\b", text) or ric.startswith("HO"):
        return "Heating Oil"
    if "gasoil" in text or ric.startswith("LGO"):
        return "LSGO"
    if "92" in text and "crack" in text:
        return "Singapore 92R Crack"
    if "crack" in text:
        return "Crack"
    if "margin" in text or "炼厂" in text:
        return "Refining Margin"
    if "vlsfo" in text or "低硫" in text:
        return "VLSFO"
    if "hsfo" in text or "高硫" in text or "fo380" in text:
        return "HSFO"
    if "fuel oil" in text or "燃料油" in text:
        return "Fuel Oil"
    if "nap" in text or "naf" in text or "石脑油" in text or "naphtha" in text:
        return "Naphtha"
    if "propane" in text or "lpg" in text or "丙烷" in text:
        return "Propane"
    if sheet in {"原油", "成品油(国内汽柴表)", "成品油(国外汽柴表)"}:
        return "Freight"
    return SECTOR_BY_SHEET.get(sheet, sheet)


def _infer_contract_month(display_name: str, ric: str) -> str:
    text = f"{display_name} {ric}"
    for pattern in [r"\bM(\d{1,2})\b", r"连\s*(\d{1,2})", r"c(\d{1,2})\b"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return f"M{int(match.group(1))}"
    return ""


def _infer_unit(sheet: str, display_name: str, ric: str) -> tuple[str, str]:
    text = f"{display_name} {ric} {sheet}".lower()
    if ric.startswith("RB") or "rbob" in text:
        return "USD/gal", "USD/bbl"
    if ric.startswith("HO") or re.search(r"\bho\b", text):
        return "USD/gal", "USD/bbl"
    if sheet == "Crude" or any(token in text for token in ["wti", "brent", "dubai", "oman"]):
        return "USD/bbl", "USD/bbl"
    if sheet in {"Crk", "Margin"} or "crack" in text:
        return "USD/bbl", "USD/bbl"
    if sheet in {"原油", "成品油(国内汽柴表)", "成品油(国外汽柴表)", "Freight"}:
        return "USD/bbl", "USD/bbl"
    if sheet in {"Diesel", "Nap", "Propane", "Fuel oil"}:
        return "USD/mt", "USD/mt"
    if sheet == "Gasoline":
        return "USD/bbl", "USD/bbl"
    if sheet == "LNG":
        return "USD/MMBtu", "USD/MMBtu"
    return "", ""


def _series_to_records(
    df: pd.DataFrame,
    parsed: ParsedSeries,
    catalog_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    date_values = coerce_excel_dates(df.iloc[parsed.first_data_row :, parsed.date_col])
    values = pd.to_numeric(df.iloc[parsed.first_data_row :, parsed.value_col], errors="coerce")
    frame = pd.DataFrame({"date": date_values, "value": values}).dropna(subset=["date", "value"])
    if frame.empty:
        return []

    display_name = parsed.display_name or parsed.short_name or parsed.name_native or parsed.ric
    series_id = _make_series_id(parsed.sheet, display_name, parsed.ric, parsed.value_col)
    unit_native, unit_normalized = _infer_unit(parsed.sheet, display_name, parsed.ric)
    sector = _infer_sector(parsed.sheet, display_name)
    product = _infer_product(display_name, parsed.ric, parsed.sheet)
    region = _infer_region(display_name, parsed.ric, parsed.sheet)
    contract_month = _infer_contract_month(display_name, parsed.ric)

    if catalog_override:
        display_name = catalog_override.get("display_name") or display_name
        sector = catalog_override.get("sector") or sector
        product = catalog_override.get("product") or product
        region = catalog_override.get("region") or region
        contract_month = catalog_override.get("contract_month") or contract_month
        unit_native = catalog_override.get("unit_native") or unit_native
        unit_normalized = catalog_override.get("unit_normalized") or unit_normalized

    frame["value_normalized"] = frame["value"]
    if unit_native == "USD/gal" and unit_normalized == "USD/bbl":
        frame["value_normalized"] = usd_per_gallon_to_usd_per_barrel(frame["value"])

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "date": pd.Timestamp(row.date).normalize(),
                "series_id": series_id,
                "display_name": display_name,
                "value": float(row.value),
                "sheet": parsed.sheet,
                "sector": sector,
                "product": product,
                "region": region,
                "contract_month": contract_month,
                "unit_native": unit_native,
                "unit_normalized": unit_normalized,
                "ric": parsed.ric,
                "is_derived": bool(parsed.is_derived),
                "source": parsed.source,
                "value_normalized": float(row.value_normalized),
                "name_native": parsed.name_native,
                "short_name": parsed.short_name,
            }
        )
    return rows


def _parse_rdp_sheet(sheet: str, df: pd.DataFrame, formula: tuple[int, int, str, str]) -> list[ParsedSeries]:
    _, _, source, formula_text = formula
    anchor = _formula_output_anchor(formula_text)
    if anchor is None:
        anchor_col0, anchor_row0 = 0, 1
    else:
        anchor_col0, anchor_row0 = anchor[0] - 1, anchor[1] - 1
    first_data_row, first_date_col = _find_pair_start_near(df, anchor_row0, anchor_col0)

    parsed: list[ParsedSeries] = []
    invalid_streak = 0
    pair_index = 0
    for date_col in range(first_date_col, df.shape[1] - 1, 2):
        value_col = date_col + 1
        date_series = df.iloc[first_data_row:, date_col]
        value_series = df.iloc[first_data_row:, value_col]
        valid_dates = coerce_excel_dates(date_series).notna()
        valid_values = pd.to_numeric(value_series, errors="coerce").notna()
        valid_count = int((valid_dates & valid_values).sum())
        if valid_count < 8:
            invalid_streak += 1
            if invalid_streak >= 3:
                break
            continue
        invalid_streak = 0
        header = _first_non_empty_above(df, first_data_row, value_col)
        meta_row = _find_meta_row(df, first_date_col, header, pair_index)
        meta = _metadata_from_row(df, meta_row, first_date_col, header)
        display_name = header or meta["short_name"] or meta["name_native"] or meta["ric"] or f"{sheet} {pair_index + 1}"
        parsed.append(
            ParsedSeries(
                sheet=sheet,
                display_name=display_name,
                short_name=meta["short_name"],
                name_native=meta["name_native"],
                ric=meta["ric"],
                source=source,
                is_derived=False,
                date_col=date_col,
                value_col=value_col,
                header_row=max(first_data_row - 1, 0),
                first_data_row=first_data_row,
            )
        )
        pair_index += 1
    return parsed


def _parse_rhistory_formula_group(sheet: str, df: pd.DataFrame, formula: tuple[int, int, str, str]) -> list[ParsedSeries]:
    row, col, source, formula_text = formula
    col0 = col - 1
    date_col = col0 + 1
    if date_col >= df.shape[1]:
        return []

    header_row0 = row if row < df.shape[0] else max(row - 1, 0)
    first_data_row = header_row0 + 1
    if not _looks_like_date_series(df.iloc[first_data_row:, date_col], min_points=5):
        # RHistory often has formula on row 1/2, field headers one row below, data below that.
        for candidate_row in range(max(0, row), min(df.shape[0] - 1, row + 4)):
            if _looks_like_date_series(df.iloc[candidate_row + 1 :, date_col], min_points=5):
                header_row0 = candidate_row
                first_data_row = candidate_row + 1
                break

    ric_list = _extract_ric_from_formula(formula_text)
    parsed: list[ParsedSeries] = []
    value_col = date_col + 1
    ric_index = 0
    while value_col < df.shape[1]:
        if not _looks_like_numeric_series(df.iloc[first_data_row:, value_col], min_points=5):
            # Stop at separator after parsing at least one value column; otherwise try the next column.
            if parsed:
                break
            value_col += 1
            continue
        if not _looks_like_date_series(df.iloc[first_data_row:, date_col], min_points=5):
            break
        header = _first_non_empty_above(df, first_data_row, value_col)
        if _norm(header) in GENERIC_META_LABELS:
            header = _first_non_empty_above(df, header_row0, value_col)
        if not header:
            header = _first_non_empty_above(df, first_data_row, date_col)
        ric = ric_list[ric_index] if ric_index < len(ric_list) else ""
        display = header or ric or f"{sheet} {value_col + 1}"
        parsed.append(
            ParsedSeries(
                sheet=sheet,
                display_name=display,
                short_name=display,
                name_native=display,
                ric=ric,
                source=source,
                is_derived=False,
                date_col=date_col,
                value_col=value_col,
                header_row=header_row0,
                first_data_row=first_data_row,
            )
        )
        value_col += 1
        ric_index += 1
        if ric_list and ric_index >= len(ric_list):
            break
    return parsed


def _load_catalog_overrides(catalog_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not catalog_path:
        return {}
    path = Path(catalog_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rows = data.get("series", data if isinstance(data, list) else [])
    return {str(row.get("series_id")): dict(row) for row in rows if row.get("series_id")}


def _apply_catalog_overrides(df: pd.DataFrame, catalog_path: str | Path | None) -> pd.DataFrame:
    overrides = _load_catalog_overrides(catalog_path)
    if not overrides or df.empty:
        return df
    out = df.copy()
    editable_cols = [
        "display_name",
        "sector",
        "product",
        "region",
        "contract_month",
        "unit_native",
        "unit_normalized",
        "ric",
    ]
    for series_id, override in overrides.items():
        mask = out["series_id"] == series_id
        if not mask.any():
            continue
        for col in editable_cols:
            value = override.get(col)
            if value not in (None, "") and col in out.columns:
                out.loc[mask, col] = value
    out["value_normalized"] = out["value"]
    mask = (out["unit_native"] == "USD/gal") & (out["unit_normalized"] == "USD/bbl")
    out.loc[mask, "value_normalized"] = usd_per_gallon_to_usd_per_barrel(out.loc[mask, "value"])
    return out


def generate_catalog(df: pd.DataFrame, catalog_path: str | Path | None = None, overwrite: bool = False) -> Path:
    path = Path(catalog_path) if catalog_path else default_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_catalog_overrides(path)
    rows: list[dict[str, Any]] = []
    for _, row in (
        df.sort_values(["sheet", "sector", "product", "display_name"])
        .drop_duplicates("series_id")
        .iterrows()
    ):
        series_id = str(row["series_id"])
        base = {
            "series_id": series_id,
            "display_name": row.get("display_name", ""),
            "sheet": row.get("sheet", ""),
            "sector": row.get("sector", ""),
            "product": row.get("product", ""),
            "region": row.get("region", ""),
            "contract_month": row.get("contract_month", ""),
            "unit_native": row.get("unit_native", ""),
            "unit_normalized": row.get("unit_normalized", ""),
            "ric": row.get("ric", ""),
            "is_derived": bool(row.get("is_derived", False)),
            "source": row.get("source", ""),
        }
        if not overwrite and series_id in existing:
            base.update({key: value for key, value in existing[series_id].items() if value not in (None, "")})
        rows.append(base)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"series": rows}, f, allow_unicode=True, sort_keys=False, width=120)
    return path


def parse_nap_workbook(
    workbook_path: str | Path = DEFAULT_NAP_WORKBOOK,
    catalog_path: str | Path | None = None,
    generate_catalog_file: bool = True,
    sheets: list[str] | None = None,
) -> pd.DataFrame:
    workbook = Path(workbook_path)
    if not workbook.exists():
        raise FileNotFoundError(f"NAP workbook not found: {workbook}")

    wb_formula = load_workbook(workbook, read_only=True, data_only=False)
    wb_values = load_workbook(workbook, read_only=True, data_only=True)
    target_sheets = sheets or [sheet for sheet in RELEVANT_SHEETS if sheet in wb_values.sheetnames]
    all_records: list[dict[str, Any]] = []

    try:
        for sheet in target_sheets:
            if sheet not in wb_values.sheetnames:
                logger.warning("Sheet not found in NAP workbook: %s", sheet)
                continue
            values_ws = wb_values[sheet]
            formula_ws = wb_formula[sheet]
            if values_ws.max_row <= 1 and values_ws.max_column <= 1:
                continue
            df = _worksheet_to_frame(values_ws)
            formulas = _top_reuters_formulas(formula_ws)
            if not formulas:
                continue
            parsed_series: list[ParsedSeries] = []
            rdp_formulas = [item for item in formulas if item[2] == "RDP.HistoricalPricing"]
            rhistory_formulas = [item for item in formulas if item[2] == "RHistory"]
            if rdp_formulas:
                parsed_series.extend(_parse_rdp_sheet(sheet, df, rdp_formulas[0]))
            for formula in rhistory_formulas:
                parsed_series.extend(_parse_rhistory_formula_group(sheet, df, formula))

            for parsed in parsed_series:
                all_records.extend(_series_to_records(df, parsed))
            logger.info("Parsed %s NAP series from sheet %s", len(parsed_series), sheet)
    finally:
        wb_formula.close()
        wb_values.close()

    if not all_records:
        return pd.DataFrame(columns=STANDARD_COLUMNS + EXTRA_COLUMNS)

    out = pd.DataFrame(all_records)
    out = out.sort_values(["series_id", "date"]).drop_duplicates(["date", "series_id"], keep="last")
    out = out[STANDARD_COLUMNS + [col for col in EXTRA_COLUMNS if col in out.columns]]

    if generate_catalog_file:
        generate_catalog(out, catalog_path or default_catalog_path(), overwrite=False)
    out = _apply_catalog_overrides(out, catalog_path or default_catalog_path())
    return out


def _write_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as exc:
        fallback = cache_path.with_suffix(cache_path.suffix + ".pkl")
        logger.warning("Unable to write parquet cache (%s); wrote pickle fallback: %s", exc, fallback)
        df.to_pickle(fallback)


def _read_cache(cache_path: Path) -> pd.DataFrame | None:
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as exc:
            logger.warning("Unable to read parquet cache %s: %s", cache_path, exc)
    fallback = cache_path.with_suffix(cache_path.suffix + ".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    return None


def load_nap_timeseries(
    workbook_path: str | Path = DEFAULT_NAP_WORKBOOK,
    cache_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    workbook = Path(workbook_path)
    cache = Path(cache_path) if cache_path else default_cache_path()
    catalog = Path(catalog_path) if catalog_path else default_catalog_path()
    if not refresh:
        cached = _read_cache(cache)
        if cached is not None:
            fallback = cache.with_suffix(cache.suffix + ".pkl")
            cache_for_mtime = cache if cache.exists() else fallback
            if not workbook.exists() or cache_for_mtime.stat().st_mtime >= workbook.stat().st_mtime:
                return _apply_catalog_overrides(cached, catalog)

    df = parse_nap_workbook(workbook, catalog_path=catalog, generate_catalog_file=True)
    _write_cache(df, cache)
    return df


def series_catalog_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    columns = [
        "series_id",
        "display_name",
        "sheet",
        "sector",
        "product",
        "region",
        "contract_month",
        "unit_native",
        "unit_normalized",
        "ric",
        "is_derived",
        "source",
        "name_native",
        "short_name",
    ]
    return df.sort_values(["sector", "product", "display_name"]).drop_duplicates("series_id")[columns]


def workbook_status(df: pd.DataFrame, workbook_path: str | Path = DEFAULT_NAP_WORKBOOK) -> dict[str, Any]:
    workbook = Path(workbook_path)
    latest = pd.to_datetime(df["date"]).max() if not df.empty else pd.NaT
    return {
        "workbook_path": str(workbook),
        "workbook_mtime": pd.Timestamp.fromtimestamp(workbook.stat().st_mtime) if workbook.exists() else pd.NaT,
        "latest_trade_date": latest,
        "series_count": int(df["series_id"].nunique()) if not df.empty else 0,
        "row_count": int(len(df)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse Reuters NAP workbook into a normalized long table.")
    parser.add_argument("--workbook", default=str(DEFAULT_NAP_WORKBOOK))
    parser.add_argument("--cache", default=str(default_cache_path()))
    parser.add_argument("--catalog", default=str(default_catalog_path()))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_nap_timeseries(args.workbook, args.cache, args.catalog, refresh=args.refresh)
    status = workbook_status(df, args.workbook)
    print(
        f"Parsed {status['series_count']} series / {status['row_count']:,} rows. "
        f"Latest date: {status['latest_trade_date']:%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
