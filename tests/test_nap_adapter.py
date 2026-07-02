from pathlib import Path

import pandas as pd
import pytest

from src.nap_adapter import (
    DEFAULT_NAP_WORKBOOK,
    coerce_excel_dates,
    parse_nap_workbook,
    usd_per_gallon_to_usd_per_barrel,
)
from src.nap_analytics import catalog_with_labels


def _workbook_path() -> Path:
    if DEFAULT_NAP_WORKBOOK.exists():
        return DEFAULT_NAP_WORKBOOK
    matches = list(Path(r"C:\Users\74100\Nutstore\1").rglob("Nap.xlsx"))
    return matches[0] if matches else DEFAULT_NAP_WORKBOOK


@pytest.fixture(scope="module")
def nap_frame() -> pd.DataFrame:
    workbook = _workbook_path()
    if not workbook.exists():
        pytest.skip("Nap.xlsx is not available in this environment")
    return parse_nap_workbook(workbook, generate_catalog_file=False)


def test_excel_serial_date_uses_1899_12_30_origin():
    parsed = coerce_excel_dates(pd.Series([1, 2, 46000]))
    assert parsed.iloc[2] == pd.Timestamp("2025-12-09")


def test_usd_per_gallon_to_usd_per_barrel_conversion():
    assert usd_per_gallon_to_usd_per_barrel(2.5) == 105.0


def test_nap_workbook_detects_required_series_counts(nap_frame: pd.DataFrame):
    counts = nap_frame.groupby("sheet")["series_id"].nunique()
    assert nap_frame["series_id"].nunique() >= 500
    assert counts["Crude"] >= 150
    assert counts["Crk"] >= 100


def test_nap_long_table_has_no_duplicate_date_series_id(nap_frame: pd.DataFrame):
    assert not nap_frame.duplicated(["date", "series_id"]).any()


def test_major_sheets_are_close_to_workbook_latest_date(nap_frame: pd.DataFrame):
    latest = nap_frame["date"].max()
    major_sheets = ["Crude", "Gasoline", "Heating Oil&Jet fuel", "Diesel", "Nap", "Crk", "Margin", "Propane", "Fuel oil"]
    by_sheet = nap_frame.groupby("sheet")["date"].max()
    missing = set(major_sheets) - set(by_sheet.index)
    assert missing <= {"Margin"}
    present = [sheet for sheet in major_sheets if sheet in by_sheet.index]
    assert len(present) >= 8
    for sheet in present:
        assert (latest - by_sheet[sheet]).days <= 5


def test_usd_gal_series_keep_raw_and_normalized_values(nap_frame: pd.DataFrame):
    gal = nap_frame[nap_frame["unit_native"] == "USD/gal"]
    assert not gal.empty
    sample = gal.iloc[0]
    assert sample["unit_normalized"] == "USD/bbl"
    assert sample["value_normalized"] == pytest.approx(sample["value"] * 42)


def test_foreign_product_vlookup_panel_is_not_loaded_as_raw_reuters_series(nap_frame: pd.DataFrame):
    foreign = nap_frame[nap_frame["sheet"] == "成品油(国外汽柴表)"]
    assert foreign["series_id"].nunique() == 5
    assert foreign["source"].eq("RHistory").all()
    assert foreign["date"].min() < pd.Timestamp("2020-01-01")


def test_nwe_naphtha_additions_are_classified_by_market_role(nap_frame: pd.DataFrame):
    catalog = catalog_with_labels(nap_frame)
    outright = catalog[catalog["ric"].astype(str).str.startswith("NAPCNWEAM", na=False)]
    crack = catalog[catalog["ric"].astype(str).str.startswith("NAPCNWEAC", na=False)]
    assert outright["series_id"].nunique() == 12
    assert crack["series_id"].nunique() == 12
    assert set(outright["sector"]) == {"Naphtha"}
    assert set(outright["product"]) == {"NWE CIF石脑油"}
    assert set(outright["region"]) == {"西北欧"}
    assert set(crack["sector"]) == {"Cracks"}
    assert set(crack["product"]) == {"NWE石脑油裂解"}
    assert set(crack["region"]) == {"西北欧"}
