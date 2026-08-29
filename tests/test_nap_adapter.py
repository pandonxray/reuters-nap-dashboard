from pathlib import Path

import pandas as pd
import pytest

from src.nap_adapter import (
    DEFAULT_NAP_WORKBOOK,
    ParsedSeries,
    _looks_like_ric,
    _quarantine_duplicate_ric_requests,
    _validate_unique_ric_requests,
    coerce_excel_dates,
    coerce_excel_numeric,
    load_nap_timeseries,
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


def test_excel_date_formatted_negative_number_is_recovered():
    values = pd.Series([pd.Timestamp("1899-12-23 12:00:00"), pd.Timestamp("2026-07-10")])
    parsed = coerce_excel_numeric(values)
    assert parsed.iloc[0] == pytest.approx(-6.5)
    assert pd.isna(parsed.iloc[1])


def test_retrieval_status_is_not_misclassified_as_a_ric():
    assert not _looks_like_ric("Retrieving...")
    assert _looks_like_ric("NAF-SIN-DIF")


def test_usd_per_gallon_to_usd_per_barrel_conversion():
    assert usd_per_gallon_to_usd_per_barrel(2.5) == 105.0


def test_duplicate_reuters_ric_requests_fail_before_contract_data_is_overwritten():
    shared = {
        "sheet": "Nap",
        "short_name": "",
        "name_native": "",
        "source": "RDP.HistoricalPricing",
        "is_derived": False,
        "header_row": 0,
        "first_data_row": 1,
    }
    parsed = [
        ParsedSeries(display_name="日本CFR连2", ric="NACFRJPSWMc2", date_col=17, value_col=18, **shared),
        ParsedSeries(display_name="日本CFR连3", ric="NACFRJPSWMc2", date_col=19, value_col=20, **shared),
    ]

    with pytest.raises(ValueError, match=r"NACFRJPSWMc2.*日本CFR连2, 日本CFR连3"):
        _validate_unique_ric_requests(parsed)


def test_duplicate_reuters_ric_requests_are_quarantined_for_dashboard_loading():
    shared = {
        "sheet": "Nap",
        "short_name": "",
        "name_native": "",
        "source": "RDP.HistoricalPricing",
        "is_derived": False,
        "header_row": 0,
        "first_data_row": 1,
    }
    parsed = [
        ParsedSeries(display_name="日本CFR连2", ric="NACFRJPSWMc2", date_col=17, value_col=18, **shared),
        ParsedSeries(display_name="日本CFR连3", ric="NACFRJPSWMc2", date_col=19, value_col=20, **shared),
        ParsedSeries(display_name="EW连1", ric="NAPJPEWMc1", date_col=21, value_col=22, **shared),
    ]

    clean, issues = _quarantine_duplicate_ric_requests(parsed)

    assert [item.display_name for item in clean] == ["日本CFR连2", "EW连1"]
    assert issues[0]["ric"] == "NACFRJPSWMc2"
    assert issues[0]["quarantined_labels"] == ["日本CFR连3"]


def test_nap_workbook_detects_required_series_counts(nap_frame: pd.DataFrame):
    counts = nap_frame.groupby("sheet")["series_id"].nunique()
    assert nap_frame["series_id"].nunique() >= 500
    assert counts["Crude"] >= 150
    assert counts["Crk"] >= 100


def test_nap_long_table_has_no_duplicate_date_series_id(nap_frame: pd.DataFrame):
    assert not nap_frame.duplicated(["date", "series_id"]).any()


def test_major_sheets_are_within_five_business_days_of_workbook_latest(nap_frame: pd.DataFrame):
    latest = nap_frame["date"].max()
    major_sheets = ["Crude", "Gasoline", "Heating Oil&Jet fuel", "Diesel", "Nap", "Crk", "Margin", "Propane", "Fuel oil"]
    by_sheet = nap_frame.groupby("sheet")["date"].max()
    missing = set(major_sheets) - set(by_sheet.index)
    assert missing <= {"Margin"}
    present = [sheet for sheet in major_sheets if sheet in by_sheet.index]
    assert len(present) >= 8
    for sheet in present:
        business_day_lag = len(pd.bdate_range(by_sheet[sheet] + pd.offsets.BDay(1), latest))
        assert business_day_lag <= 5


def test_usd_gal_series_keep_raw_and_normalized_values(nap_frame: pd.DataFrame):
    gal = nap_frame[nap_frame["unit_native"] == "USD/gal"]
    assert not gal.empty
    sample = gal.iloc[0]
    assert sample["unit_normalized"] == "USD/bbl"
    assert sample["value_normalized"] == pytest.approx(sample["value"] * 42)


def test_jet_quote_units_are_explicit_and_normalized(nap_frame: pd.DataFrame):
    usg = nap_frame[nap_frame["ric"].astype(str).str.startswith("JETFUSGCMc1", na=False)]
    nwe = nap_frame[nap_frame["ric"].astype(str).str.startswith("JETFCNWEMc1", na=False)]
    singapore = nap_frame[nap_frame["ric"].astype(str).str.startswith("JETSGSWMc1", na=False)]
    assert not usg.empty and not nwe.empty and not singapore.empty
    assert usg.iloc[-1]["unit_native"] == "USC/gal"
    assert usg.iloc[-1]["value_normalized"] == pytest.approx(usg.iloc[-1]["value"] * 0.42)
    assert nwe.iloc[-1]["unit_native"] == "USD/mt"
    assert nwe.iloc[-1]["value_normalized"] == pytest.approx(nwe.iloc[-1]["value"] / 7.88)
    assert singapore.iloc[-1]["unit_native"] == "USD/bbl"


def test_metric_ton_products_normalize_to_barrels_when_used_in_spreads(nap_frame: pd.DataFrame):
    ebob = nap_frame[nap_frame["ric"].astype(str).str.startswith("EBOBNWEMc1", na=False)]
    nwe_nap = nap_frame[nap_frame["ric"].astype(str).str.startswith("NAPCNWEAMc1", na=False)]
    assert not ebob.empty
    assert not nwe_nap.empty

    ebob_sample = ebob.iloc[-1]
    nwe_sample = nwe_nap.iloc[-1]
    assert ebob_sample["unit_native"] == "USD/mt"
    assert ebob_sample["unit_normalized"] == "USD/bbl"
    assert ebob_sample["value_normalized"] == pytest.approx(ebob_sample["value"] / 8.33)
    assert nwe_sample["unit_native"] == "USD/mt"
    assert nwe_sample["unit_normalized"] == "USD/bbl"
    assert nwe_sample["value_normalized"] == pytest.approx(nwe_sample["value"] / 8.9)


def test_ric_contract_month_overrides_mislabeled_display_name(nap_frame: pd.DataFrame):
    mopj_m2 = nap_frame[nap_frame["ric"].astype(str).eq("NACFRJPSWMc2")]
    assert not mopj_m2.empty
    assert set(mopj_m2["contract_month"].astype(str)) == {"M2"}
    assert set(mopj_m2["display_name"].astype(str)) == {"日本CFR连2"}


def test_mopj_continuous_curve_has_all_twelve_contracts(nap_frame: pd.DataFrame):
    mopj = nap_frame[
        nap_frame["ric"].astype(str).str.fullmatch(r"NACFRJPSWMc(?:[1-9]|1[0-2])", na=False)
    ]
    observed = set(mopj["contract_month"].astype(str))
    expected = {f"M{month}" for month in range(1, 13)}
    if observed != expected:
        assert expected - observed == {"M3"}
        issues = list(nap_frame.attrs.get("nap_source_issues", []))
        assert any(issue.get("ric") == "NACFRJPSWMc2" for issue in issues)
    else:
        assert mopj["series_id"].nunique() == 12


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


def test_pickle_fallback_cache_migrates_to_parquet(tmp_path: Path, nap_frame: pd.DataFrame):
    cache = tmp_path / "nap_timeseries.parquet"
    fallback = cache.with_suffix(cache.suffix + ".pkl")
    nap_frame.head(20).to_pickle(fallback)
    loaded = load_nap_timeseries(
        workbook_path=tmp_path / "missing.xlsx",
        cache_path=cache,
        catalog_path=tmp_path / "missing_catalog.yaml",
    )
    assert not loaded.empty
    assert cache.exists()
    assert not fallback.exists()
