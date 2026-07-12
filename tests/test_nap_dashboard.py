import numpy as np
import pandas as pd
import pytest

from src.nap_analytics import available_curve_groups, build_core_market_matrix, curve_contract_catalog
from src.nap_dashboard import (
    _append_continuous_crude,
    _build_converted_spread,
    _build_data_health,
    _complete_contract_curve,
    _continuous_month_mapping,
    _custom_structure_frame,
    _custom_spread_title,
    _market_contract_view,
    _optimize_frame_memory,
    _weekly_convert_panels,
)
from src.unit_conversion import convert_quote_values


def test_market_map_defaults_to_m1_and_m2_for_continuous_contracts():
    snapshot = pd.DataFrame(
        {
            "series_id": ["m1", "m2", "m3", "spot"],
            "term_type": ["continuous"] * 4,
            "contract_month": ["M1", "M2", "M3", ""],
        }
    )
    filtered = _market_contract_view(snapshot, include_other_contracts=False)
    assert filtered["series_id"].tolist() == ["m1", "m2"]
    assert len(_market_contract_view(snapshot, include_other_contracts=True)) == 4


def test_calendar_market_view_keeps_selected_natural_month_rows():
    snapshot = pd.DataFrame(
        {
            "series_id": ["jan", "feb"],
            "term_type": ["calendar", "calendar"],
            "contract_month": ["", ""],
        }
    )
    assert len(_market_contract_view(snapshot, include_other_contracts=False)) == 2


def test_dashboard_memory_compaction_preserves_string_values():
    frame = pd.DataFrame({"series_id": ["a", "a", "b"], "term_type": ["continuous"] * 3, "value": [1.0, 2.0, 3.0]})
    compact = _optimize_frame_memory(frame)
    assert compact["series_id"].astype(str).tolist() == ["a", "a", "b"]
    assert compact["value"].tolist() == [1.0, 2.0, 3.0]


def test_core_market_matrix_aligns_m1_m2_and_calculates_structure():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    wide = pd.DataFrame(
        {
            "m1": np.arange(80, dtype=float) + 100,
            "m2": np.arange(80, dtype=float) + 98,
            "m3": np.arange(80, dtype=float) + 96,
            "explicit_spread": np.arange(80, dtype=float),
        },
        index=dates,
    )
    catalog = pd.DataFrame(
        {
            "series_id": ["m1", "m2", "m3", "explicit_spread"],
            "sector": ["Naphtha"] * 4,
            "product": ["MOPJ"] * 4,
            "region": ["Japan"] * 4,
            "contract_month": ["M1", "M2", "M3", "M1"],
            "term_type": ["continuous"] * 4,
            "display_name": ["MOPJ M1", "MOPJ M2", "MOPJ M3", "MOPJ M1-M2"],
            "ric": ["MOPJMc1", "MOPJMc2", "MOPJMc3", "MOPJMc1-MOPJMc2"],
            "unit_native": ["USD/mt"] * 4,
            "unit_normalized": ["USD/bbl"] * 4,
        }
    )
    matrix = build_core_market_matrix(catalog, wide)
    assert len(matrix) == 1
    row = matrix.iloc[0]
    assert row["m1"] == 179
    assert row["m2"] == 177
    assert row["m1_m2"] == 2
    assert row["m1_m3"] == 4
    assert row["structure"] == "backwardation"


def test_data_health_accepts_categorical_metadata_and_flags_future_dates():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp.now().normalize() + pd.Timedelta(days=2)],
            "series_id": ["a"],
            "display_name": ["A"],
            "sheet": ["Nap"],
            "sector": ["Naphtha"],
            "product": ["MOPJ"],
            "region": ["Japan"],
            "unit_native": [None],
            "unit_normalized": [None],
            "unit_conversion": [None],
            "unit_source": [None],
            "ric": [None],
            "term_type": ["continuous"],
            "contract_month": ["M1"],
            "calendar_month": [None],
            "is_derived": [False],
            "source": ["test"],
            "name_native": ["A"],
            "short_name": ["A"],
            "value": [1.0],
        }
    )
    for column in ["series_id", "sheet", "sector", "product", "region", "unit_native", "ric", "term_type"]:
        frame[column] = frame[column].astype("category")
    health = _build_data_health(frame)
    assert health["future_dates"] == 1
    assert health["series"].iloc[0]["RIC状态"] == "缺失"


def test_forward_curve_groups_keep_futures_and_swaps_separate():
    catalog = pd.DataFrame(
        {
            "series_id": ["future_m1", "future_m2", "swap_m1", "swap_m2", "spread"],
            "sector": ["Crude"] * 5,
            "product": ["Brent"] * 5,
            "region": ["Europe"] * 5,
            "contract_month": ["M1", "M2", "M1", "M2", "M1"],
            "term_type": ["continuous"] * 5,
            "display_name": ["Brent连1", "Brent连2", "Brent Swap连1", "Brent Swap连2", "Brent M1-M2"],
            "ric": ["LCOc1", "LCOc2", "BRTCALAMc1", "BRTCALAMc2", "LCOc1-LCOc2"],
        }
    )
    groups = available_curve_groups(catalog)
    assert len(groups) == 2
    assert sorted(groups["count"].tolist()) == [2, 2]
    futures_key = groups.loc[groups["quote"].eq("Brent"), "family_key"].iloc[0]
    futures = curve_contract_catalog(catalog, "Crude", "Brent", "Europe", futures_key)
    assert futures["series_id"].tolist() == ["future_m1", "future_m2"]


def test_regional_display_uses_mt_outside_us_and_bbl_for_us_quotes():
    naphtha = pd.Series([89.0])
    converted, unit, _ = convert_quote_values(
        naphtha,
        {"sector": "Naphtha", "product": "Naphtha", "region": "Japan", "unit_native": "USD/mt"},
        "regional",
    )
    assert unit == "USD/mt"
    assert converted.iloc[0] == 89.0

    rbob = pd.Series([2.5])
    converted, unit, _ = convert_quote_values(
        rbob,
        {"sector": "Gasoline", "product": "RBOB", "region": "US", "unit_native": "USD/gal", "ric": "RBc1"},
        "regional",
    )
    assert unit == "USD/bbl"
    assert converted.iloc[0] == 105.0


def test_non_us_crude_can_be_displayed_in_metric_tons_with_editable_factor():
    converted, unit, formula = convert_quote_values(
        pd.Series([80.0]),
        {"sector": "Crude", "product": "Brent", "region": "Global", "unit_native": "USD/bbl"},
        "regional",
        {"crude": 7.5},
    )
    assert unit == "USD/mt"
    assert converted.iloc[0] == 600.0
    assert "7.5" in formula


def test_us_heating_oil_ric_is_treated_as_us_even_when_region_is_global():
    converted, unit, _ = convert_quote_values(
        pd.Series([2.4]),
        {"sector": "Jet/Heating Oil", "product": "Heating Oil", "region": "Global", "unit_native": "USD/gal", "ric": "HOc1"},
        "regional",
    )
    assert unit == "USD/bbl"
    assert converted.iloc[0] == 100.8


def test_calendar_detail_view_appends_continuous_crude_only():
    calendar = pd.DataFrame(
        {"series_id": ["nap_jan"], "sector": ["Naphtha"], "term_type": ["calendar"], "value": [1.0]}
    )
    raw = pd.DataFrame(
        {
            "series_id": ["nap_jan", "brent_m1", "nap_m1"],
            "sector": ["Naphtha", "Crude", "Naphtha"],
            "term_type": ["calendar", "continuous", "continuous"],
            "value": [1.0, 80.0, 70.0],
        }
    )
    combined = _append_continuous_crude(calendar, raw, "calendar")
    assert set(combined["series_id"]) == {"nap_jan", "brent_m1"}
    assert combined.attrs["nap_crude_fallback"] is True


def test_custom_spread_converts_each_leg_before_subtracting():
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    wide = pd.DataFrame({"gas": [100.0] * 3, "nap": [890.0] * 3}, index=dates)
    catalog = pd.DataFrame(
        {
            "series_id": ["gas", "nap"],
            "sector": ["Gasoline", "Naphtha"],
            "product": ["Singapore 92", "MOPJ"],
            "region": ["Singapore", "Japan"],
            "display_name": ["Singapore 92 M1", "MOPJ M1"],
            "ric": ["SG92Mc1", "NACFRJPMc1"],
            "unit_native": ["USD/bbl", "USD/mt"],
        }
    )
    per_barrel, unit, _ = _build_converted_spread(wide, catalog, [("gas", 1.0), ("nap", -1.0)], "bbl", {})
    assert unit == "USD/bbl"
    assert per_barrel.iloc[-1] == pytest.approx(0.0)

    per_ton, unit, _ = _build_converted_spread(wide, catalog, [("gas", 1.0), ("nap", -1.0)], "mt", {})
    assert unit == "USD/mt"
    assert per_ton.iloc[-1] == pytest.approx(-57.0)


def test_custom_spread_builds_matching_continuous_contract_structure():
    dates = pd.date_range("2026-01-01", periods=65, freq="B")
    wide = pd.DataFrame(
        {
            "gas_m1": np.arange(65) + 100.0,
            "gas_m2": np.arange(65) + 98.0,
            "nap_m1": (np.arange(65) + 90.0) * 8.9,
            "nap_m2": (np.arange(65) + 89.0) * 8.9,
        },
        index=dates,
    )
    catalog = pd.DataFrame(
        {
            "series_id": ["gas_m1", "gas_m2", "nap_m1", "nap_m2"],
            "sector": ["Gasoline", "Gasoline", "Naphtha", "Naphtha"],
            "product": ["Singapore 92", "Singapore 92", "MOPJ", "MOPJ"],
            "region": ["Singapore", "Singapore", "Japan", "Japan"],
            "contract_month": ["M1", "M2", "M1", "M2"],
            "term_type": ["continuous"] * 4,
            "display_name": ["Singapore 92 M1", "Singapore 92 M2", "MOPJ M1", "MOPJ M2"],
            "ric": ["SG92Mc1", "SG92Mc2", "NACFRJPMc1", "NACFRJPMc2"],
            "unit_native": ["USD/bbl", "USD/bbl", "USD/mt", "USD/mt"],
        }
    )
    curve, series_by_month, unit = _custom_structure_frame(
        wide,
        catalog,
        [("gas_m1", 1.0), ("nap_m1", -1.0)],
        "bbl",
        {},
    )
    assert curve["合约"].tolist() == [f"M{month}" for month in range(1, 13)]
    assert set(series_by_month) == {"M1", "M2"}
    assert unit == "USD/bbl"
    assert curve.set_index("合约").loc["M1", "价差"] == pytest.approx(10.0)
    assert curve.set_index("合约").loc["M2", "价差"] == pytest.approx(9.0)
    assert curve.set_index("合约").loc["M3":, "价差"].isna().all()


def test_continuous_month_mapping_uses_latest_data_month_and_rolls_year():
    mapping = _continuous_month_mapping(pd.Timestamp("2026-07-13"))
    indexed = mapping.set_index("contract_month")
    assert indexed.loc["M1", "natural_month_label"] == "2026-07"
    assert indexed.loc["M6", "natural_month_label"] == "2026-12"
    assert indexed.loc["M7", "natural_month_label"] == "2027-01"
    assert indexed.loc["M12", "natural_month_label"] == "2027-06"


def test_complete_contract_curve_keeps_all_month_slots_and_missing_gaps():
    curve = pd.DataFrame(
        {
            "合约": ["M1", "M2", "M4"],
            "价差": [74.0, 69.0, 63.0],
            "最新日期": [pd.Timestamp("2026-07-13")] * 3,
        }
    )
    completed, asof = _complete_contract_curve(curve, contract_column="合约")
    assert completed["contract_month"].tolist() == [f"M{month}" for month in range(1, 13)]
    assert completed.loc[completed["contract_month"].eq("M3"), "价差"].isna().all()
    assert completed.loc[completed["contract_month"].eq("M4"), "价差"].iloc[0] == 63.0
    assert asof == pd.Timestamp("2026-07-13")


def test_weekly_panel_unit_conversion_uses_declared_factor():
    panels = [{"title": "route", "series": pd.Series([1.0, 2.0]), "factor_key": "crude"}]
    converted = _weekly_convert_panels(panels, "mt", {"crude": 7.33})
    assert converted[0]["series"].tolist() == pytest.approx([7.33, 14.66])
    assert panels[0]["series"].tolist() == [1.0, 2.0]


def test_custom_crack_spread_honors_requested_metric_ton_unit():
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    wide = pd.DataFrame({"crack_a": [10.0, 11.0], "crack_b": [8.0, 8.5]}, index=dates)
    catalog = pd.DataFrame(
        {
            "series_id": ["crack_a", "crack_b"],
            "sector": ["Cracks", "Cracks"],
            "product": ["LSGO-Brent Crack", "LSGO-Brent Crack"],
            "region": ["Europe", "Europe"],
            "display_name": ["LSGO crack A", "LSGO crack B"],
            "ric": ["LGOCMc1", "LGOCMc2"],
            "unit_native": ["USD/bbl", "USD/bbl"],
        }
    )
    spread, unit, _ = _build_converted_spread(
        wide,
        catalog,
        [("crack_a", 1.0), ("crack_b", -1.0)],
        "mt",
        {"gasoil": 7.45},
    )
    assert unit == "USD/mt"
    assert spread.iloc[-1] == pytest.approx((11.0 - 8.5) * 7.45)


def test_custom_spread_title_uses_selected_leg_names_and_weights():
    catalog = pd.DataFrame(
        {
            "series_id": ["a", "b"],
            "product": ["欧洲低硫柴油/LSGO", "日本CFR石脑油"],
            "region": ["欧洲", "日本/东北亚"],
            "contract_month": ["M1", "M1"],
        }
    )
    title = _custom_spread_title(catalog, [("a", 1.0), ("b", -1.0)])
    assert title == "欧洲低硫柴油/LSGO / 欧洲 / M1 - 日本CFR石脑油 / 日本/东北亚 / M1"
