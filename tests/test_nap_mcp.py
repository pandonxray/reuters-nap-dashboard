from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from src.nap_mcp_service import NaphthaDataStore, NaphthaInputError


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=120, freq="B")
    definitions = [
        {
            "series_id": "nap_m1",
            "display_name": "Japan CFR Naphtha M1",
            "value": 620.0 + np.arange(len(dates)) * 0.5,
            "value_normalized": (620.0 + np.arange(len(dates)) * 0.5) / 8.9,
            "contract_month": "M1",
            "unit_native": "USD/mt",
            "unit_normalized": "USD/bbl",
            "ric": "NACFRJPSWMc1",
        },
        {
            "series_id": "nap_m2",
            "display_name": "Japan CFR Naphtha M2",
            "value": 615.0 + np.arange(len(dates)) * 0.5,
            "value_normalized": (615.0 + np.arange(len(dates)) * 0.5) / 8.9,
            "contract_month": "M2",
            "unit_native": "USD/mt",
            "unit_normalized": "USD/bbl",
            "ric": "NACFRJPSWMc2",
        },
        {
            "series_id": "brent_m1",
            "display_name": "Brent M1",
            "value": 75.0 + np.arange(len(dates)) * 0.03,
            "value_normalized": 75.0 + np.arange(len(dates)) * 0.03,
            "contract_month": "M1",
            "unit_native": "USD/bbl",
            "unit_normalized": "USD/bbl",
            "ric": "LCOc1",
            "sector": "Crude",
            "product": "Brent",
            "sheet": "Crude",
            "region": "Europe",
        },
    ]
    frames = []
    for definition in definitions:
        values = definition.pop("value")
        normalized = definition.pop("value_normalized")
        frame = pd.DataFrame(
            {
                "date": dates,
                "series_id": definition["series_id"],
                "display_name": definition["display_name"],
                "value": values,
                "sheet": definition.get("sheet", "Nap"),
                "sector": definition.get("sector", "Naphtha"),
                "product": definition.get("product", "Naphtha"),
                "region": definition.get("region", "Japan"),
                "contract_month": definition["contract_month"],
                "term_type": "continuous",
                "calendar_month": "",
                "unit_native": definition["unit_native"],
                "unit_normalized": definition["unit_normalized"],
                "ric": definition["ric"],
                "is_derived": False,
                "source": "RDP.HistoricalPricing",
                "value_normalized": normalized,
                "unit_conversion": "test conversion",
                "unit_source": "test",
                "name_native": definition["display_name"],
                "short_name": definition["display_name"],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def store(sample_frame: pd.DataFrame) -> NaphthaDataStore:
    return NaphthaDataStore.from_frame(sample_frame)


def test_search_and_series_detail_are_source_aware(store: NaphthaDataStore):
    result = store.search_series(query="naphtha", contract_month="M1")
    assert result["ok"] is True
    assert result["meta"]["total_matches"] == 1
    assert result["data"][0]["series_id"] == "nap_m1"

    detail = store.series_detail("nap_m1")
    assert detail["data"]["coverage"]["observations"] == 120
    assert detail["data"]["unit_native"] == "USD/mt"
    assert detail["meta"]["latest_trade_date"] == "2026-03-17"


def test_timeseries_validates_range_and_never_silently_truncates(store: NaphthaDataStore):
    result = store.get_timeseries(
        ["nap_m1", "nap_m2"],
        start_date="2026-01-01",
        end_date="2026-03-17",
        frequency="weekly",
        max_points=100,
    )
    assert result["meta"]["points"] <= 100
    assert {row["series_id"] for row in result["data"]} == {"nap_m1", "nap_m2"}
    assert all(row["unit"] == "USD/bbl" for row in result["data"])

    with pytest.raises(NaphthaInputError, match="start_date must be on or before"):
        store.get_timeseries(["nap_m1"], start_date="2026-03-01", end_date="2026-01-01")
    with pytest.raises(NaphthaInputError, match="above max_points"):
        store.get_timeseries(["nap_m1", "nap_m2"], max_points=2)


def test_spread_aligns_legs_and_rejects_incompatible_native_units(store: NaphthaDataStore):
    result = store.calculate_spread(
        [{"series_id": "nap_m1", "weight": 1}, {"series_id": "nap_m2", "weight": -1}],
        value_basis="normalized",
        frequency="monthly",
    )
    assert result["meta"]["unit"] == "USD/bbl"
    assert result["data"]
    assert result["meta"]["summary"]["latest"] == pytest.approx(5 / 8.9)

    with pytest.raises(NaphthaInputError, match="incompatible native units"):
        store.calculate_spread(
            [{"series_id": "nap_m1", "weight": 1}, {"series_id": "brent_m1", "weight": -1}],
            value_basis="native",
        )


def test_market_curve_analysis_and_relationship_outputs(store: NaphthaDataStore):
    snapshot = store.market_snapshot(sector="Naphtha", contract_month="M1")
    assert snapshot["data"][0]["structure"] == "backwardation"

    groups = store.list_curve_groups(sector="Naphtha")
    assert groups["meta"]["total_matches"] == 1
    group = groups["data"][0]
    curve = store.forward_curve(
        sector=group["sector"],
        product=group["product"],
        region=group["region"],
        family_key=group["family_key"],
    )
    assert [row["contract_month"] for row in curve["data"]] == ["M1", "M2"]
    assert curve["meta"]["spreads"]["M1-M2"] == pytest.approx(5 / 8.9)

    analysis = store.analyze_series("nap_m1", seasonal_years=2)
    assert analysis["data"]["summary"]["observations"] == 120
    assert len(analysis["data"]["seasonality"]["monthly"]) >= 5

    comparison = store.compare_series("nap_m1", "nap_m2", window=20)
    assert comparison["data"]["observations"] == 120
    assert comparison["data"]["level_correlation"] == pytest.approx(1.0)


def test_data_status_separates_trade_date_from_source_transport(store: NaphthaDataStore):
    result = store.data_status()
    assert result["data"]["coverage"]["latest_trade_date"] == "2026-03-17"
    assert result["data"]["quality"]["duplicate_date_series_rows"] == 0
    assert result["data"]["network_used_by_mcp"] is False
    assert result["meta"]["source_id"] == "reuters_lseg_excel_workbook"


def test_mcp_in_memory_smoke_call_returns_structured_content(monkeypatch: pytest.MonkeyPatch, store: NaphthaDataStore):
    import src.nap_mcp_server as server

    monkeypatch.setattr(server, "store", store)

    async def smoke() -> None:
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            templates = await client.list_resource_templates()
            assert "search_series" in {tool.name for tool in tools.tools}
            assert "naphtha://metrics" in {str(resource.uri) for resource in resources.resources}
            assert "naphtha://series/{series_id}" in {str(template.uri_template) for template in templates.resource_templates}

            result = await client.call_tool("search_series", {"query": "naphtha", "limit": 5})
            assert result.is_error is False
            assert result.structured_content is not None
            assert result.structured_content["ok"] is True
            assert result.structured_content["data"][0]["series_id"] == "nap_m1"

            detail = await client.read_resource("naphtha://series/nap_m1")
            assert detail.contents

    asyncio.run(smoke())
