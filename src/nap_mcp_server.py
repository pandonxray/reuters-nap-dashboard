from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from .nap_mcp_service import (
    DEFAULT_POINTS_LIMIT,
    MAX_POINTS_HARD_LIMIT,
    MAX_SEARCH_LIMIT,
    NaphthaDataStore,
    about,
    metric_dictionary,
)


logger = logging.getLogger(__name__)


class MCPResponse(BaseModel):
    """Stable response envelope shared by every Naphtha MCP tool."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    ok: bool
    data: Any
    meta: dict[str, Any]


class SpreadLeg(BaseModel):
    """One weighted leg in a custom spread."""

    series_id: str = Field(min_length=1, description="Exact series_id returned by search_series")
    weight: float = Field(default=1.0, description="Finite non-zero multiplier")


mcp = MCPServer(
    name="reuters-naphtha-dashboard",
    title="Reuters Naphtha Dashboard",
    description="Local, source-aware access to the Reuters NAP workbook and Dashboard analytics.",
    instructions=(
        "Call get_data_status before time-sensitive analysis. Use search_series to resolve exact series_id values. "
        "Every response separates the workbook modification time from the latest market trade date. "
        "The server reads local Reuters/LSEG Excel values and never claims to refresh upstream Reuters formulas."
    ),
    version="1.0.0",
)

store = NaphthaDataStore.from_environment()


def _response(payload: dict[str, Any]) -> MCPResponse:
    return MCPResponse.model_validate(payload)


def _json_resource(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


@mcp.resource(
    "naphtha://about",
    name="about",
    title="Naphtha Dashboard MCP architecture",
    description="Architecture, source boundary, refresh behavior, authentication, and environment configuration.",
    mime_type="application/json",
)
def about_resource() -> str:
    return _json_resource(about())


@mcp.resource(
    "naphtha://metrics",
    name="metric-dictionary",
    title="Naphtha metric dictionary",
    description="Definitions and units for raw, normalized, snapshot, curve, risk, relationship, and seasonal metrics.",
    mime_type="application/json",
)
def metrics_resource() -> str:
    return _json_resource(metric_dictionary())


@mcp.resource(
    "naphtha://source-status",
    name="source-status",
    title="Naphtha source and freshness status",
    description="Live workbook/cache coverage, latest trade date, and high-signal data-quality checks.",
    mime_type="application/json",
)
def source_status_resource() -> str:
    return _json_resource(store.data_status())


@mcp.resource(
    "naphtha://series/{series_id}",
    name="series-detail",
    title="Naphtha series detail",
    description="Catalog metadata, coverage, latest values, units, Reuters RIC, and research explanation for one series.",
    mime_type="application/json",
)
def series_resource(series_id: str) -> str:
    return _json_resource(store.series_detail(series_id))


@mcp.tool(
    name="get_data_status",
    description=(
        "Check workbook/cache freshness, coverage, source counts, duplicates, future dates, and stale series. "
        "force_reparse only rereads the local Excel; it does not refresh Reuters/LSEG formulas."
    ),
    structured_output=True,
)
def get_data_status(force_reparse: bool = False) -> MCPResponse:
    return _response(store.data_status(force_reparse=force_reparse))


@mcp.tool(
    name="get_metric_dictionary",
    description="Return exact definitions, units, windows, and caveats for Dashboard/MCP metrics.",
    structured_output=True,
)
def get_metric_dictionary() -> MCPResponse:
    return MCPResponse(
        ok=True,
        data=metric_dictionary(),
        meta={"canonical_resource": "naphtha://metrics", "dictionary_version": "1.0.0"},
    )


@mcp.tool(
    name="search_series",
    description=(
        "Search the normalized Reuters series catalog. Text tokens are ANDed across names, RIC, sector, product, region, and sheet. "
        "Use the returned exact series_id in data tools."
    ),
    structured_output=True,
)
def search_series(
    query: str = "",
    sector: str | None = None,
    product: str | None = None,
    region: str | None = None,
    term_type: Literal["continuous", "calendar"] | None = None,
    contract_month: str | None = None,
    calendar_month: Annotated[int | None, Field(ge=1, le=12)] = None,
    include_explanations: bool = False,
    limit: Annotated[int, Field(ge=1, le=MAX_SEARCH_LIMIT)] = 50,
) -> MCPResponse:
    return _response(
        store.search_series(
            query=query,
            sector=sector,
            product=product,
            region=region,
            term_type=term_type,
            contract_month=contract_month,
            calendar_month=calendar_month,
            include_explanations=include_explanations,
            limit=limit,
        )
    )


@mcp.tool(
    name="get_timeseries",
    description=(
        "Query one or more exact series over a date range in native or normalized units. "
        "Defaults to the latest 365 calendar days and refuses silent point truncation."
    ),
    structured_output=True,
)
def get_timeseries(
    series_ids: Annotated[list[str], Field(min_length=1, max_length=20)],
    start_date: str | None = None,
    end_date: str | None = None,
    value_basis: Literal["native", "normalized"] = "normalized",
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_points: Annotated[int, Field(ge=1, le=MAX_POINTS_HARD_LIMIT)] = DEFAULT_POINTS_LIMIT,
) -> MCPResponse:
    return _response(
        store.get_timeseries(
            series_ids,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            frequency=frequency,
            max_points=max_points,
        )
    )


@mcp.tool(
    name="get_market_snapshot",
    description=(
        "Return latest normalized values, absolute changes, 60-day z-score, 250-day percentile, 20-day volatility, "
        "and curve-structure label for matching series."
    ),
    structured_output=True,
)
def get_market_snapshot(
    query: str = "",
    sector: str | None = None,
    product: str | None = None,
    region: str | None = None,
    term_type: Literal["continuous", "calendar"] | None = None,
    contract_month: str | None = None,
    calendar_month: Annotated[int | None, Field(ge=1, le=12)] = None,
    limit: Annotated[int, Field(ge=1, le=MAX_SEARCH_LIMIT)] = 100,
) -> MCPResponse:
    return _response(
        store.market_snapshot(
            query=query,
            sector=sector,
            product=product,
            region=region,
            term_type=term_type,
            contract_month=contract_month,
            calendar_month=calendar_month,
            limit=limit,
        )
    )


@mcp.tool(
    name="list_curve_groups",
    description="Discover unambiguous sector/product/region/family_key combinations before requesting a forward curve.",
    structured_output=True,
)
def list_curve_groups(
    query: str = "",
    sector: str | None = None,
    product: str | None = None,
    region: str | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_SEARCH_LIMIT)] = 100,
) -> MCPResponse:
    return _response(store.list_curve_groups(query=query, sector=sector, product=product, region=region, limit=limit))


@mcp.tool(
    name="get_forward_curve",
    description=(
        "Return one continuous/calendar curve on the nearest available date at or before as_of, plus M1-M2/M1-M3/M1-M6 spreads. "
        "Call list_curve_groups first when family_key is unknown."
    ),
    structured_output=True,
)
def get_forward_curve(
    sector: str,
    product: str,
    region: str,
    family_key: str | None = None,
    as_of: str | None = None,
    value_basis: Literal["native", "normalized"] = "normalized",
) -> MCPResponse:
    return _response(
        store.forward_curve(
            sector=sector,
            product=product,
            region=region,
            family_key=family_key,
            as_of=as_of,
            value_basis=value_basis,
        )
    )


@mcp.tool(
    name="calculate_spread",
    description=(
        "Build a weighted, date-aligned spread from exact series IDs. Incompatible units are rejected and excess points are never silently truncated."
    ),
    structured_output=True,
)
def calculate_spread(
    legs: Annotated[list[SpreadLeg], Field(min_length=1, max_length=8)],
    start_date: str | None = None,
    end_date: str | None = None,
    value_basis: Literal["native", "normalized"] = "normalized",
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_points: Annotated[int, Field(ge=1, le=MAX_POINTS_HARD_LIMIT)] = DEFAULT_POINTS_LIMIT,
) -> MCPResponse:
    return _response(
        store.calculate_spread(
            [leg.model_dump() for leg in legs],
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            frequency=frequency,
            max_points=max_points,
        )
    )


@mcp.tool(
    name="analyze_series",
    description="Return latest/change/z-score/percentile, risk metrics, and monthly seasonality statistics for one exact series.",
    structured_output=True,
)
def analyze_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    value_basis: Literal["native", "normalized"] = "normalized",
    seasonal_years: Annotated[int, Field(ge=1, le=20)] = 10,
) -> MCPResponse:
    return _response(
        store.analyze_series(
            series_id,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            seasonal_years=seasonal_years,
        )
    )


@mcp.tool(
    name="compare_series",
    description=(
        "Compare two exact series using aligned observations, level/change correlation, rolling correlation/beta, residual z-score, and lead-lag scan."
    ),
    structured_output=True,
)
def compare_series(
    series_a: str,
    series_b: str,
    start_date: str | None = None,
    end_date: str | None = None,
    value_basis: Literal["native", "normalized"] = "normalized",
    window: Annotated[int, Field(ge=10, le=500)] = 60,
) -> MCPResponse:
    return _response(
        store.compare_series(
            series_a,
            series_b,
            start_date=start_date,
            end_date=end_date,
            value_basis=value_basis,
            window=window,
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
