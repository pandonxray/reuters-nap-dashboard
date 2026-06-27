from __future__ import annotations

import numpy as np
import pandas as pd


SEASONAL_YEAR = 2001


def _ensure_datetime_index(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data.index, pd.DatetimeIndex):
        return data.sort_index()

    coerced_index = pd.to_datetime(data.index, errors="coerce")
    valid_mask = ~pd.isna(coerced_index)
    cleaned = data.loc[valid_mask].copy()
    cleaned.index = pd.DatetimeIndex(coerced_index[valid_mask])
    return cleaned.sort_index()


def remove_feb29(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    mask = ~((df.index.month == 2) & (df.index.day == 29))
    return df.loc[mask]


def _continuous_seasonal_index() -> pd.Index:
    return pd.date_range(f"{SEASONAL_YEAR}-01-01", f"{SEASONAL_YEAR}-12-31", freq="D").strftime("%m-%d")


def seasonal_matrix(series: pd.Series, years: int = 5, interpolate: bool = True) -> pd.DataFrame:
    s = _ensure_datetime_index(series.dropna())
    if s.empty:
        return pd.DataFrame()

    cutoff_year = s.index.max().year - years + 1
    s = s[s.index.year >= cutoff_year]

    frame = pd.DataFrame({"value": s.values}, index=s.index)
    frame["year"] = frame.index.year
    frame["doy"] = frame.index.strftime("%m-%d")
    matrix = frame.pivot(index="doy", columns="year", values="value")
    matrix = matrix.reindex(_continuous_seasonal_index())

    if interpolate:
        # Only fill gaps between known observations; keep future tail gaps empty.
        matrix = matrix.interpolate(method="linear", limit_area="inside")

    return matrix


def seasonal_stats(series: pd.Series, years: int = 5) -> dict[str, float]:
    series = _ensure_datetime_index(series.dropna())
    matrix = seasonal_matrix(series, years=years)
    if matrix.empty:
        return {"seasonal_percentile": np.nan, "seasonal_deviation": np.nan}

    latest = series
    if latest.empty:
        return {"seasonal_percentile": np.nan, "seasonal_deviation": np.nan}

    today = latest.index.max().strftime("%m-%d")
    row = matrix.loc[today].dropna() if today in matrix.index else pd.Series(dtype=float)
    current = latest.iloc[-1]
    if row.empty:
        return {"seasonal_percentile": np.nan, "seasonal_deviation": np.nan}

    percentile = float((row <= current).mean() * 100)
    deviation = float(current - row.mean())
    return {"seasonal_percentile": percentile, "seasonal_deviation": deviation}


def seasonal_percentile_band(
    series: pd.Series,
    years: int = 10,
    lower: float = 10,
    upper: float = 90,
    interpolate: bool = True,
) -> pd.DataFrame:
    matrix = seasonal_matrix(series, years=years, interpolate=interpolate)
    if matrix.empty:
        return pd.DataFrame(columns=["doy", "lower", "median", "upper"])
    band = pd.DataFrame(
        {
            "doy": matrix.index,
            "lower": matrix.quantile(lower / 100.0, axis=1),
            "median": matrix.median(axis=1),
            "upper": matrix.quantile(upper / 100.0, axis=1),
        }
    )
    return band.reset_index(drop=True)


def monthly_box_frame(series: pd.Series, years: int = 10) -> pd.DataFrame:
    s = _ensure_datetime_index(series.dropna())
    if s.empty:
        return pd.DataFrame(columns=["date", "year", "month", "value"])
    cutoff_year = s.index.max().year - years + 1
    s = s[s.index.year >= cutoff_year]
    return pd.DataFrame(
        {
            "date": s.index,
            "year": s.index.year,
            "month": s.index.month,
            "value": s.values,
        }
    )


def calendar_heatmap_frame(series: pd.Series, year: int | None = None) -> pd.DataFrame:
    s = _ensure_datetime_index(series.dropna())
    if s.empty:
        return pd.DataFrame(columns=["date", "week", "weekday", "value"])
    selected_year = int(year or s.index.max().year)
    s = s[s.index.year == selected_year]
    if s.empty:
        return pd.DataFrame(columns=["date", "week", "weekday", "value"])
    iso = s.index.isocalendar()
    return pd.DataFrame(
        {
            "date": s.index,
            "week": iso.week.astype(int).to_numpy(),
            "weekday": s.index.weekday,
            "value": s.values,
        }
    )
