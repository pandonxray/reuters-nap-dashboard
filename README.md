# Reuters NAP Dashboard

Standalone Streamlit + Plotly research terminal for Reuters NAP multi-commodity daily time series.

## Run

```powershell
pip install -r requirements.txt
streamlit run src/nap_dashboard.py
```

Default workbook path:

```text
C:\Users\74100\Nutstore\1\油气-djx-\NAP-丙烯-坚果云\Nap.xlsx
```

You can override the workbook, cache, and catalog paths from the sidebar.

## Data Refresh

`src/nap_adapter.py` detects Reuters Excel output from `_xll.RDP.HistoricalPricing` and `_xll.RHistory`, parses wide output into a long table, and caches to:

```text
data/processed/nap_timeseries.parquet
```

If parquet dependencies are unavailable, the adapter writes a `.pkl` fallback beside the parquet path.

Manual refresh:

```powershell
python -m src.nap_adapter --refresh
```

## Catalog

`config/nap_series_catalog.yaml` is generated from the workbook and can be edited manually. Supported fields include `display_name`, `sector`, `product`, `region`, `contract_month`, `unit_native`, `unit_normalized`, and `ric`.

## Explanations

`config/nap_explanations.yaml` powers the Series Detail explanation card and the Price Glossary. Add exact `series` entries for instrument-level notes or edit `defaults` by sector/product.

## Formula Registry

`config/nap_formula_registry.yaml` defines reusable spreads for Relationship Lab. Legs can point to exact `series_id`s or selector fields such as `product`, `region`, `contract_month`, `ric`, and `contains`.

Included examples:

- RBOB - WTI
- HO - WTI
- LSGO - Brent
- Singapore 92R crack
- Hi-5 = VLSFO - HSFO
- Refinery margin vs crude

## Validation

```powershell
pytest -q
```

The local validation workbook currently detects 554 series, including 160 Crude series and 120 Crk series, with no duplicate `(date, series_id)` rows.
