# Architecture Documentation

## System Overview

The Marktpresentatie pipeline extracts TMS (Taxatie Management Systeem) valuation data via OAuth2-protected REST APIs, loads it into a local DuckDB database through Dagster, and transforms it via dbt models into analysis-ready datasets for chart generation.

```
┌─────────────────────┐     ┌──────────────────────────────┐     ┌────────────────────┐
│   TMS REST API      │     │       Dagster Assets         │     │   DuckDB Database  │
│   (OAuth2)          │────▶│  (Extract & Load)            │────▶│   (local.duckdb)   │
│                     │     │                              │     │                    │
│ - Company List      │     │ - company_valuation_data     │     │ Schemas:           │
│ - Valuation Rounds  │     │ - tms_vastgoedgegevens       │     │  - apis            │
│ - Reports (Excel)   │     │ - tms_market_value_basis     │     │  - tms             │
│ - JSON endpoints    │     │ - tms_parameters_overview    │     │  - external        │
│                     │     │ - tms_policy_value_*         │     │  - models          │
└─────────────────────┘     │ - tms_valuation_overview     │     └────────────────────┘
                            │ - external_sources           │              │
┌─────────────────────┐     └──────────────────────────────┘              │
│ External Sources    │                                                   ▼
│ - CBS COROP regions │─────────────────────────────────────▶  ┌────────────────────┐
│ - Postal code map   │                                        │  dbt Transforms    │
│ - GeoJSON data      │                                        │                    │
└─────────────────────┘                                        │ staging → int →    │
                                                               │ marts → charts     │
                                                               └────────────────────┘
                                                                         │
                                                                         ▼
                                                               ┌────────────────────┐
                                                               │  Chart Assets      │
                                                               │  (Plotly HTML/JPEG)│
                                                               │  → grafieken/      │
                                                               └────────────────────┘
```

## Component Architecture

### 1. Dagster Orchestration Layer

**Entry point:** `src/market_presentation/definitions.py` (auto-discovers from `defs/`)

| Component | Location | Purpose |
|-----------|----------|---------|
| Config | `defs/config.py` | Shared paths, years, directories |
| Jobs | `defs/jobs.py` | Job definitions with asset selections |
| Partitions | `defs/partitions.py` | 204 company partitions |
| Resources | `defs/resources/` | OAuth2 API client, DuckDB I/O manager |
| Assets | `defs/assets/` | Individual data extraction modules |
| Schemas | `defs/schemas/` | Pandera validation schemas |
| Utils | `defs/utils/` | Shared helpers (report parsing, chart generation) |

### 2. Data Extraction Assets

Assets follow a consistent pattern:

```python
@dg.asset(
    group_name="tms_assets",
    partitions_def=company_partitions,
    deps=[COMPANY_VALUATION_DATA_DEP],
)
def asset_name(context, tms_api: Oauth2ApiResource) -> pd.DataFrame:
    # 1. Get round metadata from company_valuation_data
    # 2. Fetch report bytes/JSON via API
    # 3. Parse Excel workbook or JSON response
    # 4. Add metadata columns (Corporatie, Jaar, Peildatum)
    # 5. Return DataFrame → DuckDB I/O manager writes to tms schema
```

| Asset | API Report | DuckDB Table(s) |
|-------|-----------|-----------------|
| `company_valuation_data` | Valuation rounds | `apis.company_valuation_data` |
| `tms_vastgoedgegevens` | Vastgoedgegevens Excel | `tms.vastgoedgegevens_vhe_gegevens`, `tms.vastgoedgegevens_waarderingscomplexen` |
| `tms_valuation_overview` | Taxatieoverzicht Excel | `tms.valuation_overview_vhe`, `tms.valuation_overview_complex` |
| `tms_market_value_basis` | Market Value Basis JSON | `tms.market_value_basis` |
| `tms_market_value_parameters` | Marktwaardeparameters Excel | `tms.market_value_parameters_*` (6 sheets) |
| `tms_parameters_overview` | Parameteroverzicht Excel | `tms.parameters_overview_*` (24 sheets) |
| `tms_policy_value_report` | Beleidswaarde Excel | `tms.policy_value_report_vhe`, `tms.policy_value_report_complex` |
| `tms_policy_value_parameters` | Beleidswaarderingsparameters Excel | `tms.policy_value_parameters` |
| `tms_ratio_rapport` | Ratio Rapport Excel | `tms.ratio_rapport_vhe`, `tms.ratio_rapport_complex` |
| `tms_rental_unit_mapping` | Rental Unit Mapping JSON | `tms.rental_unit_mapping` |
| `tms_complex_kenmerken` | Complex Kenmerken JSON | `tms.complex_kenmerken` |
| `tms_complex_references` | Complex References JSON | `tms.complex_references` |
| `tms_difference_analysis` | Difference Analysis JSON | `tms.difference_analysis` |
| `external_sources` | CBS/DLL files | `external.postal_code_mapping`, `external.cbs_corop_regions` |

### 3. dbt Transformation Layer

Layered SQL transformations in `dbt/models/`:

```
sources (54 YAML definitions)
    │
    ▼
staging/ (1 model)
    │  stg_tms_percentage_full — reconstructs % Full from VHE parameters
    ▼
int/ (10 models)
    │  int_vastgoedgegevens — joins VHE + complex property data
    │  int_marktwaarde — joins valuation overview with rental unit mapping
    │  int_beleidswaarde — combines policy value report components
    │  int_marktwaardeparameters — unpivots market value parameter sheets
    │  int_parameteroverzicht_vhe — joins 10 VHE-level parameter sheets
    │  int_parameteroverzicht_complex — joins 14 complex-level parameter sheets
    │  int_parameteroverzicht — combines VHE + complex parameters
    │  int_chart_woningen_egw_mgw — splits housing by EGW/MGW for charts
    │  int_chart_ontwikkeling_woningen_matched — YoY matched housing units
    │  int_chart_ontwikkeling_woningen_egw_mgw — YoY matched by type
    ▼
marts/ (4 models)
    │  dataset_basis — central ~100-column fact table
    │  dataset_ontwikkeling — year-over-year comparison
    │  dataset_validatie — quality-filtered subset (view)
    │  dataset_validatie_aantallen — filter funnel statistics
    ▼
charts/ (~40 view models)
       chart_historisch__*, chart_validatie__*, chart_waarde__*, etc.
```

### 4. Chart Generation Layer

Chart assets in `defs/assets/chart_*.py` read from dbt mart/chart views and produce Plotly HTML/JPEG output to `grafieken/`.

**Year convention:**
- `ISSUE_YEARS = [2024, 2025]` — raw data years
- `CHART_YEAR = 2025` — most recent valuation year
- `CHART_YEAR_M1 = 2024` — prior year for YoY comparisons
- `dbt var('jaar') = 2025` — must equal `CHART_YEAR`

## Database Schema

| Schema | Purpose | Writer |
|--------|---------|--------|
| `apis` | API metadata (company list, valuation data) | Dagster assets |
| `tms` | Raw TMS report tables (Dutch column names) | Dagster assets |
| `external` | Reference data (postal codes, COROP regions) | Dagster assets |
| `models` | dbt-transformed tables/views | dbt |

## Authentication & API Flow

```
1. OAuth2 client credentials grant → Bearer token
2. Token auto-refreshes on 401 responses
3. Report flow: trigger → poll status → download bytes
4. Retry with exponential backoff on 5xx errors
```

## Execution Model

- **Dagster executor:** `multiprocess` with `max_concurrent = 1`
- **dbt threads:** 2 (parallel independent model builds)
- **Partition fan-out:** One partition per company (204 companies)
- **Job ordering:** external_sources → tms_assets → dbt_transformations → charts
