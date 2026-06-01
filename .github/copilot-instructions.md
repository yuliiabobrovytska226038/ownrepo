# Copilot Instructions for rev-market-presentation

## Project Overview
This project builds the **Marktpresentatie** (Market Presentation) pipeline for Ortec Finance REV. It extracts TMS (Taxatie Management Systeem) valuation data via REST APIs, loads it into a local DuckDB database via Dagster, and transforms it through dbt models into analysis-ready datasets. Uses Python 3.12 and UV package manager.

## Architecture
```
TMS REST API → Dagster assets (Python) → DuckDB raw tables (tms/external schemas)
                                          ↓
                               dbt models (staging → int → marts)
                                          ↓
                               DuckDB models schema (dataset_basis, dataset_validatie, etc.)
```

### Key Components
- **Dagster** (`src/market_presentation/defs/`): Asset definitions for API extraction + external data loading
- **dbt** (`dbt/`): SQL transformation models (69 models: 1 staging, 10 intermediate, 4 marts, 54 chart views)
- **DuckDB** (`database/local.duckdb`): Local analytical database with schemas: `tms`, `external`, `models`, `apis`
- **OAuth2 API client** (`src/market_presentation/defs/resources/oauth2_api.py`): TMS API wrapper with retry/polling

### Data Sources
- **TMS API**: vastgoedgegevens, marktwaardeparameters, beleidswaarde, parameteroverzicht (24 sheets), market_value_basis, valuation_overview, policy_value_report, complex_kenmerken, complex_referenties
- **External files**: `data/extern/DLL/` (postal_code_mapping), `data/extern/cbs/` (CBS COROP regions)
- **Company partitions**: 204 companies via `StaticPartitionsDefinition`

## Launching And Validation
- Start the local Dagster UI with `uv run dg dev`
- List registered TMS assets with `uv run dg list defs --assets "group:tms_assets" --json`
- Materialize external sources: `uv run dg launch --job external_sources_job`
- Materialize one live company partition with `uv run dg launch --job tms_assets_job --partition "<company>"`
- Run dbt transformations through Dagster: `uv run dg launch --job dbt_transformations_job`
- Render chart outputs through Dagster: `uv run dg launch --job charts_job`
- Materialize all assets: `uv run dg launch --job all_assets_job`
- Prefer the dedicated `tms_assets_job` job over `dg launch --assets ...` for live TMS validation, because dbt source-test checks can otherwise introduce a circular asset graph during launch
- Run dbt build (fallback only): `cd dbt && uv run dbt build --profiles-dir . --project-dir .`
- Validate the raw-table compatibility layer with `uv run pytest tests/test_tms_workbook_validation.py tests/test_tms_legacy_raw_compatibility.py`


### Materialization Order
1. External sources (postal_code_mapping, cbs_corop_regions) — no partition required
2. TMS validation assets per company partition (fetches all raw API data)
3. dbt transformations via Dagster: `uv run dg launch --job dbt_transformations_job`
4. Chart assets via Dagster: `uv run dg launch --job charts_job`

**Important**: Prefer `uv run dg launch --job dbt_transformations_job` over `cd dbt && uv run dbt build` for running dbt models. The Dagster job ensures correct execution context, I/O manager integration, and asset metadata tracking. Only use `dbt build` directly for local debugging or when Dagster is unavailable.

## dbt Models
- **Staging**: `stg_tms_percentage_full` — reconstructs the removed "Percentage Full" sheet from VHE parameters
- **Intermediate**: `int_vastgoedgegevens`, `int_marktwaardeparameters`, `int_beleidswaarde`, `int_marktwaarde`, `int_parameteroverzicht` (+ `_vhe` / `_complex` sub-models)
- **Marts**: `dataset_basis` (central ~100 column fact table), `dataset_validatie`, `dataset_validatie_aantallen`, `dataset_ontwikkeling` (YoY comparison)
- dbt variable `jaar` in `dbt_project.yml` controls the valuation year and must equal `CHART_YEAR` from `src/market_presentation/defs/config.py` (currently 2025)

### Known Data Quirks
- Raw TMS API tables store most numeric values as VARCHAR (from Excel-style API responses). Use `TRY_CAST(col AS DOUBLE)` in dbt when doing arithmetic or COALESCE with numeric defaults.
- `market_value_basis` uses `rentalUnitInternalId` (TMS internal ID), not `VHE-nr`; join through `tms.rental_unit_mapping` when comparing or projecting VHE-level market-basis fields. The primary market value comes from `valuation_overview_vhe."Netto marktwaarde"`.
- Parameteroverzicht has 24 separate sheets (10 VHE-level + 14 complex-level) that are FULL OUTER JOINed in int models.
- External tables use normalized lowercase column names (via `normalize_column_names()` in the asset code).

## Code Style & Standards
- Follow PEP 8 with Ruff linting/formatting rules from `pyproject.toml`
- Line length: 200 characters
- Use double quotes for strings
- Use type hints for function signatures
- Use descriptive variable names following snake_case
- Allow unused imports in `__init__.py` files

## Project Structure
- `src/market_presentation/defs/`: Dagster definitions (assets, resources, jobs, partitions)
- `src/market_presentation/defs/assets/`: Individual asset modules (one per TMS report type)
- `src/market_presentation/defs/resources/`: OAuth2 API client, DuckDB I/O manager
- `dbt/models/`: dbt SQL models (staging/, int/, marts/)
- `dbt/models/sources.yml`: Source definitions for tms and external schemas
- `data/extern/`: External reference data (CBS, DLL, etc.)
- `database/`: Local DuckDB database
- `dagster_home/`: Dagster instance config and run storage
- `tests/`: Pytest unit tests
- `scripts/`: Utility scripts (benchmark_performance.py, Start-PgTunnel-Mirror.ps1)

## Development Practices
- Use UV package manager for dependency management (not pip)
- Target Python 3.12
- Include proper logging using `logging_utils.init_logging()`
- Handle exceptions with proper error messages and logging
- Use environment variables for configuration (see `environment.py`)
- Required env vars: `TMS_CLIENT_ID`, `TMS_CLIENT_SECRET`, `TMS_TOKEN_URL`, `TMS_BASE_URL`

## Testing
- Write unit tests with pytest in `tests/` directory
- Follow existing test patterns in `test_example.py`
- Tests should be runnable from VS Code Testing window

## Package Management
- Add dependencies to `pyproject.toml` under `[project.dependencies]`
- Add dev dependencies to `[dependency-groups.dev]`
- All packages should reference the private Ortec Finance Nexus feed
- Ensure `UV_INDEX_OFNEXUS_USERNAME` and `UV_INDEX_OFNEXUS_PASSWORD` environment variables are set

## When Generating Code
- Use structured logging with `python-json-logger`
- Include exception handling with traceback
- Follow the logging pattern from `main.py`
- Use type hints consistently
- Ensure code passes Ruff linting checks
- For new Dagster assets, follow the pattern in `src/market_presentation/defs/assets/` (use `@dg.asset`, `company_partitions`, `Oauth2ApiResource`)
- For new dbt models, follow the layered approach: staging (clean/rename) → int (join/combine) → marts (business logic)
