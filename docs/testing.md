# Phase 3 — Testing Implementation

## Overview

Phase 3 implements comprehensive testing across seven areas:
1. Parallel testing with the legacy system
2. dbt schema tests
3. dbt referential integrity tests
4. dbt business logic tests

---

## 3.1 Parallel Testing with Legacy System

Parallel testing between the legacy `Marktpresentatie/data/tms.db` and the new `database/local.duckdb` pipeline.

### 3.1.1 Legacy baseline run

The legacy DuckDB database at `Marktpresentatie/data/tms.db` serves as the baseline reference. Comparison scripts generate legacy outputs:

- `scripts/compare_legacy.py` — general legacy comparison
- `scripts/compare_legacy_2024.py` — 2024-specific comparison
- `scripts/compare_raw_tables.py` — raw table-level comparison

### 3.1.2 New system comparison run

The new pipeline outputs to `database/local.duckdb`. Additional comparison scripts produce new-system outputs:

- `scripts/compare_transformations.py` — dbt transformation comparison
- `scripts/compare_chart_data.py` — chart data comparison
- `scripts/compare_all_charts.py` — full chart output comparison

### 3.1.3 Row-level comparison

`tests/test_legacy_validation.py` contains 5 test classes:

| Class | Purpose |
|-------|---------|
| `TestRawTableRowCounts` | Row count comparison for 18 table pairs × 3 validation companies |
| `TestRawTableColumns` | Column name/presence comparison between legacy and new |
| `TestDbtModelRowCounts` | Row count comparison for 7 dbt model tables |
| `TestAggregateDbtModels` | Row count and column comparison for aggregate models (e.g., dataset_validatie_aantallen) |
| `TestKeyMetrics` | Aggregate metric sums (e.g., total market value) |

Run with:
```powershell
uv run pytest tests/test_legacy_validation.py -v
```

### 3.1.4 Investigate discrepancies

Known differences are documented in:
- `EXPECTED_RAW_COVERAGE_DIFFERENCES` dict in `test_legacy_validation.py` (e.g., Acantus policy-value gap)
- `.github/RESULT_PARITY_PLAN.md` — root cause analysis for each known discrepancy

### 3.1.5 Resolve discrepancies

Fixes applied:
- `market_value_basis` population filter via `rental_unit_mapping` join
- `% Full` reconstruction in `dbt/models/staging/stg_tms_percentage_full.sql`
- Remaining known gaps tracked in `.github/RESULT_PARITY_PLAN.md`

---

## 3.2 dbt Join Key & Schema Tests

All dbt tests are implemented as **singular SQL tests** in `dbt/tests/`. YAML-based generic tests were removed because DuckDB column names containing special characters (hyphens, percentages, parentheses) cannot be properly quoted in dbt's YAML test framework.

Each singular SQL test is a query that returns violating rows — a test passes when zero rows are returned.

### 3.2.1 not_null tests (join keys)

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/key_not_null_market_value_basis.sql` | rentalUnitInternalId, Corporatie, Jaar never null in market_value_basis |
| `dbt/tests/key_not_null_policy_value_parameters.sql` | VHE-nr, Corporatie, Peildatum never null in policy_value_parameters |
| `dbt/tests/key_not_null_policy_value_report_vhe.sql` | VHE-nr, Corporatie, Peildatum never null in policy_value_report_vhe |
| `dbt/tests/key_not_null_rental_unit_mapping.sql` | rentalUnitInternalId, VHE-nr, Corporatie, Jaar never null in rental_unit_mapping |
| `dbt/tests/key_not_null_valuation_overview_vhe.sql` | VHE-nr, Corporatie, Jaar never null in valuation_overview_vhe |
| `dbt/tests/key_not_null_vastgoedgegevens_vhe.sql` | VHE-nummer, Corporatie, Jaar never null in vastgoedgegevens_vhe_gegevens |

### 3.2.2 unique tests (composite keys)

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/key_unique_market_value_basis.sql` | rentalUnitInternalId + Corporatie + Jaar is unique in market_value_basis |
| `dbt/tests/key_unique_policy_value_parameters.sql` | VHE-nr + Corporatie + Peildatum is unique in policy_value_parameters |
| `dbt/tests/key_unique_policy_value_report_vhe.sql` | VHE-nr + Corporatie + Peildatum is unique in policy_value_report_vhe |
| `dbt/tests/key_unique_rental_unit_mapping.sql` | rentalUnitInternalId + Corporatie + Jaar is unique in rental_unit_mapping |
| `dbt/tests/key_unique_valuation_overview_vhe.sql` | VHE-nr + Corporatie + Jaar is unique in valuation_overview_vhe |
| `dbt/tests/key_unique_vastgoedgegevens_vhe.sql` | VHE-nummer + Corporatie + Jaar is unique in vastgoedgegevens_vhe_gegevens |
| `dbt/tests/key_unique_postal_code_mapping.sql` | Postcode is unique in postal_code_mapping |
| `dbt/tests/key_unique_cbs_corop_regions.sql` | Gemeentecode is unique in cbs_corop_regions |
| `dbt/tests/key_unique_int_marktwaarde.sql` | VHE-nr + Corporatie + Jaar is unique in int_marktwaarde output |
| `dbt/tests/key_unique_int_beleidswaarde.sql` | VHE-nr + Corporatie + Jaar is unique in int_beleidswaarde output |
| `dbt/tests/key_unique_int_parameteroverzicht.sql` | VHE-nr + Corporatie + Peildatum is unique in int_parameteroverzicht output |
| `dbt/tests/key_unique_int_marktwaardeparameters.sql` | VHE-nr + Corporatie + Jaar is unique in int_marktwaardeparameters output |
| `dbt/tests/key_unique_stg_percentage_full.sql` | VHE-nr + Corporatie + Jaar is unique in stg_tms_percentage_full |

### 3.2.3 accepted_values tests

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/schema_accepted_values_waarderingsmodel.sql` | Waarderingsmodel only contains valid values (Woningen, BOG/MOG/ZOG, Parkeren, Intramuraal, etc.) |
| `dbt/tests/schema_accepted_values_waarderingstype.sql` | Waarderingstype only Basis, Full, or Vrije waardering |
| `dbt/tests/biz_waarderingstype_allowed_values.sql` | Waarderingstype allowed values cross-check |

### 3.2.4 not_null tests (schema columns)

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/schema_not_null_dataset_basis.sql` | Peildatum, Waarderingsmodel, Waarderingstype never null |
| `dbt/tests/schema_not_null_dataset_validatie.sql` | Peildatum, Handboektype never null |
| `dbt/tests/schema_not_null_dataset_ontwikkeling.sql` | Peildatum, Waarderingsmodel never null (excludes right_only merge rows) |
| `dbt/tests/schema_not_null_int_vastgoedgegevens.sql` | Key schema columns never null in int_vastgoedgegevens |
| `dbt/tests/schema_dataset_validatie_aantallen.sql` | All count columns (Corporaties, Werkmaatschappijen, Complexen, VHEs) never null |

### 3.2.5 Range & binary flag tests

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/biz_discount_rate_range.sql` | Disconteringsvoet between 2% and 15% |
| `dbt/tests/biz_marktwaarde_non_negative.sql` | Market value ≥ 0 |
| `dbt/tests/biz_pct_full_binary.sql` | % Full is strictly 0 or 1 (queries `stg_tms_percentage_full`) |
| `dbt/tests/biz_pct_vrije_waardering_binary.sql` | % Vrije waardering is strictly 0 or 1 (queries `stg_tms_percentage_full`) |
| `dbt/tests/schema_binary_flags_dataset_basis.sql` | Binary flag columns in dataset_basis are 0 or 1 |

---

## 3.3 dbt Referential Integrity Tests

### 3.3.1 Relationships tests (foreign key checks)

| Test File | Relationship |
|-----------|-------------|
| `dbt/tests/ref_ontwikkeling_subset_of_basis.sql` | Every VHE in dataset_ontwikkeling must exist in dataset_basis (joined on VHE-nr + Corporatie + Jaar) |
| `dbt/tests/ref_marktwaarde_has_vastgoedgegevens.sql` | Every VHE in int_marktwaarde must exist in int_vastgoedgegevens (severity: warn) |
| `dbt/tests/ref_beleidswaarde_has_vastgoedgegevens.sql` | Every VHE in int_beleidswaarde must exist in int_vastgoedgegevens (severity: warn) |
| `dbt/tests/ref_parameteroverzicht_has_vastgoedgegevens.sql` | Every VHE in int_parameteroverzicht must exist in int_vastgoedgegevens (severity: warn) |

### 3.3.2 Source freshness tests

Source freshness tests are not currently active. The YAML-based freshness configuration was removed due to incompatibility with DuckDB column quoting in the Dagster execution context.

### 3.3.3 Orphan record tests

| Test File | What it checks |
|-----------|---------------|
| `dbt/tests/ref_orphan_rental_unit_mapping.sql` | rental_unit_mapping VHEs without matching vastgoedgegevens rows (severity: warn, threshold: 100) |
| `dbt/tests/biz_orphan_vhe_valuation_overview.sql` | Orphan VHEs in valuation overview without matching property data |

---

## 3.4 dbt Business Logic Tests

All business logic tests are singular SQL tests in `dbt/tests/`. They return rows that violate the business rule (test passes when 0 rows returned).

### 3.4.1 Market value calculations

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_marktwaarde_non_negative.sql` | Market value must be ≥ 0 |
| `dbt/tests/biz_beleidswaarde_leq_marktwaarde.sql` | Beleidswaarde ≤ Marktwaarde (policy value cannot exceed market value; tolerance: €100) |

### 3.4.2 Policy value calculations

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_policy_value_components_sum.sql` | `Beleidswaarde = Marktwaarde − Beschikbaarheid_totaal − Betaalbaarheid − Kwaliteit_totaal − Beheer` (tolerance: ±€10) |

### 3.4.3 Parameter aggregations

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_parameteroverzicht_coverage.sql` | Woningen VHEs in dataset_basis have matching rows in int_parameteroverzicht (VHE join completeness) |
| `dbt/tests/biz_aantallen_monotonic.sql` | Validation counts are monotonically decreasing (filter funnel: total > woningen > validated) |
| `dbt/tests/biz_vrije_waardering_count_consistency.sql` | Free-valuation counts are internally consistent |

### 3.4.4 Geographic mappings

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_geographic_mapping_complete.sql` | When Postcode exists, the full chain Gemeentecode → COROP-gebied → Provincies must be populated (no null gaps) |

### 3.4.5 Percentage calculations

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_pct_full_binary.sql` | % Full is strictly 0 or 1 (queries `stg_tms_percentage_full`) |
| `dbt/tests/biz_pct_vrije_waardering_binary.sql` | % Vrije waardering is strictly 0 or 1 (queries `stg_tms_percentage_full`) |
| `dbt/tests/biz_pct_full_matches_count.sql` | % Full = 1 when "Aantal vrijheidsgraden toegepast" > 0, and 0 otherwise |

### 3.4.6 Year-over-year calculations

| Test File | Business Rule |
|-----------|--------------|
| `dbt/tests/biz_yoy_marktwaarde_reasonable.sql` | YoY market value change within −50% to +100% (flags extreme outliers as warnings) |
| `dbt/tests/biz_yoy_beleidswaarde_reasonable.sql` | YoY beleidswaarde change within −50% to +100% (flags extreme outliers as warnings) |
| `dbt/tests/biz_validatie_subset_of_basis.sql` | dataset_validatie is a proper subset of dataset_basis |
| `dbt/tests/biz_validatie_woningen_only.sql` | dataset_validatie contains only Woningen model rows |
| `dbt/tests/biz_beleidswaarde_discount_rate_range.sql` | Beleidswaarde discount rate between 2% and 10% |
| `dbt/tests/biz_percentage_full_range.sql` | Policy value components (Beschikbaarheid, Kwaliteit, Betaalbaarheid, Beheer) are non-negative |

---

## 3.5 Python Unit Tests

Unit tests for the core Python modules, targeting 50%+ coverage on `resources/` and `utils/`.

### 3.5.1 TMS API client tests

| Test File | What it tests |
|-----------|--------------|
| `tests/test_oauth2_api.py` | `_issue_date_to_iso` helper, `normalize_company_valuation_data` edge cases, `_BearerTokenRefreshAuth` token injection + 401 retry, `Oauth2ApiResource` methods (get_company_list, get_raw_company_list, get_raw_valuation_rounds, request_with_retry backoff/RemoteProtocolError, fetch_market_value_parameters, fetch_vgr_dataset_json, fetch_vgr_valuation_complexes, fetch_policy_value_parameters, fetch_complex_references, fetch_property_info, fetch_energy_performance, fetch_ratios, fetch_complex_characteristics, fetch_complex_characteristics_excel, fetch_waterfall_analysis, download_vgr_import_file, get_vgr_datasets, get_company_valuation_data end-to-end) |

### 3.5.2 Data aggregation tests

| Test File | What it tests |
|-----------|--------------|
| `tests/test_tms_report_utils.py` | `RoundMetadata` properties, `get_round_metadata` deduplication/filtering, `add_metadata_columns`, `coerce_numeric_columns` (identifier/prefix preservation, threshold logic), `concat_frames`, `load_or_fetch_bytes`/`load_or_fetch_json` caching, `build_raw_output_path` naming, `validate_dataframe`, `align_to_schema_columns` |

### 3.5.3 Graph generation tests

| Test File | What it tests |
|-----------|--------------|
| `tests/test_graph_utils.py` | `of_template` registration (brand colors, font, separators), `save_graph` HTML output + error handling (empty figure, empty DataFrame, image export failure), `set_template_axis_format`, `set_bar_text_horizontal`, `set_line_marker_style`, `chart_appender`, `save_charts` iteration |

### 3.5.4 Dagster resources tests

| Test File | What it tests |
|-----------|--------------|
| `tests/test_dagster_resources.py` | `normalize_column_names` (camelCase→snake, Dutch→English, dedup), `enforce_column_types` (Int64/float64/str coercion, error handling), DuckDB IO manager single-quote escape monkey-patch, `RESOURCE_DEFS` structure validation, config constants consistency |

### 3.5.5 Coverage achieved

| Module | Coverage |
|--------|----------|
| `resources/oauth2_api.py` | 45% |
| `resources/resources.py` | 95% |
| `utils/graph_utils.py` | 73% |
| `utils/tms_report_utils.py` | 54% |
| `utils/tms_utils.py` | 100% |
| **TOTAL (focused scope)** | **56%** |

Coverage configuration (in `pyproject.toml`):
```toml
[tool.coverage.run]
source = ["src/market_presentation"]
omit = [
    "src/market_presentation/defs/schemas/*",
    "src/market_presentation/defs/assets/tms_*",
    "src/market_presentation/defs/assets/chart_*",
    "src/market_presentation/defs/assets/external_*",
    "src/market_presentation/defs/jobs.py"
]

[tool.coverage.report]
fail_under = 50
```

---

## 3.6 Performance Optimization

### 3.6.1 Benchmark query performance

`scripts/benchmark_performance.py` measures execution time for 10 representative DuckDB queries:

| Query | What it measures |
|-------|-----------------|
| Row count dataset_basis | Full table scan count |
| Row count dataset_ontwikkeling | Full table scan count |
| Aggregate market value by model | GROUP BY with SUM on large table |
| Filter by corporatie | Indexed lookup performance |
| Complex join basis+ontwikkeling | Multi-table join |
| Geographic grouping | GROUP BY with multiple aggregates |
| Parameteroverzicht row count | Large 10-JOIN intermediate table |
| Year-over-year comparison | Filtered aggregation |
| Distinct corporaties | Cardinality query |
| Full dataset_validatie scan | View-based filtered scan |

Run with:
```powershell
uv run python scripts/benchmark_performance.py
```

### 3.6.2 Compare with legacy performance

The benchmark script supports `--compare-legacy` mode to compare against a legacy DuckDB. Use `--legacy-db` to specify the path (defaults to `Marktpresentatie/data/tms.db`). It runs the same 10 queries on both databases and flags any query where the new system is >10% slower.

```powershell
uv run python scripts/benchmark_performance.py --compare-legacy
uv run python scripts/benchmark_performance.py --compare-legacy --legacy-db "C:\path\to\legacy.db"
```

Additionally, `dbt/target/run_results.json` (generated after each `dbt run`) contains per-model execution times. The benchmark script parses this and reports the 5 slowest models.

### 3.6.3 Optimize slow queries

Applied optimizations:

| Optimization | Models Affected | Rationale |
|-------------|----------------|-----------|
| View materialization | `int_vastgoedgegevens`, `int_beleidswaarde`, `int_marktwaarde` | Only consumed by dbt tests (not downstream models), so views eliminate redundant table writes during build |
| View materialization | `dataset_validatie` | Pure filter on `dataset_basis`; avoids duplicating data on disk |
| DuckDB index on (Corporatie, Jaar) | `dataset_basis` | Most downstream queries filter/group by corporation and year |
| DuckDB index on (Corporatie) | `dataset_ontwikkeling` | Chart assets and exports commonly filter by corporation |

Performance-critical models and their characteristics:

| Model | Build Time | Bottleneck |
|-------|-----------|-----------|
| `int_marktwaardeparameters` | ~90s | Large source table with complex unpivot logic |
| `int_parameteroverzicht` | ~74s | Combines 24 sheets via sequential FULL OUTER JOINs |
| `int_parameteroverzicht_vhe` | ~36s | 10 VHE-level sheets joined |
| `stg_tms_percentage_full` | ~4s | Reconstructs % Full from VHE parameters |

### 3.6.4 Incremental models assessment

**Decision: Not applicable for this pipeline.**

Rationale:
- All source data is fully reloaded per company partition via Dagster (no append-only sources)
- `dataset_basis` depends on 10+ FULL OUTER JOINs across tables that are completely replaced each run
- `dataset_ontwikkeling` is a FULL OUTER JOIN of two year-slices of `dataset_basis` — both sides change on rebuild
- DuckDB's in-process architecture means the overhead of incremental bookkeeping (tracking `loaded_at`, managing merge keys) exceeds the benefit for datasets under 1M rows

Instead, performance is optimized through:
1. **View materialization** — 4 models converted from TABLE to VIEW, eliminating disk writes
2. **DuckDB indexes** — 2 indexes added for common filter/group patterns
3. **Dagster concurrency** — `max_concurrent: 1` in dagster.yaml prevents DuckDB write contention
4. **dbt thread concurrency** — 2 threads in profiles.yml for parallel independent model builds

---

## Running All Tests

```powershell
# dbt tests (all 52 singular SQL tests)
cd dbt
uv run dbt test --profiles-dir . --project-dir .

# Python unit tests
uv run pytest tests/test_oauth2_api.py tests/test_tms_report_utils.py tests/test_graph_utils.py tests/test_dagster_resources.py -v

# Python unit tests with coverage
uv run pytest tests/test_oauth2_api.py tests/test_tms_report_utils.py tests/test_graph_utils.py tests/test_dagster_resources.py --cov=src/market_presentation --cov-report=term-missing

# Performance benchmark
uv run python scripts/benchmark_performance.py

# Performance benchmark with legacy comparison
uv run python scripts/benchmark_performance.py --compare-legacy --legacy-db "path/to/legacy.db"
```

## Test Inventory Summary

| Category | Count | Location |
|----------|-------|----------|
| dbt singular SQL tests | 52 | `dbt/tests/*.sql` |
| Python unit tests | 117 tests | `tests/test_oauth2_api.py`, `tests/test_tms_report_utils.py`, `tests/test_graph_utils.py`, `tests/test_dagster_resources.py` |
| **TOTAL automated tests** | **169** | |
| Comparison scripts | 6 | `scripts/compare_*.py` |
| Performance benchmark | 10 queries | `scripts/benchmark_performance.py` |

dbt test breakdown (52 total):
- Join key not_null: 6 tests
- Join key unique: 13 tests
- Schema not_null: 4 tests
- Schema accepted_values: 3 tests
- Range checks: 4 tests
- Business rules: 11 tests (+ 5 with severity: warn)
- Referential integrity: 6 tests
- Uniqueness: 1 test (biz_no_join_fanout)
- Other business logic: 4 tests

---

## 3.7 Documentation

### 3.7.1 Architecture documentation

Created [docs/architecture.md](architecture.md) covering:
- System overview diagram (API → Dagster → DuckDB → dbt → Charts)
- Component architecture (orchestration, extraction, transformation, charting)
- Data extraction asset inventory (14 asset types, their API sources, and DuckDB tables)
- dbt transformation layer (staging → int → marts → charts with model counts)
- Database schema reference (apis, tms, external, models)
- Authentication flow and execution model

### 3.7.2 dbt documentation

Generated via `dbt docs generate`:
- `dbt/target/catalog.json` (292 KB) — database catalog with table/column metadata
- `dbt/target/manifest.json` (2.5 MB) — full project manifest with model definitions
- `dbt/target/index.html` (1.7 MB) — browsable documentation site

Serve locally with:
```powershell
cd dbt
uv run dbt docs serve --profiles-dir . --project-dir .
```

### 3.7.3 Operational runbook

Created [docs/runbook.md](runbook.md) covering:
- Prerequisites and environment setup
- Standard pipeline run (5-step process)
- Full reset and rebuild procedures
- Job reference table
- Adding new companies and valuation years
- Monitoring via Dagster UI and database inspection

### 3.7.4 Troubleshooting guide

Created [docs/troubleshooting.md](troubleshooting.md) covering:
- Installation issues (UV cache, module not found, pytest)
- Dagster issues (frozen Pydantic, stuck runs, circular dependencies)
- dbt issues (missing tables, profile paths, deprecation warnings)
- API issues (OAuth2 tokens, report timeouts, empty responses)
- Data issues (Pandera validation, duplicate rows, single-quote escaping)
- Performance issues (slow models, DuckDB locks)

### 3.7.5 Data dictionary

Created [docs/data_dictionary.md](data_dictionary.md) covering:
- Database schema inventory (4 schemas)
- All `tms` schema tables (30+ tables across property data, valuations, policy value, parameters)
- All `external` schema tables (6 reference tables)
- All `models` schema objects (staging, intermediate, marts, chart views)
- Business term glossary (35+ Dutch → English translations)
- Column naming conventions

### 3.7.6 README update

Updated [README.md](../README.md) with:
- Project title and description
- Quick start guide (4-step)
- Architecture overview diagram
- Documentation index linking all docs
- Key commands reference
- Project structure tree
- Environment variables table
- Technology stack

### 3.7.7 dbt model documentation

All 69 dbt models (staging, intermediate, marts, charts) already have:
- Model-level `description:` in their YAML files
- Column-level `description:` for all key columns
- Test definitions (not_null, unique, accepted_values, custom) where applicable

Source definitions (54 YAML files) include:
- Table-level descriptions
- Column-level descriptions
- Source freshness configuration (vastgoedgegevens_vhe_gegevens)
