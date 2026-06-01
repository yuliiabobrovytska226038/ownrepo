# Operational Runbook

Step-by-step guide for running the Marktpresentatie pipeline.

## Prerequisites

### Environment Setup

1. **Python 3.12** installed locally
2. **UV package manager** installed (`pip install uv` or via installer)
3. **Environment variables** configured (copy `.env.example` to `.env` and fill in secrets):

| Variable | Purpose |
|----------|---------|
| `DAGSTER_HOME` | Path to Dagster instance directory (default: `dagster_home`) |
| `TMS_TOKEN_URL` | Keycloak OAuth2 token endpoint |
| `TMS_CLIENT_ID` | OAuth2 client ID |
| `TMS_CLIENT_SECRET` | OAuth2 client secret |
| `TMS_USER` | TMS API username |
| `TMS_PASS` | TMS API password |
| `TMS_API_BASE_URL` | TMS API base URL |
| `VGR_API_BASE_URL` | VGR API base URL |
| `UV_INDEX_OFNEXUS_USERNAME` | Nexus feed username |
| `UV_INDEX_OFNEXUS_PASSWORD` | Nexus feed password |

4. **Install dependencies:**
   ```powershell
   uv sync
   ```

### Verify Setup

```powershell
# Check Python version
uv run python --version  # Should be 3.12.x

# Check dbt connection
cd dbt
uv run dbt debug --profiles-dir . --project-dir .
cd ..

# List available assets
uv run dg list defs --assets --json | Select-Object -First 20
```

---

## Standard Pipeline Run

### Step 1: Start Dagster UI (Optional)

```powershell
uv run dg dev
```

Open http://localhost:3000 to monitor runs via the web UI.

### Step 2: Materialize External Sources

Load reference data (postal codes, COROP regions). No partition required.

```powershell
uv run dg launch --job external_sources_job
```

**Expected output:** 2 assets materialized (`postal_code_mapping`, `cbs_corop_regions`).

### Step 3: Materialize TMS Data per Company

Fetch all raw API data for a specific company:

```powershell
# Single company
uv run dg launch --job tms_assets_job --partition "3B Wonen"

# Multiple companies (one at a time due to max_concurrent=1)
uv run dg launch --job tms_assets_job --partition "Acantus"
uv run dg launch --job tms_assets_job --partition "Woonforte"
```

**Expected output:** ~15 assets materialized per company (company_valuation_data + all TMS report tables).

**Duration:** 5-15 minutes per company depending on portfolio size.

### Step 4: Run dbt Transformations

Build all dbt models (staging → int → marts):

```powershell
uv run dg launch --job dbt_transformations_job
```

**Expected output:** ~30 dbt models built (staging, intermediate, marts, chart views).

**Duration:** 2-5 minutes depending on data volume.

### Step 5: Generate Charts

```powershell
uv run dg launch --job charts_job
```

**Expected output:** HTML and JPEG chart files written to `grafieken/`.

### Step 6: Verify Results

```powershell
# Run dbt tests
cd dbt
uv run dbt test --profiles-dir . --project-dir .
cd ..

# Run Python unit tests
uv run pytest tests/ -v

# Quick database check via DuckDB CLI
uv run python -c "import duckdb; db = duckdb.connect('database/local.duckdb'); print(db.sql('SHOW ALL TABLES').fetchdf())"
```

---

## Full Reset and Rebuild

When you need to start from scratch:

```powershell
# Manual reset: clear DB, Dagster storage, dbt target
Remove-Item database/local.duckdb -ErrorAction SilentlyContinue
Remove-Item dagster_home/storage -Recurse -ErrorAction SilentlyContinue
Remove-Item dagster_home/history -Recurse -ErrorAction SilentlyContinue
Remove-Item dbt/target -Recurse -ErrorAction SilentlyContinue

# Rebuild single company
uv run dg launch --job tms_assets_job --partition "3B Wonen"
uv run dg launch --job dbt_transformations_job

# Rebuild all companies (WARNING: can take hours)
uv run dg launch --job tms_assets_job
```

---

## Job Reference

| Job | Command | Partitioned | Purpose |
|-----|---------|:-----------:|---------|
| `external_sources_job` | `uv run dg launch --job external_sources_job` | No | Load CBS/DLL reference data |
| `tms_assets_job` | `uv run dg launch --job tms_assets_job --partition "<company>"` | Yes | Fetch all TMS data for a company |
| `dbt_transformations_job` | `uv run dg launch --job dbt_transformations_job` | No | Run dbt staging → marts |
| `charts_job` | `uv run dg launch --job charts_job` | No | Generate Plotly charts |
| `company_list_job` | `uv run dg launch --job company_list_job` | No | Refresh company list from API |

---

## Adding a New Company

1. Add the company name to `INCLUDED_COMPANIES` in `src/market_presentation/defs/partitions.py`
2. Materialize: `uv run dg launch --job tms_assets_job --partition "New Company Name"`
3. Rebuild dbt: `uv run dg launch --job dbt_transformations_job`
4. Regenerate charts: `uv run dg launch --job charts_job`

---

## Adding a New Valuation Year

1. Update `ISSUE_YEARS` in `src/market_presentation/defs/config.py` (e.g., `[2023, 2024, 2025]`)
2. Update `vars.jaar` in `dbt/dbt_project.yml` to match `max(ISSUE_YEARS)`
3. Re-materialize all companies for the new year
4. Rebuild dbt transformations and charts

---

## Monitoring

### Dagster UI

Start with `uv run dg dev` and check:
- **Runs** tab: status of each job execution
- **Assets** tab: materialization status per asset
- **Partitions** tab: which companies have been materialized

### Database Inspection

```powershell
# Quick schema overview
uv run python -c "import duckdb; db = duckdb.connect('database/local.duckdb'); print(db.sql('SHOW ALL TABLES').fetchdf())"

# Table row counts
uv run python -c "import duckdb; db = duckdb.connect('database/local.duckdb'); print(db.sql('SELECT schema_name, table_name, estimated_size FROM duckdb_tables() ORDER BY schema_name, table_name').fetchdf())"
```

### Performance Benchmarking

```powershell
uv run python scripts/benchmark_performance.py
uv run python scripts/benchmark_performance.py --compare-legacy
```
