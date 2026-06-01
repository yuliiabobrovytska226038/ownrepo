# Troubleshooting Guide

Common issues and solutions for the Marktpresentatie pipeline.

---

## Installation & Environment

### UV cache permission error

**Symptom:**
```
Failed to download `gitpython==3.1.50`
failed to rename file: Access is denied. (os error 5)
```

**Fix:**
```powershell
uv cache clean
uv sync
```

If `uv sync` then fails with `failed to remove file ruff.exe: Access is denied`, close any terminals with an activated venv and retry, or use `uv run --no-sync` to skip venv updates.

### Module not found: `market_presentation`

**Symptom:** `ModuleNotFoundError: No module named 'market_presentation'`

**Fix:** Ensure the project is installed in development mode:
```powershell
uv sync
# or
uv pip install -e .
```

### pytest not found

**Symptom:** `Failed to spawn: pytest` or `No module named pytest`

**Fix:**
```powershell
uv pip install pytest pytest-cov
# Then run with:
uv run --no-sync pytest tests/ -v
```

---

## Dagster Issues

### `DagsterInvalidInvocationError: frozen=True`

**Symptom:** When monkeypatching `Oauth2ApiResource` in tests:
```
'Oauth2ApiResource' is a Pythonic resource and does not support item assignment,
as it inherits from 'pydantic.BaseModel' with frozen=True.
```

**Fix:** Monkeypatch on the **class**, not the instance:
```python
# Wrong:
monkeypatch.setattr(resource, "method_name", fake_method)

# Correct:
monkeypatch.setattr(Oauth2ApiResource, "method_name", fake_method)
```

### Dagster run stuck at `STARTING`

**Symptom:** Run shows `STARTING` status indefinitely in the UI.

**Fix:**
1. Check if another run is already executing (`max_concurrent_runs: 1` in dagster.yaml)
2. Cancel stuck runs via the UI
3. If persists, reset Dagster storage:
   ```powershell
   Remove-Item dagster_home/storage -Recurse -ErrorAction SilentlyContinue
   Remove-Item dagster_home/history -Recurse -ErrorAction SilentlyContinue
   ```

### Asset check circular dependency

**Symptom:** `DagsterInvalidDefinitionError: circular asset graph` when using `dg launch --assets`

**Fix:** Use the dedicated job instead:
```powershell
# Wrong — can trigger circular graph with dbt source-test checks:
uv run dg launch --assets "tms_vastgoedgegevens"

# Correct — uses .without_checks() to avoid the circular dependency:
uv run dg launch --job tms_assets_job --partition "3B Wonen"
```

---

## dbt Issues

### `Table with name X does not exist`

**Symptom:**
```
Catalog Error: Table with name int_marktwaardeparameters does not exist!
```

**Cause:** Upstream models haven't been built yet.

**Fix:** Use `+` prefix to build dependencies:
```powershell
cd dbt
# Build model + all upstream dependencies
uv run dbt run --profiles-dir . --project-dir . --select +dataset_basis

# Or build everything
uv run dbt run --profiles-dir . --project-dir .
```

### `Table with name postal_code_mapping does not exist`

**Symptom:** dbt run fails on `dataset_basis` referencing `external.postal_code_mapping`.

**Cause:** External sources haven't been loaded by Dagster.

**Fix:** Materialize external sources first:
```powershell
uv run dg launch --job external_sources_job
```

### dbt profile path error

**Symptom:**
```
Cannot open file "c:\users\dennisa\git\...\local.duckdb"
```

**Cause:** Hardcoded path in `dbt/profiles.yml`.

**Fix:** Ensure `profiles.yml` uses a relative path:
```yaml
dbt:
  target: local
  outputs:
    local:
      type: duckdb
      path: "../database/local.duckdb"
      threads: 2
```

### dbt deprecation warnings

**Symptom:**
```
MissingArgumentsPropertyInGenericTestDeprecation: 6 occurrences
```

**Impact:** Warning only, tests still work. Will need migration when dbt 2.0 is released.

**Fix (when ready):** Nest test arguments under `arguments:` property in YAML.

---

## API Issues

### OAuth2 token failure

**Symptom:** `401 Unauthorized` or token fetch errors.

**Fix:**
1. Verify env vars are set: `echo $env:TMS_CLIENT_ID`
2. Check token URL is accessible: `curl $env:TMS_TOKEN_URL`
3. Verify credentials haven't expired

### Report trigger timeout

**Symptom:** Report polling times out after many retries.

**Cause:** Large company portfolios can take 10+ minutes to generate reports.

**Fix:** The API client has built-in retry/polling with exponential backoff. For very large companies, check the Dagster run logs for polling status.

### API returns empty/null response

**Symptom:** Asset materializes with 0 rows.

**Cause:** Company may not have data for the requested valuation year/round.

**Fix:** Check `apis.company_valuation_data` for the company's available rounds:
```python
import duckdb
con = duckdb.connect("database/local.duckdb")
con.sql("SELECT * FROM apis.company_valuation_data WHERE Corporatie = '3B Wonen'").show()
```

---

## Data Issues

### Pandera validation error on empty DataFrame

**Symptom:**
```
SchemaErrors: expected series 'col' to have type float64, got object
```

**Cause:** Empty DataFrames have `object` dtype by default, which doesn't match `float64` schema.

**Fix:** Use `str` type in Pandera schema for columns that may be empty, or pre-cast columns.

### Duplicate VHE-nr rows in dataset_basis

**Symptom:** More rows than expected in `dataset_basis`.

**Cause:** Multiple valuation types per VHE (Basis, Full, Vrije waardering) or multiple valuation rounds.

**Fix:** This is expected behavior. Filter by `Waarderingstype` if needed:
```sql
SELECT * FROM models.dataset_basis WHERE Waarderingstype = 'Basis'
```

### Single-quote in company name breaks DuckDB

**Symptom:** SQL error on companies like `Woonstichting 'thuis`.

**Fix:** Already handled by the monkey-patch in `resources.py` that escapes single quotes in partition values.

---

## Performance Issues

### dbt build takes too long

**Cause:** `int_marktwaardeparameters` (~90s) and `int_parameteroverzicht` (~75s) are the slowest models due to large FULL OUTER JOINs.

**Fix:**
- Build only what you need: `uv run dbt run --profiles-dir . --project-dir . --select +dataset_basis`
- Use `dbt compile` first to verify SQL without running

### DuckDB lock errors

**Symptom:** `IO Error: Could not set lock on file`

**Cause:** Another process (Dagster, DBeaver, Python script) has the database open.

**Fix:** Close all other DuckDB connections before running dbt or Dagster.
