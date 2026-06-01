# Marktpresentatie — REV Market Presentation Pipeline

Pipeline for extracting TMS (Taxatie Management Systeem) valuation data, transforming it via Dagster + dbt + DuckDB, and generating market presentation charts for Ortec Finance REV.

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API credentials

# 3. Start Dagster UI
uv run dg dev
# Open http://localhost:3000

# 4. Run the pipeline (single company)
uv run dg launch --job external_sources_job
uv run dg launch --job tms_assets_job --partition "3B Wonen"
uv run dg launch --job dbt_transformations_job
uv run dg launch --job charts_job
```

## Architecture

```
TMS REST API → Dagster assets → DuckDB (tms/external schemas)
                                    ↓
                          dbt models (staging → int → marts)
                                    ↓
                          Chart assets → Plotly HTML/JPEG
```

See [docs/architecture.md](docs/architecture.md) for full architecture documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System architecture and data flow |
| [Operational Runbook](docs/runbook.md) | Step-by-step pipeline operation guide |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Data Dictionary](docs/data_dictionary.md) | All tables, columns, and business definitions |
| [Testing](docs/testing.md) | Test strategy, coverage, and running tests |

## Key Commands

```powershell
# Pipeline jobs
uv run dg launch --job external_sources_job              # Load reference data
uv run dg launch --job tms_assets_job --partition "X"     # Fetch TMS data for company X
uv run dg launch --job dbt_transformations_job            # Run dbt models
uv run dg launch --job charts_job                         # Generate charts

# Testing
uv run pytest tests/ -v                                   # Python unit tests
cd dbt && uv run dbt test --profiles-dir . --project-dir . # dbt tests

# Reset
.\scripts\reset_dagster.ps1                               # Full reset
.\scripts\reset_dagster.ps1 -Materialize -Company "3B Wonen"  # Reset + rebuild
```

## Project Structure

```
src/market_presentation/
  defs/
    assets/          # Dagster asset definitions (TMS extraction + chart generation)
    resources/       # OAuth2 API client, DuckDB I/O manager
    schemas/         # Pandera validation schemas
    utils/           # Shared helpers (report parsing, chart utilities)
    config.py        # Paths, years, directories
    jobs.py          # Dagster job definitions
    partitions.py    # Company partition list (204 companies)
dbt/
  models/
    staging/         # Reconstruct/clean raw sources
    int/             # Join/combine intermediate tables
    marts/           # Business output tables (dataset_basis, etc.)
    charts/          # Pre-aggregated chart view models
    sources/         # Source YAML definitions (54 sources)
database/            # DuckDB database (local.duckdb)
grafieken/           # Generated chart output (HTML/JPEG)
scripts/             # Utility scripts (reset, compare, benchmark)
tests/               # Pytest unit tests
docs/                # Documentation
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|----------|---------|
| `DAGSTER_HOME` | Dagster instance directory |
| `TMS_TOKEN_URL` | OAuth2 token endpoint |
| `TMS_CLIENT_ID` / `TMS_CLIENT_SECRET` | API credentials |
| `TMS_USER` / `TMS_PASS` | API user credentials |
| `TMS_API_BASE_URL` | TMS API base URL |
| `VGR_API_BASE_URL` | VGR API base URL |
| `UV_INDEX_OFNEXUS_USERNAME` / `UV_INDEX_OFNEXUS_PASSWORD` | Nexus feed auth |

## Technology Stack

- **Python 3.12** — Runtime
- **UV** — Package manager
- **Dagster** — Orchestration and asset management
- **dbt** (DuckDB adapter) — SQL transformations
- **DuckDB** — Analytical database
- **Plotly** — Chart generation (HTML/JPEG)
- **Pandera** — DataFrame validation

---

# Development Setup

This project provides two options for local development:
- Using a Python devcontainer (see `docs/How to develop with a Python devcontainer.md`)
- Using a local Python installation (see `docs/How to develop with a local Python installation.md`)

The workflow using devcontainers is the Ortec Finance standard.

## Installing the VS Code IDE

This template project assumes that the `VS code` IDE is used:
- [Download](https://code.visualstudio.com/download) and run the most recent version of the `VS Code` installer.

Install the recommended extensions listed in `.vscode/extensions.json`.
Popups should automatically appear when working in VS code.

## Connecting to a Python package feed

To link your `project` to the private `Ortec Finance` `Python package feed`, the following part must be added to **ALL** `pyproject.toml` files:
```toml
[[tool.uv.index]]
name = "ofnexus"
url = "https://nexus.orca.ortec-finance.com/repository/pypi-all/simple"
default = true
```

**Important:** ensure you have set the following environment variables to authenticate to the package feed: `UV_INDEX_OFNEXUS_USERNAME`, `UV_INDEX_OFNEXUS_PASSWORD`.
You can find a documentation page on `Confluence` to see how to generate these credentials for `Nexus Pro`.

# Build

This project template provides two options for local development:
- Using a Python devcontainer (see `docs/How to develop with a Python devcontainer.md`)
- Using a local Python installation (see `docs/How to develop with a local Python installation.md`)

The workflow using devcontainers is the `Ortec Finance` standard and should be used.
If for some reason you cannot work with devcontainers, then a local installation can be used as a fallback.
Follow the steps up to the point where the `Python environment` has been build.

# Test & Code Analysis

Run your `unittests` locally from the `Testing` window in `VS code`.

**NOTE:** a `pytest.ini` file is present in `src/tests` to inject additional configurations/information into the `unittest` run.

![alt text](docs/imgs/VS_code_testing_window.png)

The better way to run your `unittests` is to have an automated `.tekton` pipeline that runs automatically on `Pull Requests` or `merge`. These pipelines can also be used to do static code analysis using `SonarQube`. Ask someone if you are not sure how to set this up.

If the above-mentioned infrastructure has been created, then we can then also link `SonarQube` projects for local development:
- Configure the plugin for your `SonarQube` project.
  - The following will be automatically added to `.vscode/settings.json`.
    ```json
    "sonarlint.connectedMode.project": {
            "connectionId": "<your-sonarqube-connection-id>",
            "projectKey": "<your-project-key>"
        }
    ```

- Smells and issues will now be highlighted in your code files and will show up in the `Problems` window of `VS code`.

  ![alt text](docs/imgs/VS_code_problem_window.png)

# Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality before commits (`not enabled by default`).

The hooks are configured in `.pre-commit-config.yaml` and will automatically run:
- Trailing whitespace removal
- End of file fixes
- YAML/JSON/TOML validation
- Ruff linting (with auto-fix)
- Ruff formatting
- Pytest tests

You can run the following commands from `inside` your `virtual environment`:
```powershell
# install
uvx pre-commit install
```
```powershell
# manual run
uvx pre-commit run --all-files
```
```powershell
# uninstall
uvx pre-commit uninstall
```

### Bypassing pre-commit hooks

If you need to bypass the hooks for a specific commit (not recommended):
```powershell
git commit --no-verify -m "Your commit message"
```
