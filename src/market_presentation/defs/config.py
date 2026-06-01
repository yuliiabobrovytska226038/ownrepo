"""Shared configuration constants for the Marktpresentatie pipeline.

Centralises all path, directory, and year constants used across the pipeline.
Assets and utilities import from here to ensure consistent file locations and
reporting periods.  Only ``ISSUE_YEARS`` needs updating when new data is fetched;
all chart year constants derive from it automatically.

Constants:
    DUCKDB_PATH: Absolute path to the local DuckDB database file.
    OUTPUT_DIR: Directory for intermediate JSON/Excel output files.
    GRAFIEKEN_DIR: Directory for generated Plotly HTML/JPEG chart files.
    DATA_DIR: Directory for external reference data (CBS, GeoJSON, Excel).
    EXPORT_DIR: Directory for CSV/Parquet data exports.
    ISSUE_YEARS: Valuation years as stored in DuckDB (Jaar column). Equal to the
        TMS API issueDate.year directly (Q4 2024 issue → Jaar=2024).
    CHART_YEAR: Most recent valuation year; equals max(ISSUE_YEARS).
    CHART_YEAR_M1: Prior valuation year used for YoY comparisons.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUCKDB_PATH = str(_PROJECT_ROOT / "database" / "local.duckdb")
OUTPUT_DIR = Path("data/tms")
GRAFIEKEN_DIR = Path("grafieken")
DATA_DIR = Path("data/extern")
EXPORT_DIR = Path("exports")

# Valuation years — these ARE the Jaar values stored in the DuckDB raw tables,
# matching the TMS API issueDate.year directly (no offset). Q4 2025 → Jaar=2025.
# Only ISSUE_YEARS needs updating when adding a new year; everything else auto-derives.
ISSUE_YEARS = [2024, 2025]
CHART_YEAR = max(ISSUE_YEARS)  # 2025 — most recent valuation year in dataset_basis
CHART_YEAR_M1 = max(ISSUE_YEARS) - 1  # 2024 — prior year used for YoY comparison
