"""Dagster entry point for the Marktpresentatie (Market Presentation) pipeline.

This is the top-level module that bootstraps Dagster by auto-discovering all
definitions (assets, resources, jobs) from the ``defs/`` sub-package.  The
pipeline orchestrates data ingestion from TMS and VGR APIs, external CBS/dVi
sources, dbt transformations in DuckDB, and Plotly chart generation for the
annual Ortec Finance market presentation to Dutch social housing corporations.
"""

import warnings
from pathlib import Path

from dagster import definitions, load_from_defs_folder

warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning, module="openpyxl")


@definitions
def defs():
    """Load all Dagster definitions from the defs/ sub-package via auto-discovery."""
    return load_from_defs_folder(path_within_project=Path(__file__).parent)
