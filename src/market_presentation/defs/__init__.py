"""Dagster definitions package for the Marktpresentatie pipeline.

This package contains all Dagster definitions organised into sub-modules:

- ``assets/`` — Dagster assets for API ingestion, external sources, TMS reports,
  JSON flattening, and Plotly chart generation.
- ``jobs.py`` — Dagster jobs for partition-aware materialization flows.
- ``partitions.py`` — Included housing corporations and shared company partition definition.
- ``resources/`` — OAuth2 API clients (TMS/VGR), DuckDB resource, and IO manager.
- ``utils/`` — Shared helpers for column normalisation, DuckDB schema alignment,
  and Plotly graph utilities.
- ``config.py`` — Pipeline-wide constants (paths, years, directories).
"""
