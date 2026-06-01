"""Dagster resource definitions for the Marktpresentatie pipeline.

Registers all shared resources that assets depend on:

- ``tms_api`` — OAuth2-authenticated client for the TMS (Taxatie Management
  Systeem) API, used to fetch valuation rounds, reports, and per-VHE data.
- ``vgr_api`` — OAuth2-authenticated client for the VGR (Vastgoed Registratie)
  API, used to download real estate import files.
- ``duckdb`` — Direct DuckDB connection for schema introspection.
- ``io_manager`` — DuckDB Pandas IO manager that persists DataFrame assets
  as tables and reads dbt chart views back into DataFrames.

All API credentials are sourced from environment variables.
"""

import dagster as dg
import dagster_duckdb.io_manager as _duckdb_io
from dagster._core.storage.db_io_manager import TablePartitionDimension
from dagster_duckdb import DuckDBResource
from dagster_duckdb_pandas import DuckDBPandasIOManager

from ..config import DUCKDB_PATH
from .oauth2_api import Oauth2ApiResource

# ---------------------------------------------------------------------------
# Monkey-patch: escape single quotes in partition values for DuckDB IO manager.
# The upstream _static_where_clause does f"'{partition}'" which breaks on
# company names containing single quotes (e.g. "Woonstichting 'thuis").
# ---------------------------------------------------------------------------

_orig_static_where = _duckdb_io._static_where_clause


def _safe_static_where_clause(table_partition: TablePartitionDimension) -> str:
    """Escape single quotes in partition values before building the SQL WHERE clause."""
    escaped_partitions = [str(p).replace("'", "''") for p in table_partition.partitions]
    safe_partition = TablePartitionDimension(
        partition_expr=table_partition.partition_expr,
        partitions=escaped_partitions,
    )
    return _orig_static_where(safe_partition)


_duckdb_io._static_where_clause = _safe_static_where_clause


def _build_api_resource(base_url_env_var: str) -> Oauth2ApiResource:
    """Create a configured OAuth2 API resource from environment variables."""
    return Oauth2ApiResource(
        base_url=dg.EnvVar(base_url_env_var),
        token_url=dg.EnvVar("TMS_TOKEN_URL"),
        client_id=dg.EnvVar("TMS_CLIENT_ID"),
        client_secret=dg.EnvVar("TMS_CLIENT_SECRET"),
        username=dg.EnvVar("TMS_USER"),
        password=dg.EnvVar("TMS_PASS"),
    )


RESOURCE_DEFS = {
    "tms_api": _build_api_resource("TMS_API_BASE_URL"),
    "vgr_api": _build_api_resource("VGR_API_BASE_URL"),
    "duckdb": DuckDBResource(database=DUCKDB_PATH),
    "io_manager": DuckDBPandasIOManager(database=DUCKDB_PATH),
}
DEFAULT_EXECUTOR = dg.multiprocess_executor.configured({"max_concurrent": 1})


@dg.definitions
def resources() -> dg.Definitions:
    """Register all Dagster resources: TMS API, VGR API, DuckDB, and IO manager."""
    return dg.Definitions(resources=RESOURCE_DEFS, executor=DEFAULT_EXECUTOR)
