"""Dagster job definitions for the Marktpresentatie pipeline."""

import dagster as dg

COMPANY_LIST_ASSET_KEY = dg.AssetKey(["tms", "company_list"])

# API basis plus all raw TMS report assets. Uses without_checks() to avoid
# circular graph issues from dbt source-test checks during live materialization.
TMS_ASSETS_SELECTION = (dg.AssetSelection.groups("tms_downloads") | dg.AssetSelection.groups("tms_assets")).without_checks()
TMS_PARTITIONED_ASSETS_SELECTION = (dg.AssetSelection.groups("tms_assets") - dg.AssetSelection.assets(COMPANY_LIST_ASSET_KEY)).without_checks()
COMPANY_LIST_SELECTION = dg.AssetSelection.assets(COMPANY_LIST_ASSET_KEY).without_checks()

TMS_DOWNLOADS_SELECTION = dg.AssetSelection.groups("tms_downloads").without_checks()
TMS_PARSE_SELECTION = (dg.AssetSelection.groups("tms_assets") - dg.AssetSelection.assets(COMPANY_LIST_ASSET_KEY)).without_checks()

EXTERNAL_SOURCES_SELECTION = dg.AssetSelection.groups("external_sources").without_checks()

DBT_TRANSFORMATIONS_SELECTION = (dg.AssetSelection.from_coercible("kind:dbt") - dg.AssetSelection.key_prefixes(["models", "chart_"])).without_checks()

CHARTS_SELECTION = dg.AssetSelection.groups("charts").upstream(depth=1).without_checks()


@dg.definitions
def jobs() -> dg.Definitions:
    """Register Dagster jobs for the Marktpresentatie pipeline."""
    return dg.Definitions(
        jobs=[
            dg.define_asset_job(
                name="company_list_job",
                selection=COMPANY_LIST_SELECTION,
                description="Materialize the unpartitioned TMS company list once without running downstream partitioned assets.",
            ),
            dg.define_asset_job(
                name="tms_downloads_job",
                selection=TMS_DOWNLOADS_SELECTION,
                description="Fetch raw TMS report files from API to disk (API-constrained, runs sequentially).",
            ),
            dg.define_asset_job(
                name="tms_parse_job",
                selection=TMS_PARSE_SELECTION,
                description="Parse downloaded TMS report files from disk into DuckDB tables (no API calls needed).",
            ),
            dg.define_asset_job(
                name="tms_partitioned_assets_job",
                selection=TMS_PARTITIONED_ASSETS_SELECTION,
                description="Materialize only the company-partitioned raw TMS assets, skipping the unpartitioned company list.",
            ),
            dg.define_asset_job(
                name="external_sources_job",
                selection=EXTERNAL_SOURCES_SELECTION,
                description="Materialize external reference source assets.",
            ),
            dg.define_asset_job(
                name="dbt_transformations_job",
                selection=DBT_TRANSFORMATIONS_SELECTION,
                description="Materialize dbt staging, intermediate, and mart transformations without dbt chart views.",
            ),
            dg.define_asset_job(
                name="charts_job",
                selection=CHARTS_SELECTION,
                description="Materialize dbt chart views and render Plotly chart assets.",
            ),
        ]
    )
