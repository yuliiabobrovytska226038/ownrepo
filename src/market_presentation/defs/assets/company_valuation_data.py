"""Dagster assets for per-company valuation data sourced from the TMS REST API.

Assets:
    tms/company_valuation_data  (partitioned by company)
        Q4 main valuation rows for one corporation from the TMS API.
"""

import dagster as dg
import pandas as pd

from ..config import ISSUE_YEARS
from ..partitions import BACKFILL_POLICY, company_partitions, get_company_partition_keys
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import CompanyValuationDataSchema
from ..utils.tms_report_utils import RawReportConfig, add_raw_metadata, build_api_cache_path, dataframe_preview, load_or_fetch_json, validate_dataframe


@dg.asset(
    key_prefix=["tms"],
    name="company_valuation_data",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "company_id"},
    description=f"Joined TMS valuation rounds and main valuation records for data years {ISSUE_YEARS} for a single company.",
)
def company_valuation_data(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource) -> pd.DataFrame:
    """Fetch joined TMS company, valuation-round, and valuation detail for the current company partitions."""
    frames: list[pd.DataFrame] = []
    saved_paths: list[str] = []
    sources: list[str] = []
    round_count = 0
    valuation_count = 0

    for company_id in get_company_partition_keys(context):
        cache = load_or_fetch_json(
            build_api_cache_path("company_valuation_data", company_id),
            lambda company_id=company_id: tms_api.get_company_valuation_data(company_id, ISSUE_YEARS),
            force_download=config.force_download,
        )
        saved_paths.append(str(cache.path))
        sources.append(cache.source)
        rows = cache.value
        if not isinstance(rows, list):
            raise dg.Failure(description=f"Invalid company_valuation_data JSON cache for {company_id}: expected list")
        if not rows:
            raise dg.Failure(description=f"No Q4 {ISSUE_YEARS} valuation data found for {company_id}")

        raw_df = pd.DataFrame(rows)
        df = raw_df[(raw_df["archived"] == False) & (raw_df["main_valuation"] == True)].copy()  # noqa: E712 - pandas boolean mask
        if df.empty:
            raise dg.Failure(description=f"No non-archived main valuation data found for {company_id} in Q4 {ISSUE_YEARS}")
        df = validate_dataframe(df.drop(columns=["archived", "main_valuation"], errors="ignore"), CompanyValuationDataSchema)
        frames.append(df)

        company_round_count = int(df["valuation_round_id"].nunique())
        company_valuation_count = int(df["valuation_id"].dropna().nunique())
        round_count += company_round_count
        valuation_count += company_valuation_count
        context.log.info(
            "Resolved joined main valuation data for %s: %d round(s), %d non-archived main valuation row(s)",
            company_id,
            company_round_count,
            len(df),
        )

    df = pd.concat(frames, ignore_index=True)
    context.add_output_metadata(
        {
            "round_count": round_count,
            "valuation_count": valuation_count,
            "row_count": len(df),
            "preview": dg.MetadataValue.md(dataframe_preview(df, rows=10)),
        }
    )
    add_raw_metadata(context, paths=saved_paths, sources=sources)
    return df
