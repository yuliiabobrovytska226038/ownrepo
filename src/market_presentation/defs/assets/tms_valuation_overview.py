"""Waardeoverzicht — Complex + VHE sheets from the MARKET_VALUE_OVERVIEW report.

Legacy tables:
    Waardeoverzicht_Complex
    Waardeoverzicht_VHE
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import ValuationOverviewComplexSchema, ValuationOverviewVheSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    build_sheet_asset_outs,
    download_round_report_files,
    parse_and_yield_multi_sheet_results,
)

_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("valuation_overview_complex", "Waardeoverzicht_Complex", ("Complex",)),
    ("valuation_overview_vhe", "Waardeoverzicht_VHE", ("VHE",)),
]
_SCHEMA_BY_OUTPUT = {
    "valuation_overview_complex": ValuationOverviewComplexSchema,
    "valuation_overview_vhe": ValuationOverviewVheSchema,
}


@dg.asset(
    key_prefix=["tms"],
    name="valuation_overview_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Trigger/poll/download waardeoverzicht Excel to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_valuation_overview_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Trigger, poll, and download waardeoverzicht Excel per round; persist path to disk."""
    return download_round_report_files(
        context,
        config,
        company_valuation_data,
        report_key="valuation_overview",
        fetch_fn=lambda company_id, meta, path: tms_api.fetch_valuation_overview(company_id, meta.valuation_round_id, path),
    )


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP, report_label="Waardeoverzicht"),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"valuation_overview_files": dg.AssetIn(key=dg.AssetKey(["tms", "valuation_overview_files"]))},
)
def tms_valuation_overview(context: dg.AssetExecutionContext, valuation_overview_files: pd.DataFrame):
    """Parse waardeoverzicht Excel from disk and yield Complex + VHE sheets."""
    yield from parse_and_yield_multi_sheet_results(context, valuation_overview_files, sheet_map=_SHEET_MAP, schema_by_output=_SCHEMA_BY_OUTPUT, skiprows=2)
