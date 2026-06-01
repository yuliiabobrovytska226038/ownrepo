"""Beleidswaarderapport — Complex + VHE sheets from the POLICY_VALUE_WATERFALL report.

Legacy tables:
    Beleidswaarderapport_Complex
    Beleidswaarderapport_VHE
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import PolicyValueReportComplexSchema, PolicyValueReportVheSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    build_sheet_asset_outs,
    download_round_report_files,
    parse_and_yield_multi_sheet_results,
)

_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("policy_value_report_complex", "Beleidswaarderapport_Complex", ("Complex",)),
    ("policy_value_report_vhe", "Beleidswaarderapport_VHE", ("VHE",)),
]
_SCHEMA_BY_OUTPUT = {
    "policy_value_report_complex": PolicyValueReportComplexSchema,
    "policy_value_report_vhe": PolicyValueReportVheSchema,
}


@dg.asset(
    key_prefix=["tms"],
    name="policy_value_report_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Trigger/poll/download beleidswaarderapport Excel to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_policy_value_report_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Trigger, poll, and download beleidswaarderapport Excel per round; persist path to disk."""
    return download_round_report_files(
        context,
        config,
        company_valuation_data,
        report_key="policy_value_report",
        fetch_fn=lambda company_id, meta, path: tms_api.fetch_policy_value_report(company_id, meta.valuation_round_id, path),
    )


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP, report_label="Beleidswaarderapport"),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"policy_value_report_files": dg.AssetIn(key=dg.AssetKey(["tms", "policy_value_report_files"]))},
)
def tms_policy_value_report(context: dg.AssetExecutionContext, policy_value_report_files: pd.DataFrame):
    """Parse beleidswaarderapport Excel from disk and yield Complex + VHE sheets."""
    yield from parse_and_yield_multi_sheet_results(context, policy_value_report_files, sheet_map=_SHEET_MAP, schema_by_output=_SCHEMA_BY_OUTPUT, skiprows=2)
