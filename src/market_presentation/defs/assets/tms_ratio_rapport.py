"""Ratio rapport — Complex + VHE sheets from the RATIOS Excel report.

Legacy tables:
    Ratio_rapport_Complex
    Ratio_rapport_VHE
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import RatioRapportComplexSchema, RatioRapportVheSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    build_sheet_asset_outs,
    download_round_report_files,
    parse_and_yield_multi_sheet_results,
)

_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("ratio_rapport_complex", "Ratio_rapport_Complex", ("Complex",)),
    ("ratio_rapport_vhe", "Ratio_rapport_VHE", ("VHE",)),
]
_SCHEMA_BY_OUTPUT = {
    "ratio_rapport_complex": RatioRapportComplexSchema,
    "ratio_rapport_vhe": RatioRapportVheSchema,
}


@dg.asset(
    key_prefix=["tms"],
    name="ratio_rapport_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Trigger/poll/download ratio rapport Excel to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_ratio_rapport_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Trigger, poll, and download ratio rapport Excel per round; persist path to disk."""
    return download_round_report_files(
        context,
        config,
        company_valuation_data,
        report_key="ratios",
        fetch_fn=lambda company_id, meta, path: tms_api.fetch_ratios_report(company_id, meta.valuation_round_id, path),
    )


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP, report_label="Ratio rapport"),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"ratio_rapport_files": dg.AssetIn(key=dg.AssetKey(["tms", "ratio_rapport_files"]))},
)
def tms_ratio_rapport(context: dg.AssetExecutionContext, ratio_rapport_files: pd.DataFrame):
    """Parse ratio rapport Excel from disk and yield Complex + VHE sheets."""
    yield from parse_and_yield_multi_sheet_results(context, ratio_rapport_files, sheet_map=_SHEET_MAP, schema_by_output=_SCHEMA_BY_OUTPUT, skiprows=2)
