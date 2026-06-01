"""Beleidswaardeparameters — single sheet from export-policy-value-parameters Excel.

Legacy table:
    Beleidswaarde_parameters_Parameters_beleidswaarde
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import PolicyValueParametersSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    download_round_bytes_files,
    parse_single_sheet_files,
)

_OUTPUT_NAME = "policy_value_parameters"
_LEGACY_TABLE = "Beleidswaarde_parameters_Parameters_beleidswaarde"
_SHEET_CANDIDATES = ("Parameters beleidswaarde",)


@dg.asset(
    key_prefix=["tms"],
    name="policy_value_parameters_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download beleidswaardeparameters Excel files to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_policy_value_parameters_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch beleidswaardeparameters Excel from API and persist to disk per round."""
    return download_round_bytes_files(
        context,
        config,
        company_valuation_data,
        report_key="policy_value_parameters",
        fetch_fn=lambda company_id, meta: tms_api.fetch_policy_value_parameters(company_id, meta.valuation_round_id),
    )


@dg.asset(
    key_prefix=["tms"],
    name=_OUTPUT_NAME,
    compute_kind="python",
    group_name="tms_assets",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie", "legacy_table": _LEGACY_TABLE},
    description=f"Beleidswaardeparameters → legacy table {_LEGACY_TABLE}.",
    ins={"policy_value_parameters_files": dg.AssetIn(key=dg.AssetKey(["tms", "policy_value_parameters_files"]))},
)
def tms_policy_value_parameters(context: dg.AssetExecutionContext, policy_value_parameters_files: pd.DataFrame) -> pd.DataFrame:
    """Parse beleidswaardeparameters Excel from disk and extract the Parameters beleidswaarde sheet."""
    return parse_single_sheet_files(context, policy_value_parameters_files, output_name=_OUTPUT_NAME, sheet_candidates=_SHEET_CANDIDATES, schema=PolicyValueParametersSchema)
