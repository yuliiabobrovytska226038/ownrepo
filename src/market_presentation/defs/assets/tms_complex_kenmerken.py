"""Complex kenmerken — Kwalitatieve kenmerken sheet from export-complexes-characteristics Excel.

Legacy table:
    Complex_kenmerken_Kwalitatieve_kenmerken
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import ComplexKenmerkenSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    download_round_bytes_files,
    parse_single_sheet_files,
)

_OUTPUT_NAME = "complex_kenmerken"
_LEGACY_TABLE = "Complex_kenmerken_Kwalitatieve_kenmerken"
_SHEET_CANDIDATES = ("Kwalitatieve kenmerken", "Kwalitatieve_kenmerken")


@dg.asset(
    key_prefix=["tms"],
    name="complex_kenmerken_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download complex kenmerken Excel files to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_complex_kenmerken_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch complex kenmerken Excel from API and persist to disk per round."""
    return download_round_bytes_files(
        context,
        config,
        company_valuation_data,
        report_key="complex_characteristics",
        fetch_fn=lambda company_id, meta: tms_api.fetch_complex_characteristics_excel(company_id, meta.valuation_round_id),
    )


@dg.asset(
    key_prefix=["tms"],
    name=_OUTPUT_NAME,
    compute_kind="python",
    group_name="tms_assets",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie", "legacy_table": _LEGACY_TABLE},
    description=f"Complex kenmerken → legacy table {_LEGACY_TABLE}.",
    ins={"complex_kenmerken_files": dg.AssetIn(key=dg.AssetKey(["tms", "complex_kenmerken_files"]))},
)
def tms_complex_kenmerken(context: dg.AssetExecutionContext, complex_kenmerken_files: pd.DataFrame) -> pd.DataFrame:
    """Parse complex kenmerken Excel from disk and extract the Kwalitatieve kenmerken sheet."""
    return parse_single_sheet_files(context, complex_kenmerken_files, output_name=_OUTPUT_NAME, sheet_candidates=_SHEET_CANDIDATES, schema=ComplexKenmerkenSchema)
