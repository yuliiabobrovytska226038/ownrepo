"""Complex referenties — single sheet from export-complexes-references Excel.

Legacy table:
    Complex_referenties_Referentie_transacties
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import ComplexReferencesSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    download_round_bytes_files,
    parse_single_sheet_files,
)

_OUTPUT_NAME = "complex_references"
_LEGACY_TABLE = "Complex_referenties_Referentie_transacties"
_SHEET_CANDIDATES = ("Referentie transacties", "Referentie-transacties", "Referentietransacties")


@dg.asset(
    key_prefix=["tms"],
    name="complex_references_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download complex referenties Excel files to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_complex_references_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch complex referenties Excel from API and persist to disk per round."""
    return download_round_bytes_files(
        context,
        config,
        company_valuation_data,
        report_key="complex_references",
        fetch_fn=lambda company_id, meta: tms_api.fetch_complex_references(company_id, meta.valuation_round_id),
    )


@dg.asset(
    key_prefix=["tms"],
    name=_OUTPUT_NAME,
    compute_kind="python",
    group_name="tms_assets",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie", "legacy_table": _LEGACY_TABLE},
    description=f"Complex referenties → legacy table {_LEGACY_TABLE}.",
    ins={"complex_references_files": dg.AssetIn(key=dg.AssetKey(["tms", "complex_references_files"]))},
)
def tms_complex_references(context: dg.AssetExecutionContext, complex_references_files: pd.DataFrame) -> pd.DataFrame:
    """Parse complex referenties Excel from disk and extract the Referentie transacties sheet."""
    return parse_single_sheet_files(context, complex_references_files, output_name=_OUTPUT_NAME, sheet_candidates=_SHEET_CANDIDATES, schema=ComplexReferencesSchema)
