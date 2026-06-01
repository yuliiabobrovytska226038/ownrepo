"""Marktwaarde basis — per-VHE market values from the TMS API.

The old pipeline loaded this from an Oracle SQL export that joined BASIC and FULL
valuation levels side-by-side.  The new pipeline fetches per-VHE market values via
the TMS API (complexes/market-values → rental-units/market-values), pivots the two
levels into columns, and adds Corporatie/Jaar metadata.

Legacy table:
    Marktwaarde_Basis  (14 cols: Corporatie, ISSUE_DATE, Werkmaatschappij, Complexcode,
        VHE-nr, Marktwaarde basis, Marktwaarde basis doorexploiteren,
        Marktwaarde basis uitponden, Scenario basis, Marktwaarde,
        Marktwaarde doorexploiteren, Marktwaarde uitponden, Scenario, Jaar)

Note: complexCode, rentalUnitNumber, and subsidiary are NOT in the API response.
Those are available in the VGR VHE-gegevens sheet and should be joined via dbt.
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import MarketValueBasisSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    concat_frames,
    download_round_json_files,
    iter_json_file_rows,
    validate_dataframe,
)

_OUTPUT_NAME = "market_value_basis"
_LEGACY_TABLE = "Marktwaarde_Basis"

# Columns to pick from each valuation level after the pivot
_FULL_RENAMES = {
    "marketValue": "Marktwaarde",
    "marketValueExploit": "Marktwaarde doorexploiteren",
    "marketValueSell": "Marktwaarde uitponden",
    "chosenScenario": "Scenario",
}

_BASIC_RENAMES = {
    "marketValue": "Marktwaarde basis",
    "marketValueExploit": "Marktwaarde basis doorexploiteren",
    "marketValueSell": "Marktwaarde basis uitponden",
    "chosenScenario": "Scenario basis",
}

_MERGE_KEYS = ["rentalUnitInternalId", "complexInternalId"]


def _pivot_levels(records: list[dict], company_id: str, issue_date: str, issue_year: int) -> pd.DataFrame:
    """Pivot BASIC/FULL valuation levels side-by-side per rental unit, matching legacy layout."""
    df = pd.DataFrame(records)
    if df.empty:
        return df

    basic = df[df["valuationLevel"] == "BASIC"].copy()
    full = df[df["valuationLevel"] == "FULL"].copy()

    basic = basic.rename(columns=_BASIC_RENAMES)[_MERGE_KEYS + list(_BASIC_RENAMES.values())]
    full = full.rename(columns=_FULL_RENAMES)[_MERGE_KEYS + list(_FULL_RENAMES.values())]

    # Deduplicate before merging — API may return the same VHE from multiple complex endpoints
    basic = basic.drop_duplicates(subset=_MERGE_KEYS)
    full = full.drop_duplicates(subset=_MERGE_KEYS)

    pivoted = pd.merge(full, basic, on=_MERGE_KEYS, how="outer")
    pivoted["Corporatie"] = company_id
    pivoted["ISSUE_DATE"] = issue_date
    pivoted["Jaar"] = issue_year
    pivoted["Peildatum"] = pd.Timestamp(f"{issue_year}-12-31")
    return pivoted


@dg.asset(
    key_prefix=["tms"],
    name="market_value_basis_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download per-VHE market values JSON to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_market_value_basis_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch per-VHE market values JSON from API and persist to disk per round."""
    return download_round_json_files(
        context,
        config,
        company_valuation_data,
        report_key="market_value_basis",
        fetch_fn=lambda company_id, meta: tms_api.fetch_market_value_basis(company_id, meta.valuation_round_id),
    )


@dg.asset(
    key_prefix=["tms"],
    name=_OUTPUT_NAME,
    compute_kind="python",
    group_name="tms_assets",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie", "legacy_table": _LEGACY_TABLE},
    description=f"Per-VHE market values (pivoted BASIC/FULL) → legacy table {_LEGACY_TABLE}.",
    ins={"market_value_basis_files": dg.AssetIn(key=dg.AssetKey(["tms", "market_value_basis_files"]))},
)
def tms_market_value_basis(context: dg.AssetExecutionContext, market_value_basis_files: pd.DataFrame) -> pd.DataFrame:
    """Parse per-VHE market values JSON from disk, pivot BASIC/FULL, and produce one row per VHE."""
    frames: list[pd.DataFrame] = []
    for _, meta, _, records in iter_json_file_rows(context, market_value_basis_files, expected_type=list, expected_label="list"):
        if not records:
            context.log.info(f"No market value basis records for {meta.company_id}/{meta.valuation_round_id}")
            continue
        pivoted = _pivot_levels(records, meta.company_id, meta.issue_date, meta.issue_year)
        if not pivoted.empty:
            frames.append(pivoted)
    df = concat_frames(frames)
    context.log.info(f"{_OUTPUT_NAME}: {len(df)} rows, {len(df.columns)} columns")
    return validate_dataframe(df, MarketValueBasisSchema)
