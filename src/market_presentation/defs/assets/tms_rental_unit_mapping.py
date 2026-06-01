"""Rental unit mapping — maps TMS rentalUnitInternalId to VHE-nummer.

The market_value_basis API table uses ``rentalUnitInternalId`` (a TMS internal
ID) which cannot be joined directly to ``VHE-nummer``/``VHE-nr``.  The VGR
dataset JSON endpoint returns rental units with both ``internalId`` and
``rentalUnitNumber``, providing the missing mapping.

Legacy table: N/A (legacy used direct Oracle SQL join on RENTAL_UNIT_NUMBER)
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions, get_company_partition_keys
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import RentalUnitMappingSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    add_metadata_columns,
    build_api_cache_path,
    build_file_record,
    build_files_output,
    get_round_metadata,
    iter_json_file_rows,
    load_or_fetch_json,
    resolve_vgr_data_set_id,
    validate_dataframe,
)

_OUTPUT_NAME = "rental_unit_mapping"
_RENTAL_UNIT_CATEGORIES = ("residential", "parking", "bogMog", "care", "soldUnits")


def _extract_rental_unit_records(data: dict) -> list[dict]:
    records: list[dict] = []
    rental_units = data.get("rentalUnits", {})
    for category in _RENTAL_UNIT_CATEGORIES:
        for rental_unit in rental_units.get(category, []):
            internal_id = rental_unit.get("internalId")
            rental_unit_number = rental_unit.get("rentalUnitNumber")
            if internal_id is None or rental_unit_number is None:
                continue
            records.append(
                {
                    "rentalUnitInternalId": int(internal_id),
                    "VHE-nr": str(rental_unit_number),
                    "complexInternalId": rental_unit.get("valuationComplexInternalId") or rental_unit.get("complexInternalId"),
                    "Complexcode": str(rental_unit.get("valuationComplexCode") or rental_unit.get("complexCode") or ""),
                }
            )
    return records


@dg.asset(
    key_prefix=["tms"],
    name="rental_unit_mapping_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download VGR dataset JSON to disk for rental unit ID mapping per company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_rental_unit_mapping_files(context: dg.AssetExecutionContext, config: RawReportConfig, vgr_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Resolve data_set_id and fetch VGR dataset JSON per round; persist path to disk."""
    records: list[dict] = []
    for company_id in get_company_partition_keys(context):
        for meta in get_round_metadata(company_id, company_valuation_data):
            data_set_id = resolve_vgr_data_set_id(context, vgr_api, company_id, meta)
            if data_set_id is None:
                continue
            context.log.info(f"Fetching VGR dataset JSON {data_set_id} for {company_id}/{meta.issue_year}")
            cache = load_or_fetch_json(
                build_api_cache_path("vgr_dataset", f"{company_id} {meta.issue_year} {data_set_id}"),
                lambda company_id=company_id, data_set_id=data_set_id: vgr_api.fetch_vgr_dataset_json(company_id, str(data_set_id)),
                force_download=config.force_download,
            )
            records.append(build_file_record(meta, cache, data_set_id=data_set_id))
    return build_files_output(context, records)


@dg.asset(
    key_prefix=["tms"],
    name=_OUTPUT_NAME,
    compute_kind="python",
    group_name="tms_assets",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Maps rentalUnitInternalId → VHE-nummer via VGR dataset JSON.",
    ins={"rental_unit_mapping_files": dg.AssetIn(key=dg.AssetKey(["tms", "rental_unit_mapping_files"]))},
)
def tms_rental_unit_mapping(context: dg.AssetExecutionContext, rental_unit_mapping_files: pd.DataFrame) -> pd.DataFrame:
    """Parse VGR dataset JSON from disk and extract rental unit ID mapping."""
    all_records: list[dict] = []
    for _, meta, _, data in iter_json_file_rows(context, rental_unit_mapping_files, expected_type=dict, expected_label="object"):
        round_records = _extract_rental_unit_records(data)
        if round_records:
            all_records.extend(add_metadata_columns(pd.DataFrame(round_records), meta).to_dict("records"))
    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=["rentalUnitInternalId", "VHE-nr", "complexInternalId", "Complexcode", "Corporatie", "Jaar", "Peildatum"])
    df = df.drop_duplicates(subset=["rentalUnitInternalId", "VHE-nr", "Corporatie", "Jaar"], keep="first")
    context.log.info(f"{_OUTPUT_NAME}: {len(df)} rows")
    return validate_dataframe(df, RentalUnitMappingSchema)
