"""Vastgoedgegevens — multi-sheet VGR import file split into per-sheet assets.

The VGR API returns an Excel workbook with multiple sheets. The legacy pipeline
aggregated each sheet into a separate DuckDB table.

Legacy tables:
    Vastgoedgegevens_VHE_gegevens         (147 cols, ~3.7M rows)
    Vastgoedgegevens_Waarderingscomplexen  (57 cols, ~113K rows)
    Vastgoedgegevens_Sturingscomplexen     (3 cols, usually empty)
    Vastgoedgegevens_VoV_verleden          (3 cols, usually empty)
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions, get_company_partition_keys
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import SturingscomplexenSchema, VheGegevensSchema, VovVerledenSchema, WaarderingscomplexenSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    build_file_record,
    build_files_output,
    build_raw_output_path,
    build_sheet_asset_outs,
    get_round_metadata,
    load_or_fetch_bytes,
    parse_and_yield_multi_sheet_results,
    resolve_vgr_data_set_id,
)

# Sheet name → (output_name, legacy_table, candidate_sheet_names)
_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("vastgoedgegevens_vhe_gegevens", "Vastgoedgegevens_VHE_gegevens", ("VHE-gegevens", "VHE gegevens")),
    ("vastgoedgegevens_waarderingscomplexen", "Vastgoedgegevens_Waarderingscomplexen", ("Waarderingscomplexen",)),
    ("vastgoedgegevens_sturingscomplexen", "Vastgoedgegevens_Sturingscomplexen", ("Sturingscomplexen",)),
    ("vastgoedgegevens_vov_verleden", "Vastgoedgegevens_VoV_verleden", ("VoV-verleden", "VoV verleden")),
]
_SCHEMA_BY_OUTPUT = {
    "vastgoedgegevens_vhe_gegevens": VheGegevensSchema,
    "vastgoedgegevens_waarderingscomplexen": WaarderingscomplexenSchema,
    "vastgoedgegevens_sturingscomplexen": SturingscomplexenSchema,
    "vastgoedgegevens_vov_verleden": VovVerledenSchema,
}


def _drop_unknown_columns(output_name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    allowed_columns = set(_SCHEMA_BY_OUTPUT[output_name].to_schema().columns.keys())
    dropped_columns = [column for column in df.columns if column not in allowed_columns]
    if not dropped_columns:
        return df, []

    kept_columns = [column for column in df.columns if column in allowed_columns]
    return df.loc[:, kept_columns], dropped_columns


@dg.asset(
    key_prefix=["tms"],
    name="vastgoedgegevens_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download VGR import file to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_vastgoedgegevens_files(context: dg.AssetExecutionContext, config: RawReportConfig, vgr_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Resolve data_set_id, fetch VGR import Excel from API and persist to disk per round."""
    records: list[dict] = []
    for company_id in get_company_partition_keys(context):
        for meta in get_round_metadata(company_id, company_valuation_data):
            data_set_id = resolve_vgr_data_set_id(context, vgr_api, company_id, meta)
            if data_set_id is None:
                continue
            context.log.info(f"Using VGR dataset {data_set_id} for {company_id}/{meta.issue_year}")
            cache = load_or_fetch_bytes(
                build_raw_output_path("real_estate_data", meta),
                lambda company_id=company_id, data_set_id=data_set_id: vgr_api.download_vgr_import_file_bytes(company_id, str(data_set_id)),
                force_download=config.force_download,
            )
            if cache.value is None:
                context.log.warning(f"Failed to download VGR import file for {company_id}/ds={data_set_id}")
                continue
            records.append(build_file_record(meta, cache, data_set_id=data_set_id))
    return build_files_output(context, records)


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP, report_label="Vastgoedgegevens"),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"vastgoedgegevens_files": dg.AssetIn(key=dg.AssetKey(["tms", "vastgoedgegevens_files"]))},
)
def tms_vastgoedgegevens(context: dg.AssetExecutionContext, vastgoedgegevens_files: pd.DataFrame):
    """Parse VGR import file from disk and yield one asset per sheet (VHE-gegevens, Waarderingscomplexen, etc.)."""
    def prepare_frame(output_name: str, df: pd.DataFrame) -> pd.DataFrame:
        df, dropped_columns = _drop_unknown_columns(output_name, df)
        context.log.info(f"{output_name}: {len(df)} rows, {len(df.columns)} columns")
        if dropped_columns:
            context.log.warning(f"{output_name}: dropping extra workbook columns: {', '.join(dropped_columns)}")
        return df

    yield from parse_and_yield_multi_sheet_results(
        context,
        vastgoedgegevens_files,
        sheet_map=_SHEET_MAP,
        schema_by_output=_SCHEMA_BY_OUTPUT,
        skiprows=[1],
        prepare_frame=prepare_frame,
    )
