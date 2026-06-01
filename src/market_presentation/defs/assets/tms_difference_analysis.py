"""TMS Verschillenanalyse — JSON difference analysis per subsidiary.

The old pipeline downloaded one JSON per company×subsidiary, then aggregated
the marketValue and policyValue into two separate tables using pd.json_normalize.

Legacy tables:
    TMS_Verschillenanalyse_marketValue
    TMS_Verschillenanalyse_policyValue
"""

import json

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions, get_company_partition_keys
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import TmsVerschillenanalyseSchema
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    align_to_schema_columns,
    build_api_cache_path,
    build_file_record,
    build_files_output,
    build_raw_output_path,
    build_result,
    get_round_metadata,
    iter_json_file_rows,
    load_or_fetch_json,
)

_SHEET_MAP: list[tuple[str, str]] = [
    ("tms_verschillenanalyse_marketvalue", "TMS_Verschillenanalyse_marketValue"),
    ("tms_verschillenanalyse_policyvalue", "TMS_Verschillenanalyse_policyValue"),
]
_NUMERICAL_EXPLAINERS_SUFFIX = ".numericalExplainers"


def _parse_difference_analysis(json_data: dict, company_id: str, issue_year: int, subsidiary: str) -> dict[str, pd.DataFrame]:
    """Parse difference analysis JSON into marketValue and policyValue DataFrames.

    Matches the legacy agg_reports.py parse_json_verschillenanalyse logic.
    """
    result: dict[str, pd.DataFrame] = {}
    values = json_data.get("values", {})

    for val_type in ["marketValue", "policyValue"]:
        data = values.get(val_type)
        if not data:
            continue
        df = pd.json_normalize(data)
        for column in df.columns:
            if column.endswith(_NUMERICAL_EXPLAINERS_SUFFIX):
                df[column] = df[column].map(_json_text_or_none)
        df["Valuation_type"] = val_type
        df["Corporatie"] = company_id
        df["Jaar"] = issue_year
        df["Peildatum"] = pd.Timestamp(f"{issue_year}-12-31")
        df["Werkmaatschappij"] = subsidiary
        result[val_type] = df

    return result


def _json_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if pd.isna(value):
        return None
    return str(value)


def _extract_subsidiaries(vgr_data: dict) -> list[str]:
    """Extract sorted unique subsidiaries from VGR valuation complexes."""
    complexes = vgr_data.get("complexes", {}).get("valuationComplexes", [])
    subsidiaries = {str(complex_data.get("subsidiary")).strip() for complex_data in complexes if complex_data.get("subsidiary")}
    return sorted(subsidiary for subsidiary in subsidiaries if subsidiary)


def _ensure_difference_analysis_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return align_to_schema_columns(df, TmsVerschillenanalyseSchema, keep_extra=True)


@dg.asset(
    key_prefix=["tms"],
    name="difference_analysis_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download difference analysis JSON to disk per company round and subsidiary.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_difference_analysis_files(
    context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, vgr_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame
) -> pd.DataFrame:
    """Fetch VGR complexes to discover subsidiaries, then download difference analysis JSON per subsidiary."""
    records: list[dict] = []
    all_paths: list[str] = []
    all_sources: list[str] = []
    for company_id in get_company_partition_keys(context):
        for meta in get_round_metadata(company_id, company_valuation_data):
            if meta.data_set_id is None:
                context.log.warning(f"No VGR data_set_id for {company_id}/{meta.issue_year}; skipping difference analysis")
                continue
            vgr_cache = load_or_fetch_json(
                build_api_cache_path("vgr_valuation_complexes", f"{company_id} {meta.issue_year} {meta.data_set_id}"),
                lambda company_id=company_id, meta=meta: vgr_api.fetch_vgr_valuation_complexes(company_id, str(meta.data_set_id)),
                force_download=config.force_download,
            )
            all_paths.append(str(vgr_cache.path.resolve()))
            all_sources.append(vgr_cache.source)
            if not isinstance(vgr_cache.value, dict):
                raise dg.Failure(description=f"Invalid VGR valuation-complexes cache for {company_id}/ds={meta.data_set_id}: expected object")
            subsidiaries = _extract_subsidiaries(vgr_cache.value)
            if not subsidiaries:
                context.log.warning(f"No subsidiaries found for {company_id}/{meta.issue_year}/ds={meta.data_set_id}")
                continue
            context.log.info(f"Found {len(subsidiaries)} subsidiar(ies) for {company_id}/{meta.issue_year}: {subsidiaries}")
            for subsidiary in subsidiaries:
                cache = load_or_fetch_json(
                    build_raw_output_path("difference_analysis", meta, subsidiary=subsidiary),
                    lambda company_id=company_id, meta=meta, subsidiary=subsidiary: tms_api.fetch_difference_analysis(company_id, meta.valuation_round_id, subsidiary),
                    force_download=config.force_download,
                )
                all_paths.append(str(cache.path.resolve()))
                all_sources.append(cache.source)
                records.append(build_file_record(meta, cache, subsidiary=subsidiary))
    return build_files_output(context, records, paths=all_paths, sources=all_sources)


@dg.multi_asset(
    outs={
        output_name: dg.AssetOut(
            key=["tms", output_name],
            description=f"Verschilanalyse → legacy table {legacy_table}.",
            metadata={"partition_expr": "Corporatie", "legacy_table": legacy_table},
            group_name="tms_assets",
        )
        for output_name, legacy_table in _SHEET_MAP
    },
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"difference_analysis_files": dg.AssetIn(key=dg.AssetKey(["tms", "difference_analysis_files"]))},
)
def tms_difference_analysis(context: dg.AssetExecutionContext, difference_analysis_files: pd.DataFrame):
    """Parse difference analysis JSON from disk and yield marketValue + policyValue tables."""
    market_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    for row, meta, _, json_data in iter_json_file_rows(context, difference_analysis_files, expected_type=dict, expected_label="object"):
        subsidiary = str(row["subsidiary"])
        if not json_data or "values" not in json_data:
            context.log.info(f"No difference analysis data for {meta.company_id}/{meta.valuation_round_id}/{subsidiary}")
            continue
        parsed = _parse_difference_analysis(json_data, meta.company_id, meta.issue_year, subsidiary)
        if "marketValue" in parsed:
            market_frames.append(parsed["marketValue"])
        if "policyValue" in parsed:
            policy_frames.append(parsed["policyValue"])
    market_df = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
    policy_df = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    market_df = _ensure_difference_analysis_schema_columns(market_df)
    policy_df = _ensure_difference_analysis_schema_columns(policy_df)
    context.log.info(f"verschilanalyse marketValue: {len(market_df)} rows")
    context.log.info(f"verschilanalyse policyValue: {len(policy_df)} rows")
    rounds = difference_analysis_files["valuation_round_id"].nunique() if not difference_analysis_files.empty else 0
    yield build_result("tms_verschillenanalyse_marketvalue", market_df, schema=TmsVerschillenanalyseSchema, rounds=rounds)
    yield build_result("tms_verschillenanalyse_policyvalue", policy_df, schema=TmsVerschillenanalyseSchema, rounds=rounds)
