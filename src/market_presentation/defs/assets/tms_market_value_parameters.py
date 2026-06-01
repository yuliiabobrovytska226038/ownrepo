"""Marktwaardeparameters — 5 sheets from the export-market-value-parameters Excel.

Legacy tables:
    Marktwaarde_parameters_Complexparam_Woningen_Parkeren
    Marktwaarde_parameters_Complexparam_BOG_MOG_ZOG
    Marktwaarde_parameters_VHE_param_BOG_MOG_ZOG
    Marktwaarde_parameters_VHE_param_Benaderingsmethode
    Marktwaarde_parameters_VHE_param_Woningen_Parkeren

"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import (
    ComplexparamBogMogZogSchema,
    ComplexparamWoningenParkerenSchema,
    VheParamBenaderingsmethodeSchema,
    VheParamBogMogZogSchema,
    VheParamWoningenParkerenSchema,
)
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    align_to_schema_columns,
    build_sheet_asset_outs,
    download_round_bytes_files,
    parse_and_yield_multi_sheet_results,
)

# Sheet name → legacy DuckDB table name
_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("market_value_parameters_complexparam_woningen_parkeren", "Marktwaarde_parameters_Complexparam_Woningen_Parkeren", ("Complexparam Woningen Parkeren", "Complexparam. Woningen Parkeren")),
    ("market_value_parameters_complexparam_bog_mog_zog", "Marktwaarde_parameters_Complexparam_BOG_MOG_ZOG", ("Complexparam BOG MOG ZOG", "Complexparam. BOG MOG ZOG")),
    ("market_value_parameters_vhe_param_bog_mog_zog", "Marktwaarde_parameters_VHE_param_BOG_MOG_ZOG", ("VHE param BOG MOG ZOG", "VHE-param. BOG MOG ZOG")),
    ("market_value_parameters_vhe_param_benaderingsmethode", "Marktwaarde_parameters_VHE_param_Benaderingsmethode", ("VHE param Benaderingsmethode", "VHE-param. Benaderingsmethode")),
    ("market_value_parameters_vhe_param_woningen_parkeren", "Marktwaarde_parameters_VHE_param_Woningen_Parkeren", ("VHE param Woningen Parkeren", "VHE-param. Woningen Parkeren")),
]
_SCHEMA_BY_OUTPUT = {
    "market_value_parameters_complexparam_woningen_parkeren": ComplexparamWoningenParkerenSchema,
    "market_value_parameters_complexparam_bog_mog_zog": ComplexparamBogMogZogSchema,
    "market_value_parameters_vhe_param_bog_mog_zog": VheParamBogMogZogSchema,
    "market_value_parameters_vhe_param_benaderingsmethode": VheParamBenaderingsmethodeSchema,
    "market_value_parameters_vhe_param_woningen_parkeren": VheParamWoningenParkerenSchema,
}


@dg.asset(
    key_prefix=["tms"],
    name="market_value_parameters_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Download marktwaardeparameters Excel files to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_market_value_parameters_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch marktwaardeparameters Excel from API and persist to disk per round."""
    return download_round_bytes_files(
        context,
        config,
        company_valuation_data,
        report_key="market_value_parameters",
        fetch_fn=lambda company_id, meta: tms_api.fetch_market_value_parameters(company_id, meta.valuation_round_id),
    )


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"market_value_parameters_files": dg.AssetIn(key=dg.AssetKey(["tms", "market_value_parameters_files"]))},
)
def tms_market_value_parameters(context: dg.AssetExecutionContext, market_value_parameters_files: pd.DataFrame):
    """Parse marktwaardeparameters Excel from disk and yield one asset per sheet."""
    yield from parse_and_yield_multi_sheet_results(
        context,
        market_value_parameters_files,
        sheet_map=_SHEET_MAP,
        schema_by_output=_SCHEMA_BY_OUTPUT,
        skiprows=[1],
        prepare_frame=lambda output_name, df: align_to_schema_columns(df, _SCHEMA_BY_OUTPUT[output_name]),
    )
