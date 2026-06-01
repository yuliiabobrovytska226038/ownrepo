"""Parameteroverzicht — 24 sheets from the parameters-report Excel.

This report uses skiprows=2 because the first two rows are headers.

Legacy tables: Parameteroverzicht_Algemene_parameters, _Markthuur_Woningen_Parkeren,
    _Markthuur_BOG_MOG_ZOG, _Markthuurstijging, _Huurstijging_Woningen_Parkeren,
    _Leegwaarde, _Leegwaardestijging, _Disconteringsvoet, _Exit_yield,
    _Mutatiegraad, _Huurbeklemming, _Erfpacht_Woningen_Parkeren,
    _Erfpacht_BOG_MOG_ZOG, _Verkoopbeperking_Woningen,
    _Onderhoud_Woningen_Parkeren, _Onderhoud_BOG_MOG_ZOG, _Maximale_huur,
    _Splitsingskosten, _EPV, _Overige_kosten_en_opbrengsten,
    _Overige_exploitatielasten, _Leegstand_Woningen_Parkeren, _Scenario,
    _Schem_vrijheidsgraden_BOG
"""

import dagster as dg
import pandas as pd

from ..partitions import BACKFILL_POLICY, company_partitions
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import (
    ParametersOverviewAlgemeneParametersSchema,
    ParametersOverviewDisconteringsvoetSchema,
    ParametersOverviewEpvSchema,
    ParametersOverviewErfpachtBogMogZogSchema,
    ParametersOverviewErfpachtWoningenParkerenSchema,
    ParametersOverviewExitYieldSchema,
    ParametersOverviewHuurbekklemingSchema,
    ParametersOverviewHuurstijgingWoningenParkerenSchema,
    ParametersOverviewLeegstandWoningenParkerenSchema,
    ParametersOverviewLeegwaardeSchema,
    ParametersOverviewLeegwaardestijgingSchema,
    ParametersOverviewMarkthuurBogMogZogSchema,
    ParametersOverviewMarkthuurstijgingSchema,
    ParametersOverviewMarkthuurWoningenParkerenSchema,
    ParametersOverviewMaximaleHuurSchema,
    ParametersOverviewMutatiegraadSchema,
    ParametersOverviewOnderhoudBogMogZogSchema,
    ParametersOverviewOnderhoudWoningenParkerenSchema,
    ParametersOverviewOverigeExploitatielastenSchema,
    ParametersOverviewOverigeKostenEnOpbrengsten,
    ParametersOverviewScenarioSchema,
    ParametersOverviewSchemVrijheidsgradenBogSchema,
    ParametersOverviewSplitsingskostenSchema,
    ParametersOverviewVerkoopbeperkingWoningenSchema,
)
from ..utils.tms_report_utils import (
    COMPANY_VALUATION_DATA_DEP,
    RawReportConfig,
    build_sheet_asset_outs,
    download_round_report_files,
    parse_and_yield_multi_sheet_results,
)

# (output_name, legacy_table, (candidate_sheet_names...))
_SHEET_MAP: list[tuple[str, str, tuple[str, ...]]] = [
    ("parameters_overview_algemene_parameters", "Parameteroverzicht_Algemene_parameters", ("Algemene parameters",)),
    ("parameters_overview_markthuur_woningen_parkeren", "Parameteroverzicht_Markthuur_Woningen_Parkeren", ("Markthuur Woningen Parkeren",)),
    ("parameters_overview_markthuur_bog_mog_zog", "Parameteroverzicht_Markthuur_BOG_MOG_ZOG", ("Markthuur BOG MOG ZOG",)),
    ("parameters_overview_markthuurstijging", "Parameteroverzicht_Markthuurstijging", ("Markthuurstijging",)),
    (
        "parameters_overview_huurstijging_woningen_parkeren",
        "Parameteroverzicht_Huurstijging_Woningen_Parkeren",
        ("Huurstijging Woningen Parkeren", "Reguliere huurstijging Woningen Parkeren", "Reguliere huurstijging Woningen"),
    ),
    ("parameters_overview_leegwaarde", "Parameteroverzicht_Leegwaarde", ("Leegwaarde",)),
    ("parameters_overview_leegwaardestijging", "Parameteroverzicht_Leegwaardestijging", ("Leegwaardestijging",)),
    ("parameters_overview_disconteringsvoet", "Parameteroverzicht_Disconteringsvoet", ("Disconteringsvoet",)),
    ("parameters_overview_exit_yield", "Parameteroverzicht_Exit_yield", ("Exit yield",)),
    ("parameters_overview_mutatiegraad", "Parameteroverzicht_Mutatiegraad", ("Mutatiegraad",)),
    ("parameters_overview_huurbeklemming", "Parameteroverzicht_Huurbeklemming", ("Huurbeklemming",)),
    ("parameters_overview_erfpacht_woningen_parkeren", "Parameteroverzicht_Erfpacht_Woningen_Parkeren", ("Erfpacht Woningen Parkeren",)),
    ("parameters_overview_erfpacht_bog_mog_zog", "Parameteroverzicht_Erfpacht_BOG_MOG_ZOG", ("Erfpacht BOG MOG ZOG",)),
    ("parameters_overview_verkoopbeperking_woningen", "Parameteroverzicht_Verkoopbeperking_Woningen", ("Verkoopbeperking Woningen",)),
    ("parameters_overview_onderhoud_woningen_parkeren", "Parameteroverzicht_Onderhoud_Woningen_Parkeren", ("Onderhoud Woningen Parkeren",)),
    ("parameters_overview_onderhoud_bog_mog_zog", "Parameteroverzicht_Onderhoud_BOG_MOG_ZOG", ("Onderhoud BOG MOG ZOG",)),
    ("parameters_overview_maximale_huur", "Parameteroverzicht_Maximale_huur", ("Maximale huur",)),
    ("parameters_overview_splitsingskosten", "Parameteroverzicht_Splitsingskosten", ("Splitsingskosten",)),
    ("parameters_overview_epv", "Parameteroverzicht_EPV", ("EPV",)),
    ("parameters_overview_overige_kosten_en_opbrengsten", "Parameteroverzicht_Overige_kosten_en_opbrengsten", ("Overige kosten en opbrengsten",)),
    ("parameters_overview_overige_exploitatielasten", "Parameteroverzicht_Overige_exploitatielasten", ("Overige exploitatielasten",)),
    ("parameters_overview_leegstand_woningen_parkeren", "Parameteroverzicht_Leegstand_Woningen_Parkeren", ("Leegstand Woningen Parkeren",)),
    ("parameters_overview_scenario", "Parameteroverzicht_Scenario", ("Scenario",)),
    ("parameters_overview_schem_vrijheidsgraden_bog", "Parameteroverzicht_Schem_vrijheidsgraden_BOG", ("Schem. vrijheidsgraden BOG", "Schem vrijheidsgraden BOG")),
]
_SCHEMA_BY_OUTPUT = {
    "parameters_overview_algemene_parameters": ParametersOverviewAlgemeneParametersSchema,
    "parameters_overview_markthuur_woningen_parkeren": ParametersOverviewMarkthuurWoningenParkerenSchema,
    "parameters_overview_markthuur_bog_mog_zog": ParametersOverviewMarkthuurBogMogZogSchema,
    "parameters_overview_markthuurstijging": ParametersOverviewMarkthuurstijgingSchema,
    "parameters_overview_huurstijging_woningen_parkeren": ParametersOverviewHuurstijgingWoningenParkerenSchema,
    "parameters_overview_leegwaarde": ParametersOverviewLeegwaardeSchema,
    "parameters_overview_leegwaardestijging": ParametersOverviewLeegwaardestijgingSchema,
    "parameters_overview_disconteringsvoet": ParametersOverviewDisconteringsvoetSchema,
    "parameters_overview_exit_yield": ParametersOverviewExitYieldSchema,
    "parameters_overview_mutatiegraad": ParametersOverviewMutatiegraadSchema,
    "parameters_overview_huurbeklemming": ParametersOverviewHuurbekklemingSchema,
    "parameters_overview_erfpacht_woningen_parkeren": ParametersOverviewErfpachtWoningenParkerenSchema,
    "parameters_overview_erfpacht_bog_mog_zog": ParametersOverviewErfpachtBogMogZogSchema,
    "parameters_overview_verkoopbeperking_woningen": ParametersOverviewVerkoopbeperkingWoningenSchema,
    "parameters_overview_onderhoud_woningen_parkeren": ParametersOverviewOnderhoudWoningenParkerenSchema,
    "parameters_overview_onderhoud_bog_mog_zog": ParametersOverviewOnderhoudBogMogZogSchema,
    "parameters_overview_maximale_huur": ParametersOverviewMaximaleHuurSchema,
    "parameters_overview_splitsingskosten": ParametersOverviewSplitsingskostenSchema,
    "parameters_overview_epv": ParametersOverviewEpvSchema,
    "parameters_overview_overige_kosten_en_opbrengsten": ParametersOverviewOverigeKostenEnOpbrengsten,
    "parameters_overview_overige_exploitatielasten": ParametersOverviewOverigeExploitatielastenSchema,
    "parameters_overview_leegstand_woningen_parkeren": ParametersOverviewLeegstandWoningenParkerenSchema,
    "parameters_overview_scenario": ParametersOverviewScenarioSchema,
    "parameters_overview_schem_vrijheidsgraden_bog": ParametersOverviewSchemVrijheidsgradenBogSchema,
}


@dg.asset(
    key_prefix=["tms"],
    name="parameters_overview_files",
    compute_kind="api",
    group_name="tms_downloads",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    metadata={"partition_expr": "Corporatie"},
    description="Trigger/poll/download parameteroverzicht Excel to disk for each company round.",
    deps=[COMPANY_VALUATION_DATA_DEP],
    ins={"company_valuation_data": dg.AssetIn(key=dg.AssetKey(["tms", "company_valuation_data"]))},
)
def tms_parameters_overview_files(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource, company_valuation_data: pd.DataFrame) -> pd.DataFrame:
    """Trigger, poll, and download parameteroverzicht Excel per round (30-90s per company); persist path to disk."""
    return download_round_report_files(
        context,
        config,
        company_valuation_data,
        report_key="parameters_overview",
        fetch_fn=lambda company_id, meta, path: tms_api.fetch_parameters_report(company_id, meta.valuation_round_id, path),
    )


@dg.multi_asset(
    outs=build_sheet_asset_outs(_SHEET_MAP, report_label="Parameteroverzicht"),
    compute_kind="python",
    partitions_def=company_partitions,
    backfill_policy=BACKFILL_POLICY,
    ins={"parameters_overview_files": dg.AssetIn(key=dg.AssetKey(["tms", "parameters_overview_files"]))},
)
def tms_parameters_overview(context: dg.AssetExecutionContext, parameters_overview_files: pd.DataFrame):
    """Parse parameteroverzicht Excel from disk and yield one asset per sheet (24 sheets)."""
    yield from parse_and_yield_multi_sheet_results(context, parameters_overview_files, sheet_map=_SHEET_MAP, schema_by_output=_SCHEMA_BY_OUTPUT, skiprows=2)
