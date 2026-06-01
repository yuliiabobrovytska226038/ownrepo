"""Pandera DataFrameModel schemas for TMS pipeline data contracts.

Each schema defines the expected column names and types for raw TMS tables.
Schemas use ``coerce=True`` so Pandera converts types automatically, and
``strict=False`` to allow extra columns from future API changes.
"""

from market_presentation.defs.schemas.tms_complex_and_mapping import (
    ComplexKenmerkenSchema,
    ComplexReferencesSchema,
    MarketValueBasisSchema,
    RentalUnitMappingSchema,
    TmsVerschillenanalyseSchema,
)
from market_presentation.defs.schemas.tms_external import (
    CbsCoropRegionsSchema,
    CompanyListSchema,
    CompanyValuationDataSchema,
    PostalCodeMappingSchema,
)
from market_presentation.defs.schemas.tms_market_value_params import (
    ComplexparamBogMogZogSchema,
    ComplexparamWoningenParkerenSchema,
    VheParamBenaderingsmethodeSchema,
    VheParamBogMogZogSchema,
    VheParamWoningenParkerenSchema,
)
from market_presentation.defs.schemas.tms_parameters_overview import (
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
from market_presentation.defs.schemas.tms_policy_value import (
    PolicyValueParametersSchema,
    PolicyValueReportComplexSchema,
    PolicyValueReportVheSchema,
)
from market_presentation.defs.schemas.tms_ratio_rapport import (
    RatioRapportComplexSchema,
    RatioRapportVheSchema,
)
from market_presentation.defs.schemas.tms_valuation_overview import (
    ValuationOverviewComplexSchema,
    ValuationOverviewVheSchema,
)
from market_presentation.defs.schemas.tms_vastgoedgegevens import (
    SturingscomplexenSchema,
    VheGegevensSchema,
    VovVerledenSchema,
    WaarderingscomplexenSchema,
)

__all__ = [
    # tms_vastgoedgegevens
    "VheGegevensSchema",
    "WaarderingscomplexenSchema",
    "SturingscomplexenSchema",
    "VovVerledenSchema",
    # tms_market_value_params
    "VheParamWoningenParkerenSchema",
    "VheParamBogMogZogSchema",
    "VheParamBenaderingsmethodeSchema",
    "ComplexparamWoningenParkerenSchema",
    "ComplexparamBogMogZogSchema",
    # tms_policy_value
    "PolicyValueParametersSchema",
    "PolicyValueReportVheSchema",
    "PolicyValueReportComplexSchema",
    # tms_valuation_overview
    "ValuationOverviewVheSchema",
    "ValuationOverviewComplexSchema",
    # tms_ratio_rapport
    "RatioRapportVheSchema",
    "RatioRapportComplexSchema",
    # tms_complex_and_mapping
    "ComplexKenmerkenSchema",
    "ComplexReferencesSchema",
    "MarketValueBasisSchema",
    "RentalUnitMappingSchema",
    "TmsVerschillenanalyseSchema",
    # tms_parameters_overview
    "ParametersOverviewAlgemeneParametersSchema",
    "ParametersOverviewDisconteringsvoetSchema",
    "ParametersOverviewEpvSchema",
    "ParametersOverviewErfpachtBogMogZogSchema",
    "ParametersOverviewErfpachtWoningenParkerenSchema",
    "ParametersOverviewExitYieldSchema",
    "ParametersOverviewHuurbekklemingSchema",
    "ParametersOverviewHuurstijgingWoningenParkerenSchema",
    "ParametersOverviewLeegstandWoningenParkerenSchema",
    "ParametersOverviewLeegwaardeSchema",
    "ParametersOverviewLeegwaardestijgingSchema",
    "ParametersOverviewMarkthuurBogMogZogSchema",
    "ParametersOverviewMarkthuurWoningenParkerenSchema",
    "ParametersOverviewMarkthuurstijgingSchema",
    "ParametersOverviewMaximaleHuurSchema",
    "ParametersOverviewMutatiegraadSchema",
    "ParametersOverviewOnderhoudBogMogZogSchema",
    "ParametersOverviewOnderhoudWoningenParkerenSchema",
    "ParametersOverviewOverigeExploitatielastenSchema",
    "ParametersOverviewOverigeKostenEnOpbrengsten",
    "ParametersOverviewScenarioSchema",
    "ParametersOverviewSchemVrijheidsgradenBogSchema",
    "ParametersOverviewSplitsingskostenSchema",
    "ParametersOverviewVerkoopbeperkingWoningenSchema",
    # tms_external
    "PostalCodeMappingSchema",
    "CbsCoropRegionsSchema",
    "CompanyValuationDataSchema",
    "CompanyListSchema",
]
