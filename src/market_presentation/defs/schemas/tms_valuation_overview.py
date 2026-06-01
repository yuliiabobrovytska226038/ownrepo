"""Pandera schemas for valuation overview tables (VHE and complex level)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


class ValuationOverviewVheSchema(pa.DataFrameModel):
    """Schema for tms.valuation_overview_vhe (41 cols) — waardeoverzicht per VHE."""

    vhe_nr: Series[str] = pa.Field(alias="VHE-nr")
    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    deelportefeuille: Series[str] = pa.Field(alias="Deelportefeuille", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    postcode: Series[str] = pa.Field(alias="Postcode", nullable=True)
    scenario: Series[str] = pa.Field(alias="Scenario", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bouwjaar: Series[float] = pa.Field(alias="Bouwjaar", nullable=True, coerce=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    energielabel: Series[str] = pa.Field(alias="Energielabel", nullable=True)
    huur: Series[float] = pa.Field(alias="Huur", nullable=True, coerce=True)
    incentives: Series[float] = pa.Field(alias="Incentives", nullable=True, coerce=True)
    huurderving: Series[float] = pa.Field(alias="Huurderving", nullable=True, coerce=True)
    huurderving_mutatieleegstand: Series[float] = pa.Field(alias="Huurderving mutatieleegstand", nullable=True, coerce=True)
    huurderving_aanvangsleegstand: Series[float] = pa.Field(alias="Huurderving aanvangsleegstand", nullable=True, coerce=True)
    mutatiekosten: Series[float] = pa.Field(alias="Mutatiekosten", nullable=True, coerce=True)
    onderhoud: Series[float] = pa.Field(alias="Onderhoud", nullable=True, coerce=True)
    mutatieonderhoud_bij_doorexp: Series[float] = pa.Field(alias="Mutatieonderhoud bij doorexp.", nullable=True, coerce=True)
    mutatieonderhoud_bij_verkoop: Series[float] = pa.Field(alias="Mutatieonderhoud bij verkoop", nullable=True, coerce=True)
    beheerkosten: Series[float] = pa.Field(alias="Beheerkosten", nullable=True, coerce=True)
    belastingen_en_verzekeringen: Series[float] = pa.Field(alias="Belastingen en verzekeringen", nullable=True, coerce=True)
    ozb: Series[float] = pa.Field(alias="OZB", nullable=True, coerce=True)
    servicekosten: Series[float] = pa.Field(alias="Servicekosten", nullable=True, coerce=True)
    erfpacht: Series[float] = pa.Field(alias="Erfpacht", nullable=True, coerce=True)
    splitsingskosten: Series[float] = pa.Field(alias="Splitsingskosten", nullable=True, coerce=True)
    verplichting_efg_labels: Series[float] = pa.Field(alias="Verplichting EFG-labels", nullable=True, coerce=True)
    energieprestatievergoeding: Series[float] = pa.Field(alias="Energieprestatievergoeding", nullable=True, coerce=True)
    overige_opbrengsten: Series[float] = pa.Field(alias="Overige opbrengsten", nullable=True, coerce=True)
    overige_kosten: Series[float] = pa.Field(alias="Overige kosten", nullable=True, coerce=True)
    verkoopkosten: Series[float] = pa.Field(alias="Verkoopkosten", nullable=True, coerce=True)
    verkoopopbrengsten: Series[float] = pa.Field(alias="Verkoopopbrengsten", nullable=True, coerce=True)
    eindwaarde: Series[float] = pa.Field(alias="Eindwaarde", nullable=True, coerce=True)
    effect_afkapping: Series[float] = pa.Field(alias="Effect afkapping", nullable=True, coerce=True)
    bruto_marktwaarde: Series[float] = pa.Field(alias="Bruto marktwaarde", nullable=True, coerce=True)
    overdrachtskosten: Series[float] = pa.Field(alias="Overdrachtskosten", nullable=True, coerce=True)
    netto_marktwaarde: Series[float] = pa.Field(alias="Netto marktwaarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ValuationOverviewComplexSchema(pa.DataFrameModel):
    """Schema for tms.valuation_overview_complex (39 cols) — waardeoverzicht per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex")
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    deelportefeuille: Series[str] = pa.Field(alias="Deelportefeuille", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    scenario: Series[str] = pa.Field(alias="Scenario", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    aantal_vhes: Series[float] = pa.Field(alias="Aantal VHE's", nullable=True, coerce=True)
    aantal_vhes_leegstaand: Series[float] = pa.Field(alias="Aantal VHE's leegstaand", nullable=True, coerce=True)
    percentage_leegstaand: Series[float] = pa.Field(alias="Percentage leegstaand in complex", nullable=True, coerce=True)
    energielabel: Series[str] = pa.Field(alias="Energielabel", nullable=True)
    huur: Series[float] = pa.Field(alias="Huur", nullable=True, coerce=True)
    incentives: Series[float] = pa.Field(alias="Incentives", nullable=True, coerce=True)
    huurderving: Series[float] = pa.Field(alias="Huurderving", nullable=True, coerce=True)
    huurderving_mutatieleegstand: Series[float] = pa.Field(alias="Huurderving mutatieleegstand", nullable=True, coerce=True)
    huurderving_aanvangsleegstand: Series[float] = pa.Field(alias="Huurderving aanvangsleegstand", nullable=True, coerce=True)
    mutatiekosten: Series[float] = pa.Field(alias="Mutatiekosten", nullable=True, coerce=True)
    onderhoud: Series[float] = pa.Field(alias="Onderhoud", nullable=True, coerce=True)
    mutatieonderhoud_bij_doorexp: Series[float] = pa.Field(alias="Mutatieonderhoud bij doorexp.", nullable=True, coerce=True)
    mutatieonderhoud_bij_verkoop: Series[float] = pa.Field(alias="Mutatieonderhoud bij verkoop", nullable=True, coerce=True)
    beheerkosten: Series[float] = pa.Field(alias="Beheerkosten", nullable=True, coerce=True)
    belastingen_en_verzekeringen: Series[float] = pa.Field(alias="Belastingen en verzekeringen", nullable=True, coerce=True)
    ozb: Series[float] = pa.Field(alias="OZB", nullable=True, coerce=True)
    servicekosten: Series[float] = pa.Field(alias="Servicekosten", nullable=True, coerce=True)
    erfpacht: Series[float] = pa.Field(alias="Erfpacht", nullable=True, coerce=True)
    splitsingskosten: Series[float] = pa.Field(alias="Splitsingskosten", nullable=True, coerce=True)
    verplichting_efg_labels: Series[float] = pa.Field(alias="Verplichting EFG-labels", nullable=True, coerce=True)
    energieprestatievergoeding: Series[float] = pa.Field(alias="Energieprestatievergoeding", nullable=True, coerce=True)
    overige_opbrengsten: Series[float] = pa.Field(alias="Overige opbrengsten", nullable=True, coerce=True)
    overige_kosten: Series[float] = pa.Field(alias="Overige kosten", nullable=True, coerce=True)
    verkoopkosten: Series[float] = pa.Field(alias="Verkoopkosten", nullable=True, coerce=True)
    verkoopopbrengsten: Series[float] = pa.Field(alias="Verkoopopbrengsten", nullable=True, coerce=True)
    eindwaarde: Series[float] = pa.Field(alias="Eindwaarde", nullable=True, coerce=True)
    effect_afkapping: Series[float] = pa.Field(alias="Effect afkapping", nullable=True, coerce=True)
    bruto_marktwaarde: Series[float] = pa.Field(alias="Bruto marktwaarde", nullable=True, coerce=True)
    overdrachtskosten: Series[float] = pa.Field(alias="Overdrachtskosten", nullable=True, coerce=True)
    netto_marktwaarde: Series[float] = pa.Field(alias="Netto marktwaarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True
