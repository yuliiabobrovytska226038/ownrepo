"""Pandera schemas for all 24 parameters_overview tables.

Each parameteroverzicht sheet has a corresponding schema. VHE-level tables (10)
and complex-level tables (14) are all defined here.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series

# ---------------------------------------------------------------------------
# Shared VHE-level base columns (used by most VHE-level parameter sheets)
# ---------------------------------------------------------------------------
# Cannot use inheritance for base columns in Pandera DataFrameModel, but we
# document the pattern: VHE tables have Waarderingscomplex, Complexnaam, VHE-nr,
# Adres, Model, Handboektype, Classificatie + Corporatie, Jaar, Peildatum.
# Complex tables have Waarderingscomplex, Complexnaam, Straatnaam, Model + metadata.


class ParametersOverviewAlgemeneParametersSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_algemene_parameters (21 cols) — general market parameters."""

    parameter: Series[str] = pa.Field(alias="Parameter", nullable=True)
    bron: Series[str] = pa.Field(alias="Bron", nullable=True)
    jaar_min1_handboek: Series[float] = pa.Field(alias="Jaar -1 handboek", nullable=True, coerce=True)
    jaar_0_handboek: Series[float] = pa.Field(alias="Jaar 0 handboek", nullable=True, coerce=True)
    jaar_1_handboek: Series[float] = pa.Field(alias="Jaar 1 handboek", nullable=True, coerce=True)
    jaar_2_handboek: Series[float] = pa.Field(alias="Jaar 2 handboek", nullable=True, coerce=True)
    jaar_3_handboek: Series[float] = pa.Field(alias="Jaar 3 handboek", nullable=True, coerce=True)
    jaar_4_handboek: Series[float] = pa.Field(alias="Jaar 4 handboek", nullable=True, coerce=True)
    jaar_5_handboek: Series[float] = pa.Field(alias="Jaar 5 handboek", nullable=True, coerce=True)
    jaar_6_ev_handboek: Series[float] = pa.Field(alias="Jaar 6 e.v. handboek", nullable=True, coerce=True)
    jaar_min1: Series[float] = pa.Field(alias="Jaar -1", nullable=True, coerce=True)
    jaar_0: Series[float] = pa.Field(alias="Jaar 0", nullable=True, coerce=True)
    jaar_1: Series[float] = pa.Field(alias="Jaar 1", nullable=True, coerce=True)
    jaar_2: Series[float] = pa.Field(alias="Jaar 2", nullable=True, coerce=True)
    jaar_3: Series[float] = pa.Field(alias="Jaar 3", nullable=True, coerce=True)
    jaar_4: Series[float] = pa.Field(alias="Jaar 4", nullable=True, coerce=True)
    jaar_5: Series[float] = pa.Field(alias="Jaar 5", nullable=True, coerce=True)
    jaar_6_ev: Series[float] = pa.Field(alias="Jaar 6 e.v.", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewDisconteringsvoetSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_disconteringsvoet (15 cols) — discount rate per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_dv: Series[str] = pa.Field(alias="Bron disconteringsvoet (DV)", nullable=True)
    dv_doorexploiteren_handboek: Series[float] = pa.Field(alias="DV doorexploiteren handboek", nullable=True, coerce=True)
    dv_uitponden_handboek: Series[float] = pa.Field(alias="DV uitponden handboek", nullable=True, coerce=True)
    dv_doorexploiteren: Series[float] = pa.Field(alias="DV doorexploiteren", nullable=True, coerce=True)
    dv_uitponden: Series[float] = pa.Field(alias="DV uitponden", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewEpvSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_epv (12 cols) — energy performance premium per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron: Series[str] = pa.Field(alias="Bron", nullable=True)
    epv_waarde: Series[float] = pa.Field(alias="EPV waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewErfpachtBogMogZogSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_erfpacht_bog_mog_zog (28 cols) — ground lease per BOG/MOG/ZOG VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    erfpacht_ep: Series[str] = pa.Field(alias="Erfpacht (EP)", nullable=True)
    ep_waardecorrectie_per_m2_bvo: Series[float] = pa.Field(alias="EP waardecorrectie per m² BVO", nullable=True, coerce=True)
    ep_jaarlijks_indexeren: Series[str] = pa.Field(alias="EP jaarlijks indexeren", nullable=True)
    ep_per_m2_bvo_jaar_1: Series[float] = pa.Field(alias="EP per m² BVO jaar 1", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_2: Series[float] = pa.Field(alias="EP per m² BVO jaar 2", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_3: Series[float] = pa.Field(alias="EP per m² BVO jaar 3", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_4: Series[float] = pa.Field(alias="EP per m² BVO jaar 4", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_5: Series[float] = pa.Field(alias="EP per m² BVO jaar 5", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_6: Series[float] = pa.Field(alias="EP per m² BVO jaar 6", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_7: Series[float] = pa.Field(alias="EP per m² BVO jaar 7", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_8: Series[float] = pa.Field(alias="EP per m² BVO jaar 8", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_9: Series[float] = pa.Field(alias="EP per m² BVO jaar 9", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_10: Series[float] = pa.Field(alias="EP per m² BVO jaar 10", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_11: Series[float] = pa.Field(alias="EP per m² BVO jaar 11", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_12: Series[float] = pa.Field(alias="EP per m² BVO jaar 12", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_13: Series[float] = pa.Field(alias="EP per m² BVO jaar 13", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_14: Series[float] = pa.Field(alias="EP per m² BVO jaar 14", nullable=True, coerce=True)
    ep_per_m2_bvo_jaar_15: Series[float] = pa.Field(alias="EP per m² BVO jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewErfpachtWoningenParkerenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_erfpacht_woningen_parkeren (45 cols) — ground lease per Woningen/Parkeren VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    erfpacht_ep: Series[str] = pa.Field(alias="Erfpacht (EP)", nullable=True)
    ep_waardecorrectie_doorexploiteren: Series[float] = pa.Field(alias="EP waardecorrectie doorexploiteren", nullable=True, coerce=True)
    ep_waardecorrectie_uitponden: Series[float] = pa.Field(alias="EP waardecorrectie uitponden", nullable=True, coerce=True)
    ep_suppletie_bij_verkoop: Series[float] = pa.Field(alias="EP suppletie bij verkoop", nullable=True, coerce=True)
    ep_jaarlijks_indexeren: Series[str] = pa.Field(alias="EP jaarlijks indexeren", nullable=True)
    ep_doorexploiteren_jaar_1: Series[float] = pa.Field(alias="EP doorexploiteren jaar 1", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_2: Series[float] = pa.Field(alias="EP doorexploiteren jaar 2", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_3: Series[float] = pa.Field(alias="EP doorexploiteren jaar 3", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_4: Series[float] = pa.Field(alias="EP doorexploiteren jaar 4", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_5: Series[float] = pa.Field(alias="EP doorexploiteren jaar 5", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_6: Series[float] = pa.Field(alias="EP doorexploiteren jaar 6", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_7: Series[float] = pa.Field(alias="EP doorexploiteren jaar 7", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_8: Series[float] = pa.Field(alias="EP doorexploiteren jaar 8", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_9: Series[float] = pa.Field(alias="EP doorexploiteren jaar 9", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_10: Series[float] = pa.Field(alias="EP doorexploiteren jaar 10", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_11: Series[float] = pa.Field(alias="EP doorexploiteren jaar 11", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_12: Series[float] = pa.Field(alias="EP doorexploiteren jaar 12", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_13: Series[float] = pa.Field(alias="EP doorexploiteren jaar 13", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_14: Series[float] = pa.Field(alias="EP doorexploiteren jaar 14", nullable=True, coerce=True)
    ep_doorexploiteren_jaar_15: Series[float] = pa.Field(alias="EP doorexploiteren jaar 15", nullable=True, coerce=True)
    ep_uitponden_jaar_1: Series[float] = pa.Field(alias="EP uitponden jaar 1", nullable=True, coerce=True)
    ep_uitponden_jaar_2: Series[float] = pa.Field(alias="EP uitponden jaar 2", nullable=True, coerce=True)
    ep_uitponden_jaar_3: Series[float] = pa.Field(alias="EP uitponden jaar 3", nullable=True, coerce=True)
    ep_uitponden_jaar_4: Series[float] = pa.Field(alias="EP uitponden jaar 4", nullable=True, coerce=True)
    ep_uitponden_jaar_5: Series[float] = pa.Field(alias="EP uitponden jaar 5", nullable=True, coerce=True)
    ep_uitponden_jaar_6: Series[float] = pa.Field(alias="EP uitponden jaar 6", nullable=True, coerce=True)
    ep_uitponden_jaar_7: Series[float] = pa.Field(alias="EP uitponden jaar 7", nullable=True, coerce=True)
    ep_uitponden_jaar_8: Series[float] = pa.Field(alias="EP uitponden jaar 8", nullable=True, coerce=True)
    ep_uitponden_jaar_9: Series[float] = pa.Field(alias="EP uitponden jaar 9", nullable=True, coerce=True)
    ep_uitponden_jaar_10: Series[float] = pa.Field(alias="EP uitponden jaar 10", nullable=True, coerce=True)
    ep_uitponden_jaar_11: Series[float] = pa.Field(alias="EP uitponden jaar 11", nullable=True, coerce=True)
    ep_uitponden_jaar_12: Series[float] = pa.Field(alias="EP uitponden jaar 12", nullable=True, coerce=True)
    ep_uitponden_jaar_13: Series[float] = pa.Field(alias="EP uitponden jaar 13", nullable=True, coerce=True)
    ep_uitponden_jaar_14: Series[float] = pa.Field(alias="EP uitponden jaar 14", nullable=True, coerce=True)
    ep_uitponden_jaar_15: Series[float] = pa.Field(alias="EP uitponden jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewExitYieldSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_exit_yield (15 cols)."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_ey: Series[str] = pa.Field(alias="Bron exit yield (EY)", nullable=True)
    ey_doorexploiteren_handboek: Series[float] = pa.Field(alias="EY doorexploiteren handboek", nullable=True, coerce=True)
    ey_uitponden_handboek: Series[float] = pa.Field(alias="EY uitponden handboek", nullable=True, coerce=True)
    ey_doorexploiteren: Series[float] = pa.Field(alias="EY doorexploiteren", nullable=True, coerce=True)
    ey_uitponden: Series[float] = pa.Field(alias="EY uitponden", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewHuurbekklemingSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_huurbeklemming (13 cols) — rent price cap per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_huurbeklemming: Series[str] = pa.Field(alias="Bron huurbeklemming", nullable=True)
    laatste_jaar_huurbeklemming: Series[float] = pa.Field(alias="Laatste jaar huurbeklemming", nullable=True, coerce=True)
    huurbeklemming: Series[float] = pa.Field(alias="Huurbeklemming", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewHuurstijgingWoningenParkerenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_huurstijging_woningen_parkeren (36 cols) — rent increase per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_hs: Series[str] = pa.Field(alias="Bron huurstijging (HS)", nullable=True)
    hs_jaar_1_handboek: Series[float] = pa.Field(alias="HS jaar 1 handboek", nullable=True, coerce=True)
    hs_jaar_2_handboek: Series[float] = pa.Field(alias="HS jaar 2 handboek", nullable=True, coerce=True)
    hs_jaar_3_handboek: Series[float] = pa.Field(alias="HS jaar 3 handboek", nullable=True, coerce=True)
    hs_jaar_4_handboek: Series[float] = pa.Field(alias="HS jaar 4 handboek", nullable=True, coerce=True)
    hs_jaar_5_handboek: Series[float] = pa.Field(alias="HS jaar 5 handboek", nullable=True, coerce=True)
    hs_jaar_6_handboek: Series[float] = pa.Field(alias="HS jaar 6 handboek", nullable=True, coerce=True)
    hs_jaar_1: Series[float] = pa.Field(alias="HS jaar 1", nullable=True, coerce=True)
    hs_jaar_2: Series[float] = pa.Field(alias="HS jaar 2", nullable=True, coerce=True)
    hs_jaar_3: Series[float] = pa.Field(alias="HS jaar 3", nullable=True, coerce=True)
    hs_jaar_4: Series[float] = pa.Field(alias="HS jaar 4", nullable=True, coerce=True)
    hs_jaar_5: Series[float] = pa.Field(alias="HS jaar 5", nullable=True, coerce=True)
    hs_jaar_6: Series[float] = pa.Field(alias="HS jaar 6", nullable=True, coerce=True)
    bron_hso: Series[str] = pa.Field(alias="Bron huurstijging opslag (HSO)", nullable=True)
    hso_jaar_1_handboek: Series[float] = pa.Field(alias="HSO jaar 1 handboek", nullable=True, coerce=True)
    hso_jaar_2_handboek: Series[float] = pa.Field(alias="HSO jaar 2 handboek", nullable=True, coerce=True)
    hso_jaar_3_handboek: Series[float] = pa.Field(alias="HSO jaar 3 handboek", nullable=True, coerce=True)
    hso_jaar_4_handboek: Series[float] = pa.Field(alias="HSO jaar 4 handboek", nullable=True, coerce=True)
    hso_jaar_5_handboek: Series[float] = pa.Field(alias="HSO jaar 5 handboek", nullable=True, coerce=True)
    hso_jaar_6_handboek: Series[float] = pa.Field(alias="HSO jaar 6 handboek", nullable=True, coerce=True)
    hso_jaar_1: Series[float] = pa.Field(alias="HSO jaar 1", nullable=True, coerce=True)
    hso_jaar_2: Series[float] = pa.Field(alias="HSO jaar 2", nullable=True, coerce=True)
    hso_jaar_3: Series[float] = pa.Field(alias="HSO jaar 3", nullable=True, coerce=True)
    hso_jaar_4: Series[float] = pa.Field(alias="HSO jaar 4", nullable=True, coerce=True)
    hso_jaar_5: Series[float] = pa.Field(alias="HSO jaar 5", nullable=True, coerce=True)
    hso_jaar_6: Series[float] = pa.Field(alias="HSO jaar 6", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewLeegstandWoningenParkerenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_leegstand_woningen_parkeren (16 cols) — vacancy per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_al: Series[str] = pa.Field(alias="Bron aanvangsleegstand (AL)", nullable=True)
    al_handboek_waarde_maanden: Series[float] = pa.Field(alias="AL handboek waarde (maanden)", nullable=True, coerce=True)
    al_waarde_maanden: Series[float] = pa.Field(alias="AL waarde (maanden)", nullable=True, coerce=True)
    bron_ml: Series[str] = pa.Field(alias="Bron mutatieleegstand (ML)", nullable=True)
    ml_handboek_waarde_maanden: Series[float] = pa.Field(alias="ML handboek waarde (maanden)", nullable=True, coerce=True)
    ml_waarde_maanden: Series[float] = pa.Field(alias="ML waarde (maanden)", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewLeegwaardeSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_leegwaarde (14 cols) — vacant possession value per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_lw: Series[str] = pa.Field(alias="Bron leegwaarde (LW)", nullable=True)
    woz_waarde: Series[float] = pa.Field(alias="WOZ-waarde", nullable=True, coerce=True)
    lw_handboek_waarde: Series[float] = pa.Field(alias="LW handboek waarde", nullable=True, coerce=True)
    lw_waarde: Series[float] = pa.Field(alias="LW waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewLeegwaardestijgingSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_leegwaardestijging (25 cols) — vacant possession value growth per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_lsv: Series[str] = pa.Field(alias="Bron leegwaardestijging verleden (LSV)", nullable=True)
    lsv_jaar_min1_handboek: Series[float] = pa.Field(alias="LSV jaar -1 handboek", nullable=True, coerce=True)
    lsv_jaar_0_handboek: Series[float] = pa.Field(alias="LSV jaar 0 handboek", nullable=True, coerce=True)
    lsv_jaar_min1: Series[float] = pa.Field(alias="LSV jaar -1", nullable=True, coerce=True)
    lsv_jaar_0: Series[float] = pa.Field(alias="LSV jaar 0", nullable=True, coerce=True)
    bron_ls: Series[str] = pa.Field(alias="Bron leegwaardestijging (LS)", nullable=True)
    ls_jaar_1_handboek: Series[float] = pa.Field(alias="LS jaar 1 handboek", nullable=True, coerce=True)
    ls_jaar_2_handboek: Series[float] = pa.Field(alias="LS jaar 2 handboek", nullable=True, coerce=True)
    ls_jaar_3_handboek: Series[float] = pa.Field(alias="LS jaar 3 handboek", nullable=True, coerce=True)
    ls_jaar_4_handboek: Series[float] = pa.Field(alias="LS jaar 4 handboek", nullable=True, coerce=True)
    ls_jaar_5_handboek: Series[float] = pa.Field(alias="LS jaar 5 handboek", nullable=True, coerce=True)
    ls_jaar_6_handboek: Series[float] = pa.Field(alias="LS jaar 6 handboek", nullable=True, coerce=True)
    ls_jaar_1: Series[float] = pa.Field(alias="LS jaar 1", nullable=True, coerce=True)
    ls_jaar_2: Series[float] = pa.Field(alias="LS jaar 2", nullable=True, coerce=True)
    ls_jaar_3: Series[float] = pa.Field(alias="LS jaar 3", nullable=True, coerce=True)
    ls_jaar_4: Series[float] = pa.Field(alias="LS jaar 4", nullable=True, coerce=True)
    ls_jaar_5: Series[float] = pa.Field(alias="LS jaar 5", nullable=True, coerce=True)
    ls_jaar_6: Series[float] = pa.Field(alias="LS jaar 6", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewMarkthuurBogMogZogSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_markthuur_bog_mog_zog (13 cols) — market rent per BOG/MOG/ZOG VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_mh: Series[str] = pa.Field(alias="Bron markthuur (MH)", nullable=True)
    mh_per_m2_vvo_handboek_waarde: Series[float] = pa.Field(alias="MH per m² VVO handboek waarde", nullable=True, coerce=True)
    mh_per_m2_vvo_waarde: Series[float] = pa.Field(alias="MH per m² VVO waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewMarkthuurWoningenParkerenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_markthuur_woningen_parkeren (13 cols) — market rent per Woningen/Parkeren VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_mh: Series[str] = pa.Field(alias="Bron markthuur (MH)", nullable=True)
    mh_handboek_waarde: Series[float] = pa.Field(alias="MH handboek waarde", nullable=True, coerce=True)
    mh_waarde: Series[float] = pa.Field(alias="MH waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewMarkthuurstijgingSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_markthuurstijging (20 cols) — market rent growth per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_mhs: Series[str] = pa.Field(alias="Bron markthuurstijging (MHS)", nullable=True)
    mhs_jaar_1_handboek: Series[float] = pa.Field(alias="MHS jaar 1 handboek", nullable=True, coerce=True)
    mhs_jaar_2_handboek: Series[float] = pa.Field(alias="MHS jaar 2 handboek", nullable=True, coerce=True)
    mhs_jaar_3_handboek: Series[float] = pa.Field(alias="MHS jaar 3 handboek", nullable=True, coerce=True)
    mhs_jaar_4_handboek: Series[float] = pa.Field(alias="MHS jaar 4 handboek", nullable=True, coerce=True)
    mhs_jaar_5_handboek: Series[float] = pa.Field(alias="MHS jaar 5 handboek", nullable=True, coerce=True)
    mhs_jaar_6_handboek: Series[float] = pa.Field(alias="MHS jaar 6 handboek", nullable=True, coerce=True)
    mhs_jaar_1: Series[float] = pa.Field(alias="MHS jaar 1", nullable=True, coerce=True)
    mhs_jaar_2: Series[float] = pa.Field(alias="MHS jaar 2", nullable=True, coerce=True)
    mhs_jaar_3: Series[float] = pa.Field(alias="MHS jaar 3", nullable=True, coerce=True)
    mhs_jaar_4: Series[float] = pa.Field(alias="MHS jaar 4", nullable=True, coerce=True)
    mhs_jaar_5: Series[float] = pa.Field(alias="MHS jaar 5", nullable=True, coerce=True)
    mhs_jaar_6: Series[float] = pa.Field(alias="MHS jaar 6", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewMaximaleHuurSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_maximale_huur (19 cols) — maximum rent per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_maximale_huur: Series[str] = pa.Field(alias="Bron maximale huur", nullable=True)
    wws_punten: Series[float] = pa.Field(alias="WWS punten", nullable=True, coerce=True)
    maximale_huur_input: Series[float] = pa.Field(alias="Maximale huur (input)", nullable=True, coerce=True)
    maximale_huur_excl_opslag: Series[float] = pa.Field(alias="Maximale huur excl. opslag in berekening", nullable=True, coerce=True)
    beschermd_bezit: Series[str] = pa.Field(alias="Beschermd bezit", nullable=True)
    startdatum_contract: Series[str] = pa.Field(alias="Startdatum contract", nullable=True, coerce=True)
    beschermd_bezit_opslag_huidig: Series[float] = pa.Field(alias="Beschermd bezit opslag huidig contract (%)", nullable=True, coerce=True)
    beschermd_bezit_opslag_nieuw: Series[float] = pa.Field(alias="Beschermd bezit opslag nieuw contract (%)", nullable=True, coerce=True)
    nieuwbouwopslag: Series[str] = pa.Field(alias="Nieuwbouwopslag", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewMutatiegraadSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_mutatiegraad (41 cols) — mutation rate per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_md: Series[str] = pa.Field(alias="Bron mutatiegraad doorexploiteren (MD)", nullable=True)
    md_handboek_waarde: Series[float] = pa.Field(alias="MD handboek waarde", nullable=True, coerce=True)
    md_waarde: Series[float] = pa.Field(alias="MD waarde", nullable=True, coerce=True)
    bron_mu: Series[str] = pa.Field(alias="Bron mutatiegraad uitponden (MU)", nullable=True)
    mu_jaar_1_handboek: Series[float] = pa.Field(alias="MU jaar 1 handboek", nullable=True, coerce=True)
    mu_jaar_2_handboek: Series[float] = pa.Field(alias="MU jaar 2 handboek", nullable=True, coerce=True)
    mu_jaar_3_handboek: Series[float] = pa.Field(alias="MU jaar 3 handboek", nullable=True, coerce=True)
    mu_jaar_4_handboek: Series[float] = pa.Field(alias="MU jaar 4 handboek", nullable=True, coerce=True)
    mu_jaar_5_handboek: Series[float] = pa.Field(alias="MU jaar 5 handboek", nullable=True, coerce=True)
    mu_jaar_6_handboek: Series[float] = pa.Field(alias="MU jaar 6 handboek", nullable=True, coerce=True)
    mu_jaar_7_handboek: Series[float] = pa.Field(alias="MU jaar 7 handboek", nullable=True, coerce=True)
    mu_jaar_8_handboek: Series[float] = pa.Field(alias="MU jaar 8 handboek", nullable=True, coerce=True)
    mu_jaar_9_handboek: Series[float] = pa.Field(alias="MU jaar 9 handboek", nullable=True, coerce=True)
    mu_jaar_10_handboek: Series[float] = pa.Field(alias="MU jaar 10 handboek", nullable=True, coerce=True)
    mu_jaar_11_handboek: Series[float] = pa.Field(alias="MU jaar 11 handboek", nullable=True, coerce=True)
    mu_jaar_12_handboek: Series[float] = pa.Field(alias="MU jaar 12 handboek", nullable=True, coerce=True)
    mu_jaar_13_handboek: Series[float] = pa.Field(alias="MU jaar 13 handboek", nullable=True, coerce=True)
    mu_jaar_14_handboek: Series[float] = pa.Field(alias="MU jaar 14 handboek", nullable=True, coerce=True)
    mu_jaar_15_handboek: Series[float] = pa.Field(alias="MU jaar 15 handboek", nullable=True, coerce=True)
    mu_jaar_1: Series[float] = pa.Field(alias="MU jaar 1", nullable=True, coerce=True)
    mu_jaar_2: Series[float] = pa.Field(alias="MU jaar 2", nullable=True, coerce=True)
    mu_jaar_3: Series[float] = pa.Field(alias="MU jaar 3", nullable=True, coerce=True)
    mu_jaar_4: Series[float] = pa.Field(alias="MU jaar 4", nullable=True, coerce=True)
    mu_jaar_5: Series[float] = pa.Field(alias="MU jaar 5", nullable=True, coerce=True)
    mu_jaar_6: Series[float] = pa.Field(alias="MU jaar 6", nullable=True, coerce=True)
    mu_jaar_7: Series[float] = pa.Field(alias="MU jaar 7", nullable=True, coerce=True)
    mu_jaar_8: Series[float] = pa.Field(alias="MU jaar 8", nullable=True, coerce=True)
    mu_jaar_9: Series[float] = pa.Field(alias="MU jaar 9", nullable=True, coerce=True)
    mu_jaar_10: Series[float] = pa.Field(alias="MU jaar 10", nullable=True, coerce=True)
    mu_jaar_11: Series[float] = pa.Field(alias="MU jaar 11", nullable=True, coerce=True)
    mu_jaar_12: Series[float] = pa.Field(alias="MU jaar 12", nullable=True, coerce=True)
    mu_jaar_13: Series[float] = pa.Field(alias="MU jaar 13", nullable=True, coerce=True)
    mu_jaar_14: Series[float] = pa.Field(alias="MU jaar 14", nullable=True, coerce=True)
    mu_jaar_15: Series[float] = pa.Field(alias="MU jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewOnderhoudBogMogZogSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_onderhoud_bog_mog_zog (14 cols) — maintenance per BOG/MOG/ZOG VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_oh: Series[str] = pa.Field(alias="Bron Onderhoud (OH)", nullable=True)
    oh_per_m2_bvo_handboek_waarde: Series[float] = pa.Field(alias="OH per m² BVO handboek waarde", nullable=True, coerce=True)
    oh_per_m2_bvo_waarde: Series[float] = pa.Field(alias="OH per m² BVO waarde", nullable=True, coerce=True)
    achterstallig_onderhoud_per_m2_bvo: Series[float] = pa.Field(alias="Achterstallig onderhoud per m2 BVO", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewOnderhoudWoningenParkerenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_onderhoud_woningen_parkeren (19 cols) — maintenance per Woningen/Parkeren VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_oh: Series[str] = pa.Field(alias="Bron Onderhoud (OH)", nullable=True)
    oh_doorexploiteren_handboek: Series[float] = pa.Field(alias="OH doorexploiteren handboek", nullable=True, coerce=True)
    oh_uitponden_handboek: Series[float] = pa.Field(alias="OH uitponden handboek", nullable=True, coerce=True)
    oh_doorexploiteren: Series[float] = pa.Field(alias="OH doorexploiteren", nullable=True, coerce=True)
    oh_uitponden: Series[float] = pa.Field(alias="OH uitponden", nullable=True, coerce=True)
    bron_moh: Series[str] = pa.Field(alias="Bron mutatieonderhoud (MOH)", nullable=True)
    moh_doorexploiteren: Series[float] = pa.Field(alias="MOH doorexploiteren", nullable=True, coerce=True)
    moh_uitponden: Series[float] = pa.Field(alias="MOH uitponden", nullable=True, coerce=True)
    achterstallig_onderhoud: Series[float] = pa.Field(alias="Achterstallig onderhoud", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewOverigeExploitatielastenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_overige_exploitatielasten (27 cols) — other operating costs per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_vk: Series[str] = pa.Field(alias="Bron verkoopkosten (VK)", nullable=True)
    vk_handboek_bedrag: Series[float] = pa.Field(alias="VK handboek (bedrag)", nullable=True, coerce=True)
    vk_bedrag: Series[float] = pa.Field(alias="VK bedrag", nullable=True, coerce=True)
    vk_percentage: Series[float] = pa.Field(alias="VK percentage", nullable=True, coerce=True)
    bron_bk_woningen: Series[str] = pa.Field(alias="Bron beheerkosten (BK) (Woningen/Parkeren)", nullable=True)
    bk_waarde_handboek_woningen: Series[float] = pa.Field(alias="BK waarde handboek (Woningen/Parkeren)", nullable=True, coerce=True)
    bk_waarde_woningen: Series[float] = pa.Field(alias="BK waarde (Woningen/Parkeren)", nullable=True, coerce=True)
    bron_bk_bog: Series[str] = pa.Field(alias="Bron beheerkosten (BK) (% marktjaarhuur, BOG/MOG/ZOG)", nullable=True)
    bk_waarde_handboek_bog: Series[float] = pa.Field(alias="BK waarde handboek (% marktjaarhuur, BOG/MOG/ZOG)", nullable=True, coerce=True)
    bk_waarde_bog: Series[float] = pa.Field(alias="BK waarde (% marktjaarhuur, BOG/MOG/ZOG)", nullable=True, coerce=True)
    bron_bv: Series[str] = pa.Field(alias="Bron belastingen en verzekeringen (BV)", nullable=True)
    bv_handboek_percentage: Series[float] = pa.Field(alias="BV handboek (percentage)", nullable=True, coerce=True)
    bv_bedrag: Series[float] = pa.Field(alias="BV bedrag", nullable=True, coerce=True)
    bv_percentage: Series[float] = pa.Field(alias="BV percentage", nullable=True, coerce=True)
    bron_ozb: Series[str] = pa.Field(alias="Bron OZB", nullable=True)
    ozb_tarief_handboek: Series[float] = pa.Field(alias="OZB-tarief handboek", nullable=True, coerce=True)
    ozb_tarief: Series[float] = pa.Field(alias="OZB-tarief", nullable=True, coerce=True)
    bron_hd: Series[str] = pa.Field(alias="Bron huurderving (HD)", nullable=True)
    hd_handboek_waarde: Series[float] = pa.Field(alias="HD handboek waarde", nullable=True, coerce=True)
    hd_waarde: Series[float] = pa.Field(alias="HD waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewOverigeKostenEnOpbrengsten(pa.DataFrameModel):
    """Schema for tms.parameters_overview_overige_kosten_en_opbrengsten (39 cols) — other costs & revenues per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_ovk: Series[str] = pa.Field(alias="Bron overige kosten (OVK)", nullable=True)
    ovk_jaar_1: Series[float] = pa.Field(alias="OVK jaar 1", nullable=True, coerce=True)
    ovk_jaar_2: Series[float] = pa.Field(alias="OVK jaar 2", nullable=True, coerce=True)
    ovk_jaar_3: Series[float] = pa.Field(alias="OVK jaar 3", nullable=True, coerce=True)
    ovk_jaar_4: Series[float] = pa.Field(alias="OVK jaar 4", nullable=True, coerce=True)
    ovk_jaar_5: Series[float] = pa.Field(alias="OVK jaar 5", nullable=True, coerce=True)
    ovk_jaar_6: Series[float] = pa.Field(alias="OVK jaar 6", nullable=True, coerce=True)
    ovk_jaar_7: Series[float] = pa.Field(alias="OVK jaar 7", nullable=True, coerce=True)
    ovk_jaar_8: Series[float] = pa.Field(alias="OVK jaar 8", nullable=True, coerce=True)
    ovk_jaar_9: Series[float] = pa.Field(alias="OVK jaar 9", nullable=True, coerce=True)
    ovk_jaar_10: Series[float] = pa.Field(alias="OVK jaar 10", nullable=True, coerce=True)
    ovk_jaar_11: Series[float] = pa.Field(alias="OVK jaar 11", nullable=True, coerce=True)
    ovk_jaar_12: Series[float] = pa.Field(alias="OVK jaar 12", nullable=True, coerce=True)
    ovk_jaar_13: Series[float] = pa.Field(alias="OVK jaar 13", nullable=True, coerce=True)
    ovk_jaar_14: Series[float] = pa.Field(alias="OVK jaar 14", nullable=True, coerce=True)
    ovk_jaar_15: Series[float] = pa.Field(alias="OVK jaar 15", nullable=True, coerce=True)
    bron_ovo: Series[str] = pa.Field(alias="Bron overige opbrengsten (OVO)", nullable=True)
    ovo_jaar_1: Series[float] = pa.Field(alias="OVO jaar 1", nullable=True, coerce=True)
    ovo_jaar_2: Series[float] = pa.Field(alias="OVO jaar 2", nullable=True, coerce=True)
    ovo_jaar_3: Series[float] = pa.Field(alias="OVO jaar 3", nullable=True, coerce=True)
    ovo_jaar_4: Series[float] = pa.Field(alias="OVO jaar 4", nullable=True, coerce=True)
    ovo_jaar_5: Series[float] = pa.Field(alias="OVO jaar 5", nullable=True, coerce=True)
    ovo_jaar_6: Series[float] = pa.Field(alias="OVO jaar 6", nullable=True, coerce=True)
    ovo_jaar_7: Series[float] = pa.Field(alias="OVO jaar 7", nullable=True, coerce=True)
    ovo_jaar_8: Series[float] = pa.Field(alias="OVO jaar 8", nullable=True, coerce=True)
    ovo_jaar_9: Series[float] = pa.Field(alias="OVO jaar 9", nullable=True, coerce=True)
    ovo_jaar_10: Series[float] = pa.Field(alias="OVO jaar 10", nullable=True, coerce=True)
    ovo_jaar_11: Series[float] = pa.Field(alias="OVO jaar 11", nullable=True, coerce=True)
    ovo_jaar_12: Series[float] = pa.Field(alias="OVO jaar 12", nullable=True, coerce=True)
    ovo_jaar_13: Series[float] = pa.Field(alias="OVO jaar 13", nullable=True, coerce=True)
    ovo_jaar_14: Series[float] = pa.Field(alias="OVO jaar 14", nullable=True, coerce=True)
    ovo_jaar_15: Series[float] = pa.Field(alias="OVO jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewScenarioSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_scenario (9 cols) — valuation scenario per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_sc: Series[str] = pa.Field(alias="Bron scenario (SC)", nullable=True)
    sc_waarde: Series[str] = pa.Field(alias="SC waarde", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewSchemVrijheidsgradenBogSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_schem_vrijheidsgraden_bog (45 cols) — BOG/MOG/ZOG freedom-degree parameters per VHE."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr", nullable=True)
    adres: Series[str] = pa.Field(alias="Adres", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bron_mkt: Series[str] = pa.Field(alias="Bron mutatiekosten technisch (MKT)", nullable=True)
    mkt_handboek_waarde: Series[float] = pa.Field(alias="MKT handboek waarde", nullable=True, coerce=True)
    mkt_waarde: Series[float] = pa.Field(alias="MKT waarde", nullable=True, coerce=True)
    bron_mkm: Series[str] = pa.Field(alias="Bron mutatiekosten marketing (MKM)", nullable=True)
    mkm_handboek_waarde: Series[float] = pa.Field(alias="MKM handboek waarde", nullable=True, coerce=True)
    mkm_waarde: Series[float] = pa.Field(alias="MKM waarde", nullable=True, coerce=True)
    bron_hvp: Series[str] = pa.Field(alias="Bron huurvrije periode na mutatie (HVP)", nullable=True)
    hvp_waarde_maanden: Series[float] = pa.Field(alias="HVP waarde (maanden)", nullable=True, coerce=True)
    bron_hvm: Series[str] = pa.Field(alias="Bron huurverhogingsmoment (HVM)", nullable=True)
    hvm_waarde_maand: Series[float] = pa.Field(alias="HVM waarde (maand)", nullable=True, coerce=True)
    bron_mul: Series[str] = pa.Field(alias="Bron mutatieleegstand (MUL)", nullable=True)
    mul_waarde_maanden: Series[float] = pa.Field(alias="MUL waarde (maanden)", nullable=True, coerce=True)
    bron_nhc: Series[str] = pa.Field(alias="Bron nieuw huurcontract (NHC)", nullable=True)
    nhc_waarde: Series[str] = pa.Field(alias="NHC waarde", nullable=True)
    bron_lbh: Series[str] = pa.Field(alias="Bron looptijd bij herziening (LBH)", nullable=True)
    lbh_waarde: Series[float] = pa.Field(alias="LBH waarde", nullable=True, coerce=True)
    bron_hs: Series[str] = pa.Field(alias="Bron reguliere huurstijging boven inflatie (HS)", nullable=True)
    hs_jaar_0_handboek: Series[float] = pa.Field(alias="HS jaar 0 handboek", nullable=True, coerce=True)
    hs_jaar_0: Series[float] = pa.Field(alias="HS jaar 0", nullable=True, coerce=True)
    bron_inc: Series[str] = pa.Field(alias="Bron incentives (INC)", nullable=True)
    inc_jaar_1: Series[float] = pa.Field(alias="INC jaar 1", nullable=True, coerce=True)
    inc_jaar_2: Series[float] = pa.Field(alias="INC jaar 2", nullable=True, coerce=True)
    inc_jaar_3: Series[float] = pa.Field(alias="INC jaar 3", nullable=True, coerce=True)
    inc_jaar_4: Series[float] = pa.Field(alias="INC jaar 4", nullable=True, coerce=True)
    inc_jaar_5: Series[float] = pa.Field(alias="INC jaar 5", nullable=True, coerce=True)
    inc_jaar_6: Series[float] = pa.Field(alias="INC jaar 6", nullable=True, coerce=True)
    inc_jaar_7: Series[float] = pa.Field(alias="INC jaar 7", nullable=True, coerce=True)
    inc_jaar_8: Series[float] = pa.Field(alias="INC jaar 8", nullable=True, coerce=True)
    inc_jaar_9: Series[float] = pa.Field(alias="INC jaar 9", nullable=True, coerce=True)
    inc_jaar_10: Series[float] = pa.Field(alias="INC jaar 10", nullable=True, coerce=True)
    inc_jaar_11: Series[float] = pa.Field(alias="INC jaar 11", nullable=True, coerce=True)
    inc_jaar_12: Series[float] = pa.Field(alias="INC jaar 12", nullable=True, coerce=True)
    inc_jaar_13: Series[float] = pa.Field(alias="INC jaar 13", nullable=True, coerce=True)
    inc_jaar_14: Series[float] = pa.Field(alias="INC jaar 14", nullable=True, coerce=True)
    inc_jaar_15: Series[float] = pa.Field(alias="INC jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewSplitsingskostenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_splitsingskosten (12 cols) — apartment split costs per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_js: Series[str] = pa.Field(alias="Bron juridische splitsingskosten (JS)", nullable=True)
    js_handboek_waarde: Series[float] = pa.Field(alias="JS handboek waarde", nullable=True, coerce=True)
    js_waarde: Series[float] = pa.Field(alias="JS waarde", nullable=True, coerce=True)
    bron_ts: Series[str] = pa.Field(alias="Bron technische splitsingskosten (TS)", nullable=True)
    ts_waarde: Series[float] = pa.Field(alias="TS waarde", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ParametersOverviewVerkoopbeperkingWoningenSchema(pa.DataFrameModel):
    """Schema for tms.parameters_overview_verkoopbeperking_woningen (23 cols) — sales restriction per complex."""

    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    model: Series[str] = pa.Field(alias="Model", nullable=True)
    bron_vm: Series[str] = pa.Field(alias="Bron gedeelte niet verkopen bij mutatie (VM)", nullable=True)
    vm_jaar_1: Series[float] = pa.Field(alias="VM jaar 1", nullable=True, coerce=True)
    vm_jaar_2: Series[float] = pa.Field(alias="VM jaar 2", nullable=True, coerce=True)
    vm_jaar_3: Series[float] = pa.Field(alias="VM jaar 3", nullable=True, coerce=True)
    vm_jaar_4: Series[float] = pa.Field(alias="VM jaar 4", nullable=True, coerce=True)
    vm_jaar_5: Series[float] = pa.Field(alias="VM jaar 5", nullable=True, coerce=True)
    vm_jaar_6: Series[float] = pa.Field(alias="VM jaar 6", nullable=True, coerce=True)
    vm_jaar_7: Series[float] = pa.Field(alias="VM jaar 7", nullable=True, coerce=True)
    vm_jaar_8: Series[float] = pa.Field(alias="VM jaar 8", nullable=True, coerce=True)
    vm_jaar_9: Series[float] = pa.Field(alias="VM jaar 9", nullable=True, coerce=True)
    vm_jaar_10: Series[float] = pa.Field(alias="VM jaar 10", nullable=True, coerce=True)
    vm_jaar_11: Series[float] = pa.Field(alias="VM jaar 11", nullable=True, coerce=True)
    vm_jaar_12: Series[float] = pa.Field(alias="VM jaar 12", nullable=True, coerce=True)
    vm_jaar_13: Series[float] = pa.Field(alias="VM jaar 13", nullable=True, coerce=True)
    vm_jaar_14: Series[float] = pa.Field(alias="VM jaar 14", nullable=True, coerce=True)
    vm_jaar_15: Series[float] = pa.Field(alias="VM jaar 15", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True
