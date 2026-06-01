"""Pandera schemas for vastgoedgegevens tables (VHE gegevens, waarderingscomplexen, sturingscomplexen, vov_verleden)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


class VheGegevensSchema(pa.DataFrameModel):
    """Schema for tms.vastgoedgegevens_vhe_gegevens — the main VHE property table (65 cols)."""

    vhe_nummer: Series[str] = pa.Field(alias="VHE-nummer")
    vastgoedcategorie: Series[str] = pa.Field(alias="Vastgoedcategorie", nullable=True)
    straatnaam: Series[str] = pa.Field(alias="Straatnaam", nullable=True)
    huisnummer: Series[float] = pa.Field(alias="Huisnummer", nullable=True, coerce=True)
    toevoeging: Series[str] = pa.Field(alias="Toevoeging", nullable=True)
    postcode: Series[str] = pa.Field(alias="Postcode", nullable=True)
    waarderingscomplex: Series[str] = pa.Field(alias="Waarderingscomplex", nullable=True)
    classificatie: Series[str] = pa.Field(alias="Classificatie", nullable=True)
    bouwjaar: Series[float] = pa.Field(alias="Bouwjaar", nullable=True, coerce=True)
    startdatum_contract: Series[pd.Timestamp] = pa.Field(alias="Startdatum contract", nullable=True)
    renovatiejaar: Series[float] = pa.Field(alias="Renovatiejaar", nullable=True, coerce=True)
    gebruiksoppervlakte: Series[float] = pa.Field(alias="Gebruiksoppervlakte", nullable=True, coerce=True)
    zelfstandig: Series[str] = pa.Field(alias="Zelfstandig", nullable=True)
    woz_waarde: Series[float] = pa.Field(alias="WOZ-waarde primo jaar-1", nullable=True, coerce=True)
    achterstallig_onderhoud: Series[float] = pa.Field(alias="Achterstallig onderhoud", nullable=True, coerce=True)
    leegstand: Series[str] = pa.Field(alias="Leegstand", nullable=True)
    netto_huur: Series[float] = pa.Field(alias="Netto huur", nullable=True, coerce=True)
    methode_maximale_huur: Series[str] = pa.Field(alias="Methode maximale huur", nullable=True)
    wws_punten: Series[float] = pa.Field(alias="WWS-punten", nullable=True, coerce=True)
    maximale_huur_excl_opslag: Series[float] = pa.Field(alias="Maximale huur excl. opslag", nullable=True, coerce=True)
    beschermd_bezit: Series[str] = pa.Field(alias="Beschermd bezit", nullable=True)
    nieuwbouwopslag: Series[str] = pa.Field(alias="Nieuwbouwopslag", nullable=True)
    segment_huurregime: Series[str] = pa.Field(alias="Segment huurregime", nullable=True)
    markthuur: Series[float] = pa.Field(alias="Markthuur", nullable=True, coerce=True)
    beklemmingshuur: Series[float] = pa.Field(alias="Beklemmingshuur", nullable=True, coerce=True)
    laatste_jaar_huurbeklemming: Series[float] = pa.Field(alias="Laatste jaar huurbeklemming", nullable=True, coerce=True)
    epv: Series[float] = pa.Field(alias="EPV", nullable=True, coerce=True)
    energieprestatie_ep2: Series[float] = pa.Field(alias="Energieprestatie (EP2)", nullable=True, coerce=True)
    handboektype: Series[str] = pa.Field(alias="Handboektype", nullable=True)
    erfpacht_waardecorrectie_doorexploiteren: Series[float] = pa.Field(alias="Erfpacht waardecorrectie doorexploiteren", nullable=True, coerce=True)
    erfpacht_waardecorrectie_uitponden: Series[float] = pa.Field(alias="Erfpacht waardecorrectie uitponden", nullable=True, coerce=True)
    bvo_m2: Series[float] = pa.Field(alias="BVO (m2)", nullable=True, coerce=True)
    vvo_m2: Series[float] = pa.Field(alias="VVO (m2)", nullable=True, coerce=True)
    aantal_contracten: Series[float] = pa.Field(alias="Aantal contracten", nullable=True, coerce=True)
    aantal_plekken: Series[float] = pa.Field(alias="Aantal plekken", nullable=True, coerce=True)
    einddatum_contract: Series[pd.Timestamp] = pa.Field(alias="Einddatum contract", nullable=True)
    nieuw_huurcontract: Series[str] = pa.Field(alias="Nieuw huurcontract", nullable=True)
    looptijd_bij_herziening: Series[float] = pa.Field(alias="Looptijd bij herziening", nullable=True, coerce=True)
    contracttype: Series[str] = pa.Field(alias="Contracttype", nullable=True)
    contracthuur_jaar_1: Series[float] = pa.Field(alias="Contracthuur jaar 1", nullable=True, coerce=True)
    contracthuur_jaar_2: Series[float] = pa.Field(alias="Contracthuur jaar 2", nullable=True, coerce=True)
    contracthuur_jaar_3: Series[float] = pa.Field(alias="Contracthuur jaar 3", nullable=True, coerce=True)
    contracthuur_jaar_4: Series[float] = pa.Field(alias="Contracthuur jaar 4", nullable=True, coerce=True)
    contracthuur_jaar_5: Series[float] = pa.Field(alias="Contracthuur jaar 5", nullable=True, coerce=True)
    contracthuur_jaar_6: Series[float] = pa.Field(alias="Contracthuur jaar 6", nullable=True, coerce=True)
    contracthuur_jaar_7: Series[float] = pa.Field(alias="Contracthuur jaar 7", nullable=True, coerce=True)
    contracthuur_jaar_8: Series[float] = pa.Field(alias="Contracthuur jaar 8", nullable=True, coerce=True)
    contracthuur_jaar_9: Series[float] = pa.Field(alias="Contracthuur jaar 9", nullable=True, coerce=True)
    contracthuur_jaar_10: Series[float] = pa.Field(alias="Contracthuur jaar 10", nullable=True, coerce=True)
    contracthuur_jaar_11: Series[float] = pa.Field(alias="Contracthuur jaar 11", nullable=True, coerce=True)
    contracthuur_jaar_12: Series[float] = pa.Field(alias="Contracthuur jaar 12", nullable=True, coerce=True)
    contracthuur_jaar_13: Series[float] = pa.Field(alias="Contracthuur jaar 13", nullable=True, coerce=True)
    contracthuur_jaar_14: Series[float] = pa.Field(alias="Contracthuur jaar 14", nullable=True, coerce=True)
    contracthuur_jaar_15: Series[float] = pa.Field(alias="Contracthuur jaar 15", nullable=True, coerce=True)
    straatnaam_opgezocht: Series[str] = pa.Field(alias="Straatnaam (opgezocht)", nullable=True)
    gemeente_opgezocht: Series[str] = pa.Field(alias="Gemeente (opgezocht)", nullable=True)
    plaatsnaam_opgezocht: Series[str] = pa.Field(alias="Plaatsnaam (opgezocht)", nullable=True)
    wijk_opgezocht: Series[str] = pa.Field(alias="Wijk (opgezocht)", nullable=True)
    buurt_opgezocht: Series[str] = pa.Field(alias="Buurt (opgezocht)", nullable=True)
    buurtcode_opgezocht: Series[str] = pa.Field(alias="Buurtcode (opgezocht)", nullable=True)
    bag_pand_id_opgezocht: Series[str] = pa.Field(alias="BAG pand id (opgezocht)", nullable=True)
    bag_verblijfsobject_id_opgezocht: Series[str] = pa.Field(alias="BAG verblijfsobject id (opgezocht)", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class WaarderingscomplexenSchema(pa.DataFrameModel):
    """Schema for tms.vastgoedgegevens_waarderingscomplexen (16 cols)."""

    complexcode: Series[str] = pa.Field(alias="Complexcode")
    complexnaam: Series[str] = pa.Field(alias="Complexnaam", nullable=True)
    werkmaatschappij: Series[str] = pa.Field(alias="Werkmaatschappij", nullable=True)
    waarderingsmodel: Series[str] = pa.Field(alias="Waarderingsmodel", nullable=True)
    exploitatiebeperking: Series[str] = pa.Field(alias="Exploitatiebeperking 7 jaar", nullable=True)
    complexstatus: Series[str] = pa.Field(alias="Complexstatus", nullable=True)
    gesplitst: Series[str] = pa.Field(alias="Gesplitst", nullable=True)
    flexwoning: Series[str] = pa.Field(alias="Flexwoning", nullable=True)
    aangebroken: Series[str] = pa.Field(alias="Aangebroken", nullable=True)
    mutatiekans: Series[float] = pa.Field(alias="Mutatiekans", nullable=True, coerce=True)
    maximaal_verkoopbaar: Series[float] = pa.Field(alias="Maximaal verkoopbaar (%)", nullable=True, coerce=True)
    deelportefeuille: Series[str] = pa.Field(alias="Deelportefeuille", nullable=True)
    referentiegroep: Series[str] = pa.Field(alias="Referentiegroep", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class SturingscomplexenSchema(pa.DataFrameModel):
    """Schema for tms.vastgoedgegevens_sturingscomplexen (usually empty; Corporatie inferred as INTEGER when table is empty)."""

    corporatie: Series[str] = pa.Field(alias="Corporatie", nullable=True, coerce=True)
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class VovVerledenSchema(pa.DataFrameModel):
    """Schema for tms.vastgoedgegevens_vov_verleden (usually empty; VoV (Verkoop onder Voorwaarden) history)."""

    corporatie: Series[str] = pa.Field(alias="Corporatie", nullable=True, coerce=True)
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True
