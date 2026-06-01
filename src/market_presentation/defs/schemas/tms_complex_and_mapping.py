"""Pandera schemas for complex kenmerken, complex references, market_value_basis, rental_unit_mapping, and tms_verschillenanalyse tables."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


class ComplexKenmerkenSchema(pa.DataFrameModel):
    """Schema for tms.complex_kenmerken (24 cols) — kwalitatieve kenmerken per complex.

    Most qualitative fields are INTEGER in DuckDB because they hold NULL for most companies.
    """

    complexcode: Series[str] = pa.Field(alias="Complexcode")
    waarderingsmodel: Series[str] = pa.Field(alias="Waarderingsmodel", nullable=True)
    deelportefeuille: Series[str] = pa.Field(alias="Deelportefeuille", nullable=True)
    naam_van_de_taxateur: Series[str] = pa.Field(alias="Naam van de taxateur", nullable=True, coerce=True)
    doel_taxatie: Series[str] = pa.Field(alias="Doel taxatie", nullable=True, coerce=True)
    type_taxatie: Series[str] = pa.Field(alias="Type taxatie", nullable=True, coerce=True)
    gebruikssituatie: Series[str] = pa.Field(alias="Gebruikssituatie", nullable=True, coerce=True)
    eigendomssituatie: Series[str] = pa.Field(alias="Eigendomssituatie", nullable=True, coerce=True)
    objectomschrijving: Series[str] = pa.Field(alias="Objectomschrijving", nullable=True, coerce=True)
    onderhoudsstaat: Series[str] = pa.Field(alias="Onderhoudsstaat", nullable=True, coerce=True)
    uitgangspunten: Series[str] = pa.Field(alias="Uitgangspunten", nullable=True, coerce=True)
    locatieomschrijving: Series[str] = pa.Field(alias="Locatieomschrijving", nullable=True, coerce=True)
    locatiebeoordeling: Series[str] = pa.Field(alias="Locatiebeoordeling", nullable=True, coerce=True)
    bereikbaarheid: Series[str] = pa.Field(alias="Bereikbaarheid", nullable=True, coerce=True)
    verhuurbaarheid: Series[str] = pa.Field(alias="Verhuurbaarheid", nullable=True, coerce=True)
    verkoopbaarheid: Series[str] = pa.Field(alias="Verkoopbaarheid", nullable=True, coerce=True)
    bezichtigingsdatum: Series[str] = pa.Field(alias="Bezichtigingsdatum", nullable=True, coerce=True)
    sterkte_swot: Series[str] = pa.Field(alias="Sterkte (SWOT)", nullable=True, coerce=True)
    zwakte_swot: Series[str] = pa.Field(alias="Zwakte (SWOT)", nullable=True, coerce=True)
    kans_swot: Series[str] = pa.Field(alias="Kans (SWOT)", nullable=True, coerce=True)
    bedreiging_swot: Series[str] = pa.Field(alias="Bedreiging (SWOT)", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class ComplexReferencesSchema(pa.DataFrameModel):
    """Schema for tms.complex_references (18 cols) — referentie transacties per complex.

    Most fields are INTEGER in DuckDB because they hold NULL (reference data rarely provided).
    """

    complexcode: Series[str] = pa.Field(alias="Complexcode")
    waarderingsmodel: Series[str] = pa.Field(alias="Waarderingsmodel", nullable=True)
    deelportefeuille: Series[str] = pa.Field(alias="Deelportefeuille", nullable=True)
    referentietype: Series[str] = pa.Field(alias="Referentietype", nullable=True)
    volgnummer: Series[float] = pa.Field(alias="Volgnummer", nullable=True, coerce=True)
    adresaanduiding: Series[str] = pa.Field(alias="Adresaanduiding", nullable=True, coerce=True)
    objecttype: Series[str] = pa.Field(alias="Objecttype", nullable=True, coerce=True)
    oppervlakte: Series[float] = pa.Field(alias="Oppervlakte", nullable=True, coerce=True)
    bouwjaar: Series[float] = pa.Field(alias="Bouwjaar", nullable=True, coerce=True)
    transactieprijs: Series[float] = pa.Field(alias="Transactieprijs", nullable=True, coerce=True)
    bar: Series[float] = pa.Field(alias="BAR", nullable=True, coerce=True)
    transactiedatum: Series[str] = pa.Field(alias="Transactiedatum", nullable=True, coerce=True)
    kwalificatie_tov_te_taxeren_object: Series[str] = pa.Field(alias="Kwalificatie t.o.v. te taxeren object", nullable=True, coerce=True)
    toelichting: Series[str] = pa.Field(alias="Toelichting", nullable=True, coerce=True)
    bron: Series[str] = pa.Field(alias="Bron", nullable=True, coerce=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class MarketValueBasisSchema(pa.DataFrameModel):
    """Schema for tms.market_value_basis (14 cols) — BASIC/FULL market values per rentalUnitInternalId.

    Note: does NOT have VHE-nr; join via tms.rental_unit_mapping to get VHE-nr.
    """

    rental_unit_internal_id: Series[int] = pa.Field(alias="rentalUnitInternalId", coerce=True)
    complex_internal_id: Series[float] = pa.Field(alias="complexInternalId", nullable=True, coerce=True)
    marktwaarde: Series[float] = pa.Field(alias="Marktwaarde", nullable=True, coerce=True)
    marktwaarde_doorexploiteren: Series[float] = pa.Field(alias="Marktwaarde doorexploiteren", nullable=True, coerce=True)
    marktwaarde_uitponden: Series[float] = pa.Field(alias="Marktwaarde uitponden", nullable=True, coerce=True)
    scenario: Series[str] = pa.Field(alias="Scenario", nullable=True)
    marktwaarde_basis: Series[float] = pa.Field(alias="Marktwaarde basis", nullable=True, coerce=True)
    marktwaarde_basis_doorexploiteren: Series[float] = pa.Field(alias="Marktwaarde basis doorexploiteren", nullable=True, coerce=True)
    marktwaarde_basis_uitponden: Series[float] = pa.Field(alias="Marktwaarde basis uitponden", nullable=True, coerce=True)
    scenario_basis: Series[str] = pa.Field(alias="Scenario basis", nullable=True)
    issue_date: Series[str] = pa.Field(alias="ISSUE_DATE", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class RentalUnitMappingSchema(pa.DataFrameModel):
    """Schema for tms.rental_unit_mapping (7 cols) — maps rentalUnitInternalId → VHE-nr via VGR dataset API."""

    rental_unit_internal_id: Series[int] = pa.Field(alias="rentalUnitInternalId", coerce=True)
    vhe_nr: Series[str] = pa.Field(alias="VHE-nr")
    complex_internal_id: Series[float] = pa.Field(alias="complexInternalId", nullable=True, coerce=True)
    complexcode: Series[str] = pa.Field(alias="Complexcode", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True


class TmsVerschillenanalyseSchema(pa.DataFrameModel):
    """Schema for tms.tms_verschillenanalyse_marketvalue and tms.tms_verschillenanalyse_policyvalue (68 cols).

    STRUCT array columns (numericalExplainers) are stored as JSON text for stable DuckDB typing.
    """

    step_name: Series[str] = pa.Field(alias="stepName", nullable=True)
    group_name: Series[str] = pa.Field(alias="groupName", nullable=True)
    textual_explainer: Series[str] = pa.Field(alias="textualExplainer", nullable=True)

    # DAEB segments
    daeb_residential_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.RESIDENTIAL.value", nullable=True, coerce=True)
    daeb_residential_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.RESIDENTIAL.deltaWithPreviousStep", nullable=True, coerce=True)
    daeb_residential_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.RESIDENTIAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    daeb_residential_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.DAEB.RESIDENTIAL.numericalExplainers", nullable=True, coerce=True)
    daeb_parking_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.PARKING.value", nullable=True, coerce=True)
    daeb_parking_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.PARKING.deltaWithPreviousStep", nullable=True, coerce=True)
    daeb_parking_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.PARKING.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    daeb_parking_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.DAEB.PARKING.numericalExplainers", nullable=True, coerce=True)
    daeb_bog_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.BOG.value", nullable=True, coerce=True)
    daeb_bog_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.BOG.deltaWithPreviousStep", nullable=True, coerce=True)
    daeb_bog_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.BOG.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    daeb_bog_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.DAEB.BOG.numericalExplainers", nullable=True, coerce=True)
    daeb_approx_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.APPROX.value", nullable=True, coerce=True)
    daeb_approx_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.APPROX.deltaWithPreviousStep", nullable=True, coerce=True)
    daeb_approx_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.APPROX.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    daeb_approx_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.DAEB.APPROX.numericalExplainers", nullable=True, coerce=True)
    daeb_total_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.TOTAL.value", nullable=True, coerce=True)
    daeb_total_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.TOTAL.deltaWithPreviousStep", nullable=True, coerce=True)
    daeb_total_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.DAEB.TOTAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    daeb_total_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.DAEB.TOTAL.numericalExplainers", nullable=True, coerce=True)

    # NOT_DAEB segments
    not_daeb_residential_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.RESIDENTIAL.value", nullable=True, coerce=True)
    not_daeb_residential_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.RESIDENTIAL.deltaWithPreviousStep", nullable=True, coerce=True)
    not_daeb_residential_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.RESIDENTIAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    not_daeb_residential_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.RESIDENTIAL.numericalExplainers", nullable=True, coerce=True)
    not_daeb_parking_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.PARKING.value", nullable=True, coerce=True)
    not_daeb_parking_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.PARKING.deltaWithPreviousStep", nullable=True, coerce=True)
    not_daeb_parking_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.PARKING.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    not_daeb_parking_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.PARKING.numericalExplainers", nullable=True, coerce=True)
    not_daeb_bog_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.BOG.value", nullable=True, coerce=True)
    not_daeb_bog_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.BOG.deltaWithPreviousStep", nullable=True, coerce=True)
    not_daeb_bog_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.BOG.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    not_daeb_bog_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.BOG.numericalExplainers", nullable=True, coerce=True)
    not_daeb_approx_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.APPROX.value", nullable=True, coerce=True)
    not_daeb_approx_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.APPROX.deltaWithPreviousStep", nullable=True, coerce=True)
    not_daeb_approx_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.APPROX.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    not_daeb_approx_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.APPROX.numericalExplainers", nullable=True, coerce=True)
    not_daeb_total_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.TOTAL.value", nullable=True, coerce=True)
    not_daeb_total_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.TOTAL.deltaWithPreviousStep", nullable=True, coerce=True)
    not_daeb_total_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.TOTAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    not_daeb_total_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.NOT_DAEB.TOTAL.numericalExplainers", nullable=True, coerce=True)

    # TOTAL segments
    total_residential_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.RESIDENTIAL.value", nullable=True, coerce=True)
    total_residential_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.RESIDENTIAL.deltaWithPreviousStep", nullable=True, coerce=True)
    total_residential_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.RESIDENTIAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    total_residential_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.RESIDENTIAL.numericalExplainers", nullable=True, coerce=True)
    total_parking_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.PARKING.value", nullable=True, coerce=True)
    total_parking_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.PARKING.deltaWithPreviousStep", nullable=True, coerce=True)
    total_parking_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.PARKING.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    total_parking_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.PARKING.numericalExplainers", nullable=True, coerce=True)
    total_bog_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.BOG.value", nullable=True, coerce=True)
    total_bog_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.BOG.deltaWithPreviousStep", nullable=True, coerce=True)
    total_bog_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.BOG.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    total_bog_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.BOG.numericalExplainers", nullable=True, coerce=True)
    total_approx_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.APPROX.value", nullable=True, coerce=True)
    total_approx_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.APPROX.deltaWithPreviousStep", nullable=True, coerce=True)
    total_approx_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.APPROX.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    total_approx_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.APPROX.numericalExplainers", nullable=True, coerce=True)
    total_total_value: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.TOTAL.value", nullable=True, coerce=True)
    total_total_delta: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.TOTAL.deltaWithPreviousStep", nullable=True, coerce=True)
    total_total_relative: Series[float] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.TOTAL.relativeDifferenceWithFirstStep", nullable=True, coerce=True)
    total_total_explainers: Series[str] = pa.Field(alias="valuePerClassificationAndModel.TOTAL.TOTAL.numericalExplainers", nullable=True, coerce=True)

    valuation_type: Series[str] = pa.Field(alias="Valuation_type", nullable=True)
    werkmaatschappij: Series[str] = pa.Field(alias="Werkmaatschappij", nullable=True)

    corporatie: Series[str] = pa.Field(alias="Corporatie")
    jaar: Series[int] = pa.Field(alias="Jaar", coerce=True)
    peildatum: Series[pd.Timestamp] = pa.Field(alias="Peildatum")

    class Config:
        strict = False
        coerce = True
