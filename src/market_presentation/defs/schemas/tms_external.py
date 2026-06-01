"""Pandera schemas for external and APIs tables."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series


class PostalCodeMappingSchema(pa.DataFrameModel):
    """Schema for external.postal_code_mapping — DLL postcode → region mapping."""

    postcode: Series[str] = pa.Field(alias="postcode")
    corop_gebied: Series[str] = pa.Field(alias="corop_gebied", nullable=True)
    regio: Series[str] = pa.Field(alias="regio", nullable=True)
    gemeentecode: Series[str] = pa.Field(alias="gemeentecode", nullable=True)
    gemeente: Series[str] = pa.Field(alias="gemeente", nullable=True)
    categorie: Series[str] = pa.Field(alias="categorie", nullable=True)

    class Config:
        strict = False
        coerce = True


class CbsCoropRegionsSchema(pa.DataFrameModel):
    """Schema for external.cbs_corop_regions — CBS municipality → COROP region mapping."""

    gemeentecode: Series[str] = pa.Field(alias="gemeentecode")
    corop_code: Series[str] = pa.Field(alias="corop_code", nullable=True)
    corop_naam: Series[str] = pa.Field(alias="corop_naam", nullable=True)
    provincie_code: Series[str] = pa.Field(alias="provincie_code", nullable=True)
    provincie_naam: Series[str] = pa.Field(alias="provincie_naam", nullable=True)

    class Config:
        strict = False
        coerce = True


class CompanyValuationDataSchema(pa.DataFrameModel):
    """Schema for tms.company_valuation_data — joined TMS company, round, and valuation data."""

    company_id: Series[str] = pa.Field(alias="company_id")
    valuation_round_id: Series[int] = pa.Field(alias="valuation_round_id", coerce=True)
    round_data_set_id: Series[int] = pa.Field(alias="round_data_set_id", nullable=True, coerce=True)
    issue_year: Series[int] = pa.Field(alias="issue_year", nullable=True, coerce=True)
    issue_quarter: Series[str] = pa.Field(alias="issue_quarter", nullable=True)
    issue_date: Series[str] = pa.Field(alias="issue_date", nullable=True)
    round_status: Series[str] = pa.Field(alias="round_status", nullable=True)
    round_is_free: Series[bool] = pa.Field(alias="round_is_free", nullable=True)
    valuation_id: Series[int] = pa.Field(alias="valuation_id", nullable=True, coerce=True)
    valuation_data_set_id: Series[int] = pa.Field(alias="valuation_data_set_id", nullable=True, coerce=True)
    valuation_name: Series[str] = pa.Field(alias="valuation_name", nullable=True)
    model_year: Series[int] = pa.Field(alias="model_year", nullable=True, coerce=True)
    archived_date_time: Series[str] = pa.Field(alias="archived_date_time", nullable=True)
    creation_date_time: Series[str] = pa.Field(alias="creation_date_time", nullable=True)
    data_set_update_date_time: Series[str] = pa.Field(alias="data_set_update_date_time", nullable=True)
    avm_issue_date: Series[str] = pa.Field(alias="avm_issue_date", nullable=True)
    is_old_data_set: Series[bool] = pa.Field(alias="is_old_data_set", nullable=True)
    has_changed_parameters: Series[bool] = pa.Field(alias="has_changed_parameters", nullable=True)
    is_new_calculation_rules: Series[bool] = pa.Field(alias="is_new_calculation_rules", nullable=True)
    sub_portfolio_id: Series[int] = pa.Field(alias="sub_portfolio_id", nullable=True, coerce=True)
    sub_portfolio_name: Series[str] = pa.Field(alias="sub_portfolio_name", nullable=True)
    sub_portfolio_valuation_level: Series[str] = pa.Field(alias="sub_portfolio_valuation_level", nullable=True)

    class Config:
        strict = False
        coerce = True


class CompanyListSchema(pa.DataFrameModel):
    """Schema for tms.company_list — all TMS companies."""

    company_id: Series[str] = pa.Field(alias="company_id")
    is_customer: Series[bool] = pa.Field(alias="is_customer", nullable=True)

    class Config:
        strict = False
        coerce = True
