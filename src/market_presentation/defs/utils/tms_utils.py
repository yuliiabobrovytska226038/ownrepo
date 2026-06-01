"""Shared utility functions for column normalisation.

This module provides:

- **Column normalisation**: ``normalize_column_names()`` converts camelCase and
  Dutch column headers to lowercase English snake_case, then enforces canonical
  types (Int64, float64, str) via ``enforce_column_types()``.
- **Dutch→English mapping**: ``_DUTCH_TO_ENGLISH`` is the single source of truth
  for renaming Dutch TMS/CBS column names to their English equivalents.

Report-specific parsing and raw-file persistence live in
``market_presentation.defs.utils.tms_report_utils``.
"""

import re

import pandas as pd

_SNAKE_RE1 = re.compile(r"([a-z0-9])([A-Z])")
_SNAKE_RE2 = re.compile(r"[^a-zA-Z0-9]")
_SNAKE_RE3 = re.compile(r"_+")

# Dutch → English column name mapping applied *after* camelCase → snake_case conversion.
# This is the single central point for all Dutch→English column renaming in the pipeline.
_DUTCH_TO_ENGLISH: dict[str, str] = {
    # TMS market value parameters (Excel export)
    "complexcode": "complex_code",
    "complexnaam": "complex_name",
    "waarderingsmodel": "valuation_model",
    "deelportefeuille": "sub_portfolio",
    "disconteringsvoet_dv": "discount_rate_type",
    "dv_doorexploiteren": "discount_rate_continue",
    "dv_uitponden": "discount_rate_sell",
    "exit_yield_ey": "exit_yield_type",
    "ey_doorexploiteren": "exit_yield_continue",
    "ey_uitponden": "exit_yield_sell",
    "scenario_sc": "scenario_type",
    "sc_waarde": "scenario_value",
    "onderhoud_oh": "maintenance_type",
    "oh_doorexploiteren": "maintenance_continue",
    "oh_uitponden": "maintenance_sell",
    "mutatieonderhoud_moh": "mutation_maintenance_type",
    "moh_doorexploiteren": "mutation_maintenance_continue",
    "moh_uitponden": "mutation_maintenance_sell",
    "mutatiegraad_doorexploiteren_md": "mutation_rate_continue_type",
    "md_waarde": "mutation_rate_continue_value",
    "technische_splitsingskosten_ts": "technical_splitting_cost_type",
    "ts_waarde": "technical_splitting_cost_value",
    # Historical data (Excel)
    "waarderingstype": "valuation_type",
    "disconteringsvoet": "discount_rate",
    "mediaan": "median",
    # Valuation overview flat
    "waarderingscomplex": "complex_code",
    "netto_marktwaarde": "net_market_value",
    "aantal_vhes": "n_vhe",
    # CBS COROP regions
    "corop_plusgebieden_code": "corop_code",
    "corop_plusgebieden_naam": "corop_naam",
    "provincies_code": "provincie_code",
    "provincies_naam": "provincie_naam",
}

# Column type enforcement — applied *after* normalize_column_names().
# "int" columns are cast with pd.to_numeric(errors="coerce").astype("Int64") (nullable).
# "float" columns are cast with pd.to_numeric(errors="coerce") (float64).
# "str" columns are cast with .astype(str) and '' for NA.
_COLUMN_TYPES: dict[str, str] = {
    # CBS house price index
    "jaar": "int",
    "kwartaal": "int",
    "prijsindex_verkoopprijzen_1": "float",
    "ontwikkeling_tovvoorgaande_periode_2": "float",
    "ontwikkeling_toveen_jaar_eerder_3": "float",
    "verkochte_woningen_4": "int",
    "gemiddelde_verkoopprijs_7": "float",
    "mutatie_yoy": "float",
    # Historical data
    "pct_mutatie": "float",
    "index_mutatie": "float",
    "indexcijfer": "float",
    "discount_rate": "float",
    "percentage": "float",
    "median": "float",
    # dVi housing
    "aantal": "float",
    "marktwaarde": "float",
    "woz_waarde": "float",
    "nettohuur": "float",
    # TMS difference analysis
    "valuation_round_id": "int",
    "report_id": "int",
    "value": "float",
    "delta_with_previous_step": "float",
    "relative_delta": "float",
    # TMS energy performance
    "policy_value": "float",
    "quality_energy_label": "float",
    # Postal codes / geography — ensure string for consistent joins
    "postcode": "str",
    "gemeentecode": "str",
    "vhe_nr": "str",
    "complex_code": "str",
    "complex_internal_id": "str",
    # TMS ratio outputs
    "ratio_discount_rate": "float",
    "ratio_exit_yield": "float",
    "net_market_value_per_unit": "float",
    "net_market_value_per_m2": "float",
    "vacant_value_ratio": "float",
    "woz_ratio": "float",
    "wault": "float",
    "bar_contract_rent": "float",
    "bar_market_rent": "float",
    "nar": "float",
}


def enforce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their canonical types per ``_COLUMN_TYPES``.

    Applied after ``normalize_column_names()`` so that DuckDB source tables
    are written with correct types and downstream dbt models need no casts.

    Args:
        df: DataFrame whose columns should be type-enforced.

    Returns:
        The same DataFrame with type-cast columns (modified in-place).
    """
    for col, dtype in _COLUMN_TYPES.items():
        if col not in df.columns:
            continue
        if dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "str":
            df[col] = df[col].astype(str).replace({"nan": None, "None": None, "<NA>": None, "": None})
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase English snake_case.

    Three-step process:
      1. camelCase → snake_case (``complexInternalId`` → ``complex_internal_id``)
      2. Dutch names → English via ``_DUTCH_TO_ENGLISH`` mapping
      3. Enforce canonical column types via ``enforce_column_types()``

    Args:
        df: DataFrame whose columns should be normalised.

    Returns:
        The same DataFrame with renamed columns and enforced types (modified in-place).
    """

    def _to_snake(name: str) -> str:
        s = str(name)
        s = s.replace("%", "pct").replace("'", "")
        s = _SNAKE_RE1.sub(r"\1_\2", s)
        s = _SNAKE_RE2.sub("_", s)
        s = _SNAKE_RE3.sub("_", s)
        return s.strip("_").lower()

    df.columns = [_DUTCH_TO_ENGLISH.get(snake, snake) for snake in (_to_snake(c) for c in df.columns)]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return enforce_column_types(df)
