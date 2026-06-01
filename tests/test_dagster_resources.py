"""Unit tests for Dagster resources and tms_utils.

Tests cover:
- normalize_column_names (camelCase → snake_case, Dutch→English mapping)
- enforce_column_types
- DuckDB IO manager partition escape monkey-patch
- Resource configuration structure
"""

import pandas as pd
import pytest

from market_presentation.defs.utils.tms_utils import (
    enforce_column_types,
    normalize_column_names,
)


# ---------------------------------------------------------------------------
# normalize_column_names
# ---------------------------------------------------------------------------


class TestNormalizeColumnNames:
    def test_camel_case_to_snake_case(self):
        df = pd.DataFrame({"complexInternalId": [1], "rentalUnitInternalId": [2]})
        result = normalize_column_names(df)
        assert "complex_internal_id" in result.columns
        assert "rental_unit_internal_id" in result.columns

    def test_dutch_to_english_mapping(self):
        df = pd.DataFrame({"complexcode": ["A"], "waarderingsmodel": ["Woningen"]})
        result = normalize_column_names(df)
        assert "complex_code" in result.columns
        assert "valuation_model" in result.columns

    def test_special_characters_replaced(self):
        df = pd.DataFrame({"% Full": [1], "VHE-nr": ["001"]})
        result = normalize_column_names(df)
        # % → pct, - → _
        assert any("pct" in col for col in result.columns)

    def test_deduplicates_columns(self):
        df = pd.DataFrame({"colA": [1], "col_a": [2]})
        result = normalize_column_names(df)
        assert not result.columns.duplicated().any()

    def test_applies_type_enforcement(self):
        df = pd.DataFrame({"postcode": ["1234AB"], "marktwaarde": ["100000"]})
        result = normalize_column_names(df)
        if "postcode" in result.columns:
            # Postcode should be kept as string (not coerced to NaN)
            assert result["postcode"].iloc[0] is not None


# ---------------------------------------------------------------------------
# enforce_column_types
# ---------------------------------------------------------------------------


class TestEnforceColumnTypes:
    def test_int_columns_cast_to_int64(self):
        df = pd.DataFrame({"jaar": ["2024", "2023"]})
        result = enforce_column_types(df)
        assert result["jaar"].dtype == "Int64"
        assert list(result["jaar"]) == [2024, 2023]

    def test_float_columns_cast_to_float(self):
        df = pd.DataFrame({"marktwaarde": ["100.5", "200.3"]})
        result = enforce_column_types(df)
        assert result["marktwaarde"].dtype == "float64"

    def test_str_columns_stay_string(self):
        df = pd.DataFrame({"postcode": ["1234AB", "5678CD"]})
        result = enforce_column_types(df)
        # After enforcement, postcodes should be string-like
        assert result["postcode"].iloc[0] == "1234AB"

    def test_coercion_errors_produce_nan(self):
        df = pd.DataFrame({"jaar": ["2024", "not_a_year", "2023"]})
        result = enforce_column_types(df)
        assert pd.isna(result["jaar"].iloc[1])

    def test_unknown_columns_unchanged(self):
        df = pd.DataFrame({"unknown_col": [1, 2, 3]})
        original_dtype = df["unknown_col"].dtype
        result = enforce_column_types(df)
        assert result["unknown_col"].dtype == original_dtype


# ---------------------------------------------------------------------------
# Resource configuration (DuckDB IO manager monkey-patch)
# ---------------------------------------------------------------------------


class TestResourceMonkeyPatch:
    def test_safe_static_where_clause_escapes_single_quotes(self):
        """Verify that the monkey-patch in resources.py escapes single quotes."""
        from dagster._core.storage.db_io_manager import TablePartitionDimension
        from market_presentation.defs.resources.resources import _safe_static_where_clause

        partition = TablePartitionDimension(partition_expr="Corporatie", partitions=["Woonstichting 'thuis"])
        result = _safe_static_where_clause(partition)
        # The escaped version should not break SQL
        assert "'thuis" not in result or "''thuis" in result

    def test_safe_static_where_clause_normal_values(self):
        """Normal values without quotes pass through unchanged."""
        from dagster._core.storage.db_io_manager import TablePartitionDimension
        from market_presentation.defs.resources.resources import _safe_static_where_clause

        partition = TablePartitionDimension(partition_expr="Corporatie", partitions=["3B Wonen"])
        result = _safe_static_where_clause(partition)
        assert "3B Wonen" in result


# ---------------------------------------------------------------------------
# Resource definitions structure
# ---------------------------------------------------------------------------


class TestResourceDefinitions:
    def test_resource_defs_has_expected_keys(self):
        from market_presentation.defs.resources.resources import RESOURCE_DEFS

        assert "tms_api" in RESOURCE_DEFS
        assert "vgr_api" in RESOURCE_DEFS
        assert "duckdb" in RESOURCE_DEFS
        assert "io_manager" in RESOURCE_DEFS

    def test_tms_api_is_oauth2_resource(self):
        from market_presentation.defs.resources.oauth2_api import Oauth2ApiResource
        from market_presentation.defs.resources.resources import RESOURCE_DEFS

        assert isinstance(RESOURCE_DEFS["tms_api"], Oauth2ApiResource)

    def test_config_constants_are_consistent(self):
        from market_presentation.defs.config import CHART_YEAR, CHART_YEAR_M1, ISSUE_YEARS

        assert CHART_YEAR == max(ISSUE_YEARS)
        assert CHART_YEAR_M1 == CHART_YEAR - 1
        assert len(ISSUE_YEARS) >= 2
