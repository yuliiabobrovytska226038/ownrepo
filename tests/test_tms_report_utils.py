"""Unit tests for report aggregation logic in tms_report_utils.

Tests cover:
- get_round_metadata deduplication
- add_metadata_columns
- coerce_numeric_columns (identifier preservation, string prefix preservation)
- concat_frames
- validate_dataframe
- align_to_schema_columns
- load_or_fetch_bytes / load_or_fetch_json caching
- build_raw_output_path naming
- RoundMetadata properties
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import dagster as dg
import pandas as pd
import pytest

from market_presentation.defs.utils.tms_report_utils import (
    RoundMetadata,
    add_metadata_columns,
    align_to_schema_columns,
    build_raw_output_path,
    coerce_numeric_columns,
    concat_frames,
    get_round_metadata,
    load_or_fetch_bytes,
    load_or_fetch_json,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# RoundMetadata
# ---------------------------------------------------------------------------


class TestRoundMetadata:
    def test_issue_year_extracted_from_date(self):
        meta = RoundMetadata(company_id="3B Wonen", valuation_round_id=100, issue_date="2024-12-31", data_set_id=500)
        assert meta.issue_year == 2024

    def test_frozen_dataclass(self):
        meta = RoundMetadata(company_id="X", valuation_round_id=1, issue_date="2023-12-31")
        with pytest.raises(Exception):
            meta.company_id = "Y"  # type: ignore


# ---------------------------------------------------------------------------
# get_round_metadata
# ---------------------------------------------------------------------------


class TestGetRoundMetadata:
    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame()
        assert get_round_metadata("TestCo", df) == []

    def test_missing_column_returns_empty_list(self):
        df = pd.DataFrame({"other_column": [1, 2]})
        assert get_round_metadata("TestCo", df) == []

    def test_deduplicates_by_round_id_issue_date_dataset_id(self):
        df = pd.DataFrame(
            {
                "company_id": ["Co", "Co", "Co"],
                "valuation_round_id": [100, 100, 200],
                "issue_date": ["2024-12-31", "2024-12-31", "2024-12-31"],
                "issue_year": [2024, 2024, 2024],
                "round_data_set_id": [10, 10, 20],
            }
        )
        result = get_round_metadata("Co", df)
        assert len(result) == 2
        assert result[0].valuation_round_id == 100
        assert result[1].valuation_round_id == 200

    def test_filters_by_company_id(self):
        df = pd.DataFrame(
            {
                "company_id": ["CoA", "CoB"],
                "valuation_round_id": [1, 2],
                "issue_date": ["2024-12-31", "2024-12-31"],
                "issue_year": [2024, 2024],
                "round_data_set_id": [10, 20],
            }
        )
        result = get_round_metadata("CoA", df)
        assert len(result) == 1
        assert result[0].company_id == "CoA"

    def test_handles_nan_data_set_id(self):
        df = pd.DataFrame(
            {
                "company_id": ["Co"],
                "valuation_round_id": [100],
                "issue_date": ["2024-12-31"],
                "issue_year": [2024],
                "round_data_set_id": [float("nan")],
            }
        )
        result = get_round_metadata("Co", df)
        assert len(result) == 1
        assert result[0].data_set_id is None


# ---------------------------------------------------------------------------
# add_metadata_columns
# ---------------------------------------------------------------------------


class TestAddMetadataColumns:
    def test_adds_corporatie_jaar_peildatum(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        meta = RoundMetadata(company_id="3B Wonen", valuation_round_id=100, issue_date="2024-12-31")
        result = add_metadata_columns(df, meta)
        assert list(result["Corporatie"]) == ["3B Wonen", "3B Wonen", "3B Wonen"]
        assert list(result["Jaar"]) == [2024, 2024, 2024]
        assert result["Peildatum"].iloc[0] == pd.Timestamp("2024-12-31")

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"x": [1]})
        meta = RoundMetadata(company_id="Co", valuation_round_id=1, issue_date="2023-12-31")
        add_metadata_columns(df, meta)
        assert "Corporatie" not in df.columns


# ---------------------------------------------------------------------------
# coerce_numeric_columns
# ---------------------------------------------------------------------------


class TestCoerceNumericColumns:
    def test_converts_numeric_strings(self):
        df = pd.DataFrame({"value": ["1.5", "2.0", "3.7"]})
        result = coerce_numeric_columns(df)
        assert result["value"].dtype == "float64"

    def test_preserves_identifier_columns(self):
        df = pd.DataFrame({"VHE-nr": ["001", "002", "003"], "Postcode": ["1234AB", "5678CD", "9012EF"]})
        result = coerce_numeric_columns(df)
        assert result["VHE-nr"].dtype == object
        assert result["Postcode"].dtype == object

    def test_preserves_bron_prefixed_columns(self):
        df = pd.DataFrame({"Bron DV": ["1", "2", "3"]})
        result = coerce_numeric_columns(df)
        assert result["Bron DV"].dtype == object

    def test_keeps_column_as_string_when_less_than_50pct_numeric(self):
        df = pd.DataFrame({"mixed": ["hello", "world", "foo", "bar", "1"]})
        result = coerce_numeric_columns(df)
        assert result["mixed"].dtype == object

    def test_coerces_when_over_50pct_numeric(self):
        df = pd.DataFrame({"mostly_numeric": ["1", "2", "3", "not_a_number"]})
        result = coerce_numeric_columns(df)
        assert result["mostly_numeric"].dtype == "float64"

    def test_skips_non_object_dtypes(self):
        df = pd.DataFrame({"already_int": [1, 2, 3]})
        result = coerce_numeric_columns(df)
        assert result["already_int"].dtype in ("int64", "int32")


# ---------------------------------------------------------------------------
# concat_frames
# ---------------------------------------------------------------------------


class TestConcatFrames:
    def test_empty_list_returns_empty_dataframe(self):
        result = concat_frames([])
        assert result.empty

    def test_single_frame_returned_as_is(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = concat_frames([df])
        assert len(result) == 2

    def test_concatenates_multiple_frames(self):
        df1 = pd.DataFrame({"val": [1.0, 2.0]})
        df2 = pd.DataFrame({"val": [3.0, 4.0]})
        result = concat_frames([df1, df2])
        assert len(result) == 4

    def test_re_coerces_numeric_after_concat(self):
        # Simulates a real scenario: one frame has float, another has object from all-NaN
        df1 = pd.DataFrame({"value": [1.0, 2.0]})
        df2 = pd.DataFrame({"value": ["3.0", "4.0"]})
        result = concat_frames([df1, df2])
        assert result["value"].dtype == "float64"


# ---------------------------------------------------------------------------
# load_or_fetch_bytes / load_or_fetch_json
# ---------------------------------------------------------------------------


class TestLoadOrFetchBytes:
    def test_fetches_when_file_missing(self, tmp_path):
        path = tmp_path / "data.bin"
        content = b"hello bytes"
        result = load_or_fetch_bytes(path, lambda: content)
        assert result.value == content
        assert result.source == "download"
        assert path.read_bytes() == content

    def test_uses_cache_when_file_exists(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"cached")
        result = load_or_fetch_bytes(path, lambda: b"should not be called")
        assert result.value == b"cached"
        assert result.source == "cache"

    def test_force_download_ignores_cache(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"old")
        result = load_or_fetch_bytes(path, lambda: b"new", force_download=True)
        assert result.value == b"new"
        assert result.source == "download"

    def test_raises_failure_when_fetch_returns_none(self, tmp_path):
        path = tmp_path / "data.bin"
        with pytest.raises(dg.Failure):
            load_or_fetch_bytes(path, lambda: None)


class TestLoadOrFetchJson:
    def test_fetches_and_saves_json(self, tmp_path):
        path = tmp_path / "data.json"
        data = {"key": "value", "num": 42}
        result = load_or_fetch_json(path, lambda: data)
        assert result.value == data
        assert result.source == "download"
        assert json.loads(path.read_text()) == data

    def test_uses_cache_when_valid_json_exists(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"cached": true}')
        result = load_or_fetch_json(path, lambda: {"fresh": True})
        assert result.value == {"cached": True}
        assert result.source == "cache"

    def test_refetches_on_invalid_json_cache(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("not valid json {{")
        result = load_or_fetch_json(path, lambda: {"fixed": True})
        assert result.value == {"fixed": True}
        assert result.source == "download"

    def test_raises_failure_when_fetch_returns_none(self, tmp_path):
        path = tmp_path / "data.json"
        with pytest.raises(dg.Failure):
            load_or_fetch_json(path, lambda: None)


# ---------------------------------------------------------------------------
# build_raw_output_path
# ---------------------------------------------------------------------------


class TestBuildRawOutputPath:
    def test_standard_path_format(self):
        meta = RoundMetadata(company_id="3B Wonen", valuation_round_id=100, issue_date="2024-12-31")
        path = build_raw_output_path("market_value_parameters", meta)
        assert "Marktwaarde parameters 3B Wonen 2024.xlsx" in str(path)
        assert "marktwaardeparameters" in str(path)

    def test_json_extension_for_difference_analysis(self):
        meta = RoundMetadata(company_id="Acantus", valuation_round_id=200, issue_date="2023-12-31")
        path = build_raw_output_path("difference_analysis", meta)
        assert str(path).endswith(".json")
        assert "TMS Verschillenanalyse Acantus 2023" in str(path)

    def test_subsidiary_suffix(self):
        meta = RoundMetadata(company_id="Co", valuation_round_id=1, issue_date="2024-12-31")
        path = build_raw_output_path("market_value_parameters", meta, subsidiary="SubCo")
        assert "@ SubCo" in str(path)


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------


class TestValidateDataframe:
    def test_validates_with_schema(self):
        import pandera.pandas as pa

        class SimpleSchema(pa.DataFrameModel):
            name: str
            value: float

        df = pd.DataFrame({"name": ["a", "b"], "value": [1.0, 2.0]})
        result = validate_dataframe(df, SimpleSchema)
        assert len(result) == 2

    def test_empty_df_gets_schema_columns(self):
        import pandera.pandas as pa

        class TestSchema(pa.DataFrameModel):
            col_a: str
            col_b: str  # Use str to avoid type coercion issues on empty frames

        df = pd.DataFrame()
        result = validate_dataframe(df, TestSchema)
        assert "col_a" in result.columns
        assert "col_b" in result.columns


# ---------------------------------------------------------------------------
# align_to_schema_columns
# ---------------------------------------------------------------------------


class TestAlignToSchemaColumns:
    def test_adds_missing_columns(self):
        import pandera.pandas as pa

        class Schema(pa.DataFrameModel):
            existing: str
            missing_str: str
            missing_float: float

        df = pd.DataFrame({"existing": ["x", "y"]})
        result = align_to_schema_columns(df, Schema)
        assert "missing_str" in result.columns
        assert "missing_float" in result.columns
        assert list(result.columns[:3]) == ["existing", "missing_str", "missing_float"]

    def test_keep_extra_preserves_non_schema_columns(self):
        import pandera.pandas as pa

        class Schema(pa.DataFrameModel):
            a: str

        df = pd.DataFrame({"a": ["x"], "extra": [99]})
        result = align_to_schema_columns(df, Schema, keep_extra=True)
        assert "extra" in result.columns

    def test_keep_extra_false_drops_non_schema_columns(self):
        import pandera.pandas as pa

        class Schema(pa.DataFrameModel):
            a: str

        df = pd.DataFrame({"a": ["x"], "extra": [99]})
        result = align_to_schema_columns(df, Schema, keep_extra=False)
        assert "extra" not in result.columns


# ---------------------------------------------------------------------------
# build_raw_output_path
# ---------------------------------------------------------------------------


class TestBuildRawOutputPath:
    def test_standard_path_construction(self):
        meta = RoundMetadata(company_id="3B Wonen", valuation_round_id=100, issue_date="2024-12-31", data_set_id=500)
        path = build_raw_output_path("market_value_parameters", meta)
        assert "3B Wonen" in str(path)
        assert "2024" in str(path)
        assert path.suffix in (".xlsx", ".json")

    def test_path_with_subsidiary(self):
        meta = RoundMetadata(company_id="TestCo", valuation_round_id=1, issue_date="2025-12-31")
        path = build_raw_output_path("difference_analysis", meta, subsidiary="SubA")
        assert "SubA" in str(path)


# ---------------------------------------------------------------------------
# get_sheet
# ---------------------------------------------------------------------------


class TestGetSheet:
    def test_exact_match(self):
        from market_presentation.defs.utils.tms_report_utils import get_sheet

        workbook = {"Sheet1": pd.DataFrame({"a": [1]}), "Other": pd.DataFrame({"b": [2]})}
        result = get_sheet(workbook, "Sheet1")
        assert result is not None
        assert "a" in result.columns

    def test_case_insensitive_match(self):
        from market_presentation.defs.utils.tms_report_utils import get_sheet

        workbook = {"SHEET1": pd.DataFrame({"a": [1]})}
        result = get_sheet(workbook, "sheet1")
        assert result is not None

    def test_partial_match(self):
        from market_presentation.defs.utils.tms_report_utils import get_sheet

        workbook = {"Marktwaardeparameters VHE": pd.DataFrame({"x": [1]})}
        result = get_sheet(workbook, "Marktwaardeparameters")
        assert result is not None

    def test_no_match_returns_none(self):
        from market_presentation.defs.utils.tms_report_utils import get_sheet

        workbook = {"Sheet1": pd.DataFrame({"a": [1]})}
        result = get_sheet(workbook, "NonExistent")
        assert result is None

    def test_first_candidate_wins(self):
        from market_presentation.defs.utils.tms_report_utils import get_sheet

        workbook = {"First": pd.DataFrame({"a": [1]}), "Second": pd.DataFrame({"b": [2]})}
        result = get_sheet(workbook, "First", "Second")
        assert result is not None
        assert "a" in result.columns


# ---------------------------------------------------------------------------
# dataframe_preview
# ---------------------------------------------------------------------------


class TestDataframePreview:
    def test_empty_df_returns_placeholder(self):
        from market_presentation.defs.utils.tms_report_utils import dataframe_preview

        assert dataframe_preview(pd.DataFrame()) == "(empty)"

    def test_non_empty_df_returns_markdown(self):
        from market_presentation.defs.utils.tms_report_utils import dataframe_preview

        df = pd.DataFrame({"col": [1, 2, 3]})
        result = dataframe_preview(df)
        assert "col" in result
        assert "|" in result


# ---------------------------------------------------------------------------
# round_metadata_from_row
# ---------------------------------------------------------------------------


class TestRoundMetadataFromRow:
    def test_reconstructs_metadata(self):
        from market_presentation.defs.utils.tms_report_utils import round_metadata_from_row

        row = pd.Series({"Corporatie": "TestCo", "valuation_round_id": 42, "issue_date": "2024-12-31", "data_set_id": 100})
        meta = round_metadata_from_row(row)
        assert meta.company_id == "TestCo"
        assert meta.valuation_round_id == 42
        assert meta.issue_date == "2024-12-31"
        assert meta.data_set_id == 100

    def test_handles_nan_data_set_id(self):
        from market_presentation.defs.utils.tms_report_utils import round_metadata_from_row

        row = pd.Series({"Corporatie": "Co", "valuation_round_id": 1, "issue_date": "2025-12-31", "data_set_id": float("nan")})
        meta = round_metadata_from_row(row)
        assert meta.data_set_id is None

    def test_handles_none_data_set_id(self):
        from market_presentation.defs.utils.tms_report_utils import round_metadata_from_row

        row = pd.Series({"Corporatie": "Co", "valuation_round_id": 1, "issue_date": "2025-12-31", "data_set_id": None})
        meta = round_metadata_from_row(row)
        assert meta.data_set_id is None
