"""Shared utility functions for TMS report asset implementations.

Provides helpers for:
- Extracting valuation round metadata from company_valuation_data
- Reading Excel workbooks (from bytes or file path)
- Saving raw files to disk with legacy-compatible naming
- Adding Corporatie/Jaar/Peildatum metadata columns (matching legacy tms.db)
- Building MaterializeResult with standard metadata
"""

import io
import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import dagster as dg
import pandas as pd
import pandera.pandas as pa

from ..config import OUTPUT_DIR
from ..partitions import get_company_partition_keys

logger = logging.getLogger(__name__)
T = TypeVar("T")
SchemaModel = type[pa.DataFrameModel]
type SkipRows = int | list[int] | None
type WorkbookSkipRows = SkipRows | dict[str, SkipRows]
type SheetMap = list[tuple[str, str, tuple[str, ...]]]
type SchemaByOutput = dict[str, SchemaModel]


# Standard dependency / input declarations for company_valuation_data.
COMPANY_VALUATION_DATA_DEP = dg.AssetDep(dg.AssetKey(["tms", "company_valuation_data"]))

# Map of report keys to (report_display_name, subfolder) for file persistence
_RAW_OUTPUT_SPECS: dict[str, tuple[str, str, str]] = {
    # key: (display_name, subfolder, extension)
    "market_value_parameters": ("Marktwaarde parameters", "marktwaardeparameters", ".xlsx"),
    "policy_value_parameters": ("Beleidswaarde parameters", "beleidswaardeparameters", ".xlsx"),
    "complex_references": ("Complex referenties", "complex_referenties", ".xlsx"),
    "complex_characteristics": ("Complex kenmerken", "complex_kenmerken", ".xlsx"),
    "parameters_overview": ("Parameteroverzicht", "parameteroverzicht", ".xlsx"),
    "valuation_overview": ("Waardeoverzicht", "waardeoverzicht", ".xlsx"),
    "ratios": ("Ratio rapport", "ratios", ".xlsx"),
    "policy_value_report": ("Beleidswaarderapport", "beleidswaarde", ".xlsx"),
    "real_estate_data": ("Vastgoedgegevens", "vastgoedgegevens", ".xlsx"),
    "difference_analysis": ("TMS Verschillenanalyse", "tms_verschillenanalyse", ".json"),
    "waterfall_analysis": ("Beleidswaarde waterfall analysis", "beleidswaarde_waterfall_analysis", ".json"),
    "energy_performance": ("EP2", "ep2", ".json"),
    "market_value_basis": ("Marktwaarde basis", "marktwaarde_basis", ".json"),
}


@dataclass(frozen=True)
class RoundMetadata:
    """Metadata for a single valuation round, extracted from company_valuation_data."""

    company_id: str
    valuation_round_id: int
    issue_date: str  # "YYYY-12-31"
    data_set_id: int | None = None

    @property
    def issue_year(self) -> int:
        return int(self.issue_date[:4])


class RawReportConfig(dg.Config):
    """Run config shared by raw API assets."""

    force_download: bool = False


@dataclass(frozen=True)
class CacheResult(Generic[T]):  # noqa: UP046 - project metadata still allows Python 3.10.
    """Payload loaded from either disk cache or a fresh API download."""

    value: T
    path: Path
    source: str


def get_round_metadata(company_id: str, company_valuation_data: pd.DataFrame) -> list[RoundMetadata]:
    """Extract unique RoundMetadata rows from a company_valuation_data DataFrame."""
    if company_valuation_data.empty or "valuation_round_id" not in company_valuation_data.columns:
        return []
    if "company_id" in company_valuation_data.columns:
        company_valuation_data = company_valuation_data[company_valuation_data["company_id"].astype(str) == company_id]

    rounds = []
    seen: set[tuple[int, str, int | None]] = set()
    for _, row in company_valuation_data.iterrows():
        data_set_id = row.get("data_set_id") if "data_set_id" in row else row.get("round_data_set_id")
        data_set_id = None if pd.isna(data_set_id) or str(data_set_id).strip() == "" else int(data_set_id)
        issue_date = row.get("issue_date")
        if pd.isna(issue_date) or str(issue_date).strip() == "":
            issue_year = int(row["issue_year"])
            issue_date = f"{issue_year}-12-31"
        issue_date = str(issue_date)
        key = (int(row["valuation_round_id"]), issue_date, data_set_id)
        if key in seen:
            continue
        seen.add(key)
        rounds.append(
            RoundMetadata(
                company_id=company_id,
                valuation_round_id=int(row["valuation_round_id"]),
                issue_date=issue_date,
                data_set_id=data_set_id,
            )
        )
    return rounds


def add_metadata_columns(df: pd.DataFrame, meta: RoundMetadata) -> pd.DataFrame:
    """Add Corporatie, Jaar, Peildatum columns matching the legacy tms.db convention."""
    result = df.copy()
    result["Corporatie"] = meta.company_id
    result["Jaar"] = meta.issue_year
    result["Peildatum"] = pd.Timestamp(meta.issue_date)
    return result


# ---------------------------------------------------------------------------
# Download/parse split helpers
# ---------------------------------------------------------------------------

_FILES_DF_BASE_COLUMNS = ("Corporatie", "valuation_round_id", "issue_date", "data_set_id", "file_path", "source")


def build_files_output(context: dg.AssetExecutionContext, records: list[dict], *, paths: list[str] | None = None, sources: list[str] | None = None) -> pd.DataFrame:
    """Attach standard raw-file metadata and return the manifest DataFrame."""
    paths = [r["file_path"] for r in records] if paths is None else paths
    sources = [r["source"] for r in records] if sources is None else sources
    add_raw_metadata(context, paths=paths, sources=sources)
    if not records:
        return pd.DataFrame(columns=list(_FILES_DF_BASE_COLUMNS))

    df = pd.DataFrame(records)
    standard = [column for column in _FILES_DF_BASE_COLUMNS if column in df.columns]
    extra = [column for column in df.columns if column not in _FILES_DF_BASE_COLUMNS]
    return df[standard + extra]


def build_sheet_asset_outs(sheet_map: SheetMap, *, report_label: str | None = None) -> dict[str, dg.AssetOut]:
    """Build standard AssetOut declarations for report sheets."""
    description_prefix = "Sheet" if report_label is None else f"{report_label} sheet"
    return {
        output_name: dg.AssetOut(
            key=["tms", output_name],
            description=f"{description_prefix} '{candidates[0]}' → legacy table {legacy_table}.",
            metadata={"partition_expr": "Corporatie", "legacy_table": legacy_table},
            group_name="tms_assets",
        )
        for output_name, legacy_table, candidates in sheet_map
    }


def build_file_record(meta: RoundMetadata, cache: CacheResult, *, data_set_id: int | None = None, **extra: object) -> dict:
    """Build the standard file-manifest row used by raw download assets."""
    return {
        "Corporatie": meta.company_id,
        "valuation_round_id": meta.valuation_round_id,
        "issue_date": meta.issue_date,
        "data_set_id": meta.data_set_id if data_set_id is None else data_set_id,
        "file_path": str(cache.path.resolve()),
        "source": cache.source,
        **extra,
    }


def _download_round_files(
    context: dg.AssetExecutionContext,
    company_valuation_data: pd.DataFrame,
    *,
    report_key: str,
    load_cache: Callable[[str, RoundMetadata, Path], CacheResult],
) -> pd.DataFrame:
    records: list[dict] = []
    for company_id in get_company_partition_keys(context):
        for meta in get_round_metadata(company_id, company_valuation_data):
            cache = load_cache(company_id, meta, build_raw_output_path(report_key, meta))
            records.append(build_file_record(meta, cache))
    return build_files_output(context, records)


def download_round_bytes_files(
    context: dg.AssetExecutionContext,
    config: RawReportConfig,
    company_valuation_data: pd.DataFrame,
    *,
    report_key: str,
    fetch_fn: Callable[[str, RoundMetadata], bytes | None],
) -> pd.DataFrame:
    """Download or reuse one cached byte report per company valuation round."""
    return _download_round_files(
        context,
        company_valuation_data,
        report_key=report_key,
        load_cache=lambda company_id, meta, path: load_or_fetch_bytes(path, lambda: fetch_fn(company_id, meta), force_download=config.force_download),
    )


def download_round_report_files(
    context: dg.AssetExecutionContext,
    config: RawReportConfig,
    company_valuation_data: pd.DataFrame,
    *,
    report_key: str,
    fetch_fn: Callable[[str, RoundMetadata, Path], bool],
) -> pd.DataFrame:
    """Download or reuse one trigger/poll report file per company valuation round."""
    return _download_round_files(
        context,
        company_valuation_data,
        report_key=report_key,
        load_cache=lambda company_id, meta, path: load_or_fetch_report_file(path, lambda target: fetch_fn(company_id, meta, target), force_download=config.force_download),
    )


def download_round_json_files(
    context: dg.AssetExecutionContext,
    config: RawReportConfig,
    company_valuation_data: pd.DataFrame,
    *,
    report_key: str,
    fetch_fn: Callable[[str, RoundMetadata], object],
) -> pd.DataFrame:
    """Download or reuse one cached JSON report per company valuation round."""
    return _download_round_files(
        context,
        company_valuation_data,
        report_key=report_key,
        load_cache=lambda company_id, meta, path: load_or_fetch_json(path, lambda: fetch_fn(company_id, meta), force_download=config.force_download),
    )


def resolve_vgr_data_set_id(context: dg.AssetExecutionContext, vgr_api, company_id: str, meta: RoundMetadata) -> int | None:
    """Resolve the VGR data_set_id for assets that can fall back to the latest VGR dataset."""
    if meta.data_set_id is not None:
        return meta.data_set_id

    datasets = vgr_api.get_vgr_datasets(company_id, [meta.issue_year])
    if not datasets:
        context.log.warning(f"No VGR dataset for {company_id} year {meta.issue_year}")
        return None

    ds = max(datasets, key=lambda d: d["data_set_id"])
    context.log.warning(f"company_valuation_data has no data_set_id for {company_id}/{meta.issue_year}; falling back to latest VGR dataset {ds['data_set_id']}")
    return int(ds["data_set_id"])


def round_metadata_from_row(row: pd.Series) -> RoundMetadata:
    """Reconstruct a RoundMetadata from a row of a files DataFrame."""
    raw_ds = row.get("data_set_id")
    data_set_id = None if (raw_ds is None or (isinstance(raw_ds, float) and pd.isna(raw_ds))) else int(raw_ds)
    return RoundMetadata(
        company_id=str(row["Corporatie"]),
        valuation_round_id=int(row["valuation_round_id"]),
        issue_date=str(row["issue_date"]),
        data_set_id=data_set_id,
    )


# ---------------------------------------------------------------------------
# Raw file persistence (legacy-compatible naming)
# ---------------------------------------------------------------------------


def build_raw_output_path(report_key: str, meta: RoundMetadata, subsidiary: str | None = None) -> Path:
    """Build output file path: data/tms/<subfolder>/<display_name> <company> <year>[.ext]"""
    display_name, subfolder, ext = _RAW_OUTPUT_SPECS[report_key]
    filename = f"{display_name} {meta.company_id} {meta.issue_year}"
    if subsidiary is not None:
        filename += f" @ {subsidiary}"
    filename += ext
    return OUTPUT_DIR / subfolder / filename


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip()


def build_api_cache_path(cache_key: str, company_id: str | None = None, extension: str = ".json") -> Path:
    """Build a local JSON cache path for API endpoints that are not reports."""
    filename = cache_key
    if company_id is not None:
        filename += f" {company_id}"
    return OUTPUT_DIR / "api" / f"{_safe_filename(filename)}{extension}"


def _cache_available(path: Path, *, force_download: bool) -> bool:
    return not force_download and path.exists() and path.stat().st_size > 0


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_or_fetch_bytes(path: Path, fetch_fn: Callable[[], bytes | None], *, force_download: bool = False) -> CacheResult[bytes]:
    """Read bytes from disk unless forced or missing, otherwise fetch and save."""
    if _cache_available(path, force_download=force_download):
        return CacheResult(path.read_bytes(), path, "cache")

    content = fetch_fn()
    if content is None:
        raise dg.Failure(description=f"Failed to download raw bytes for {path}")
    _write_bytes(path, content)
    return CacheResult(content, path, "download")


def load_or_fetch_json(path: Path, fetch_fn: Callable[[], object], *, force_download: bool = False) -> CacheResult[object]:
    """Read JSON from disk unless forced or invalid, otherwise fetch and save."""
    if _cache_available(path, force_download=force_download):
        try:
            return CacheResult(json.loads(path.read_text(encoding="utf-8")), path, "cache")
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid JSON cache file %s", path)

    data = fetch_fn()
    if data is None:
        raise dg.Failure(description=f"Failed to download raw JSON for {path}")
    _write_json(path, data)
    return CacheResult(data, path, "download")


def load_or_fetch_report_file(path: Path, fetch_fn: Callable[[Path], bool], *, force_download: bool = False) -> CacheResult[Path]:
    """Ensure a trigger/poll/download report file exists, respecting cache policy."""
    if _cache_available(path, force_download=force_download):
        return CacheResult(path, path, "cache")

    path.parent.mkdir(parents=True, exist_ok=True)
    if not fetch_fn(path):
        raise dg.Failure(description=f"Failed to download report file to {path}")
    return CacheResult(path, path, "download")


def add_raw_metadata(context: dg.AssetExecutionContext, *, paths: list[str] | None = None, sources: list[str] | None = None) -> None:
    """Attach raw cache/download metadata to a single asset output."""
    metadata: dict[str, dg.MetadataValue] = {}
    if paths:
        metadata["raw_paths"] = dg.MetadataValue.text(", ".join(paths))
    if sources:
        cache_count = sum(1 for source in sources if source == "cache")
        download_count = sum(1 for source in sources if source == "download")
        metadata["raw_cache_hits"] = dg.MetadataValue.int(cache_count)
        metadata["raw_downloads"] = dg.MetadataValue.int(download_count)
    if metadata:
        context.add_output_metadata(metadata)


def validate_dataframe(df: pd.DataFrame, schema_model: SchemaModel) -> pd.DataFrame:
    """Validate and coerce *df* before handing it to DuckDB."""
    schema = schema_model.to_schema()
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame({column: pd.Series(dtype="object") for column in schema.columns})
    df = schema_model.validate(df, lazy=True)

    for column_name, column_schema in schema.columns.items():
        if column_name in df.columns and str(column_schema.dtype) == "str":
            df[column_name] = df[column_name].astype("string")
    return df


def align_to_schema_columns(df: pd.DataFrame, schema_model: SchemaModel, *, keep_extra: bool = False) -> pd.DataFrame:
    """Add missing schema columns and order schema columns first."""
    schema_columns = schema_model.to_schema().columns
    result = df.copy()
    for column, column_schema in schema_columns.items():
        if column not in result.columns:
            result[column] = float("nan") if str(column_schema.dtype) == "float64" else None

    ordered_columns = list(schema_columns)
    if keep_extra:
        ordered_columns.extend(column for column in result.columns if column not in schema_columns)
    return result.loc[:, ordered_columns]


# ---------------------------------------------------------------------------
# Excel workbook reading (preserves original Dutch column names)
# ---------------------------------------------------------------------------

# Identifier/code columns where Excel inference can irreversibly lose exact text,
# such as leading zeroes. Report-specific semantic types belong in Pandera schemas.
_IDENTIFIER_STRING_COLUMNS = frozenset(
    {
        "VHE-nr",
        "VHE-nummer",
        "Complexcode",
        "Waarderingscomplex",
        "Postcode",
        "BAG pand id (opgezocht)",
        "BAG verblijfsobject id (opgezocht)",
    }
)
_EXCEL_DTYPES = dict.fromkeys(_IDENTIFIER_STRING_COLUMNS, str)

# Columns whose dtype name starts with "Bron" are source indicator strings
_STRING_PREFIXES = ("Bron ",)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-dtype columns to numeric where possible.

    Identifier columns or columns starting with ``_STRING_PREFIXES``
    are kept as strings. All other ``object`` columns are tested: if ≥50 % of
    non-null values convert to a number, the column is coerced to float64.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if col in _IDENTIFIER_STRING_COLUMNS:
            continue
        if any(col.startswith(p) for p in _STRING_PREFIXES):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and numeric.notna().sum() / non_null >= 0.5:
            df[col] = numeric
    return df


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames from multiple rounds and re-coerce numeric columns.

    ``pd.concat`` produces ``object`` dtype when one round has float64 and
    another has object (e.g. all-NaN). Re-running coercion fixes this.
    """
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return coerce_numeric_columns(df)


def dataframe_preview(df: pd.DataFrame, rows: int = 5) -> str:
    """Render a small markdown preview without leaking pandas NA into tabulate."""
    if df.empty:
        return "(empty)"
    preview_df = df.head(rows).copy()
    preview_df = preview_df.astype(object).where(preview_df.notna(), None)
    return preview_df.to_markdown()


def read_workbook(content_or_path: bytes | Path, skiprows: WorkbookSkipRows = 0) -> dict[str, pd.DataFrame]:
    """Read all sheets from an Excel workbook, preserving original Dutch column names.

    Returns a dict of {sheet_name: DataFrame} with minimal cleaning:
    - Applies the same explicit ``skiprows`` configuration as the legacy parser
    - Converts datetime.time columns to strings (DuckDB compat)
    """
    import datetime as dt

    excel_source = io.BytesIO(content_or_path) if isinstance(content_or_path, bytes) else content_or_path
    if isinstance(skiprows, dict):
        workbook_file = pd.ExcelFile(excel_source)
        workbook = {sheet_name: pd.read_excel(workbook_file, sheet_name=sheet_name, skiprows=skiprows.get(sheet_name, 0), dtype=_EXCEL_DTYPES) for sheet_name in workbook_file.sheet_names}
    else:
        workbook = pd.read_excel(excel_source, sheet_name=None, skiprows=skiprows, dtype=_EXCEL_DTYPES)

    result = {}
    for sheet_name, df in workbook.items():
        # Convert time columns to string for DuckDB
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dt.time)).any():
                df[col] = df[col].astype(str)
        # Coerce object columns to numeric where appropriate
        df = coerce_numeric_columns(df)
        result[sheet_name] = df
    return result


def get_sheet(workbook: dict[str, pd.DataFrame], *candidate_names: str) -> pd.DataFrame | None:
    """Find a sheet by trying multiple candidate names (case-insensitive)."""
    name_map = {k.lower().strip(): k for k in workbook}
    for candidate in candidate_names:
        key = candidate.lower().strip()
        if key in name_map:
            return workbook[name_map[key]]
        # Try partial match
        for existing_key, original_name in name_map.items():
            if key in existing_key or existing_key in key:
                return workbook[original_name]
    return None


def iter_existing_file_rows(context: dg.AssetExecutionContext, files_df: pd.DataFrame) -> Iterator[tuple[pd.Series, RoundMetadata, Path]]:
    """Yield manifest rows with reconstructed metadata and existing file paths."""
    for _, row in files_df.iterrows():
        meta = round_metadata_from_row(row)
        file_path = Path(row["file_path"])
        if not file_path.exists():
            context.log.warning(f"File not found: {file_path}; skipping {meta.company_id}/{meta.valuation_round_id}")
            continue
        yield row, meta, file_path


def iter_json_file_rows(  # noqa: UP047 - project metadata still allows Python 3.10.
    context: dg.AssetExecutionContext,
    files_df: pd.DataFrame,
    *,
    expected_type: type[T],
    expected_label: str,
) -> Iterator[tuple[pd.Series, RoundMetadata, Path, T]]:
    """Yield existing file rows with parsed JSON payloads of the expected type."""
    for row, meta, file_path in iter_existing_file_rows(context, files_df):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, expected_type):
            raise dg.Failure(description=f"Invalid JSON for {meta.company_id}/{meta.valuation_round_id} at {file_path}: expected {expected_label}")
        yield row, meta, file_path, data


def parse_single_sheet_files(
    context: dg.AssetExecutionContext,
    files_df: pd.DataFrame,
    *,
    output_name: str,
    sheet_candidates: tuple[str, ...],
    schema: SchemaModel,
    skiprows: WorkbookSkipRows = None,
) -> pd.DataFrame:
    """Parse cached Excel files that expose one legacy output sheet."""
    skiprows = [1] if skiprows is None else skiprows
    frames: list[pd.DataFrame] = []
    for _, meta, file_path in iter_existing_file_rows(context, files_df):
        sheets = read_workbook(file_path.read_bytes(), skiprows=skiprows)
        sheet_df = get_sheet(sheets, *sheet_candidates)
        if sheet_df is not None and not sheet_df.empty:
            frames.append(add_metadata_columns(sheet_df, meta))
            continue

        first_sheet = next(iter(sheets.values()), None)
        if first_sheet is not None and not first_sheet.empty:
            frames.append(add_metadata_columns(first_sheet, meta))
        else:
            context.log.warning(f"Empty workbook for {meta.company_id}/{meta.valuation_round_id}")

    df = concat_frames(frames)
    context.log.info(f"{output_name}: {len(df)} rows, {len(df.columns)} columns")
    return validate_dataframe(df, schema)


# ---------------------------------------------------------------------------
# MaterializeResult builder
# ---------------------------------------------------------------------------


def build_result(
    output_name: str,
    df: pd.DataFrame,
    *,
    schema: SchemaModel,
    rounds: int = 0,
    paths: list[str] | None = None,
    sources: list[str] | None = None,
) -> dg.Output:
    """Wrap a DataFrame in an Output for multi_asset yielding.

    Uses ``dg.Output`` with matching output_name for the multi_asset outs dict.
    Ensures empty DataFrames have at least the metadata columns so DuckDB
    doesn't reject a zero-column DataFrame.
    """
    df = validate_dataframe(df, schema)

    metadata = {
        "row_count": dg.MetadataValue.int(len(df)),
        "column_count": dg.MetadataValue.int(len(df.columns)),
        "rounds_processed": dg.MetadataValue.int(rounds),
    }
    if paths:
        metadata["raw_paths"] = dg.MetadataValue.text(", ".join(paths))
    if sources:
        metadata["raw_cache_hits"] = dg.MetadataValue.int(sum(1 for source in sources if source == "cache"))
        metadata["raw_downloads"] = dg.MetadataValue.int(sum(1 for source in sources if source == "download"))
    metadata["preview"] = dg.MetadataValue.md(dataframe_preview(df))
    return dg.Output(value=df, output_name=output_name, metadata=metadata)


def parse_and_yield_multi_sheet_results(
    context: dg.AssetExecutionContext,
    files_df: pd.DataFrame,
    *,
    sheet_map: SheetMap,
    schema_by_output: SchemaByOutput,
    skiprows: WorkbookSkipRows,
    prepare_frame: Callable[[str, pd.DataFrame], pd.DataFrame] | None = None,
) -> Iterator[dg.Output]:
    """Parse cached Excel files and yield standard Outputs for each mapped sheet."""
    frames_by_output: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _ in sheet_map}
    for _, meta, file_path in iter_existing_file_rows(context, files_df):
        sheets = read_workbook(file_path, skiprows=skiprows)
        for output_name, _, candidates in sheet_map:
            sheet_df = get_sheet(sheets, *candidates)
            if sheet_df is not None and not sheet_df.empty:
                frames_by_output[output_name].append(add_metadata_columns(sheet_df, meta))
            else:
                context.log.warning(f"Missing sheet {candidates[0]} for {meta.company_id}/{meta.valuation_round_id}")

    for output_name, _, _ in sheet_map:
        df = concat_frames(frames_by_output[output_name])
        context.log.info(f"{output_name}: {len(df)} rows, {len(df.columns)} columns")
        if prepare_frame is not None:
            df = prepare_frame(output_name, df)
        yield build_result(output_name, df, schema=schema_by_output[output_name], rounds=len(files_df))
