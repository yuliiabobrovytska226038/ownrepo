"""Dagster assets for external (non-API) data sources.

These assets load reference data that is not sourced from the TMS/VGR APIs:
CBS Open Data tables, dVi housing corporation disclosures, local GeoJSON
boundary files, postcode mapping files, and historical benchmark data.  All
DataFrame assets are persisted to DuckDB via the IO manager, where dbt staging
models pick them up.

Assets:
    external/postal_code_mapping
        Postcode → COROP → province → gemeente mapping (local text file).
    external/cbs_corop_regions
        CBS 85755NED — regionale kerncijfers Nederland (geographic regions).
    external/cbs_house_price_index
        CBS 85773NED — bestaande koopwoningen prijsindex with YoY mutation.
    external/dvi_housing_data
        dVi housing corporation disclosures (data.overheid.nl).
    external/geographic_boundaries
        Cartomap GeoJSON boundaries for COROP-plus regions and provinces.
    external/historical_data
        Marktpresentatie Historie — historical benchmark data (Excel).
"""

import io
import json

import cbsodata
import dagster as dg
import pandas as pd
import requests

from ..config import CHART_YEAR, DATA_DIR
from ..utils.tms_utils import normalize_column_names


def _read_excel_sheets_with_sheet_column(source, context: dg.AssetExecutionContext, *, log_columns: bool = False) -> tuple[pd.DataFrame, int]:
    sheets = pd.read_excel(source, sheet_name=None)
    frames: list[pd.DataFrame] = []
    for sheet_name, sheet_df in sheets.items():
        sheet_df = sheet_df.assign(_sheet=sheet_name)
        frames.append(sheet_df)
        detail = f"{len(sheet_df)} rows, {len(sheet_df.columns)} columns" if log_columns else f"{len(sheet_df)} rows"
        context.log.info(f"  Sheet '{sheet_name}': {detail}")
    return normalize_column_names(pd.concat(frames, ignore_index=True)), len(sheets)


# ---------------------------------------------------------------------------
# 2.3.1  Postal codes – Waarderingsparameters postcode mapping
# ---------------------------------------------------------------------------
@dg.asset(
    key_prefix=["external"],
    name="postal_code_mapping",
    compute_kind="file",
    group_name="external_sources",
    description="Waarderingsparameters postcode mapping – maps postal codes to COROP regions and provinces.",
)
def postal_code_mapping(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Parse the [Postcodes] section from the Waarderingsparameters text file.

    Expected file: data/extern/Waarderingsparameters {CHART_YEAR}.txt
    Handles two formats:
      - 2021 format: 7 fields (Postcode  COROP-gebied  Indexatiegebied  Regio  Gemeentecode  Gemeente  Krimp/aardbeving)
      - Other years:  6 fields (Postcode  COROP-gebied  Regio  Gemeentecode  Gemeente  Full/aardbeving)
    """
    path = DATA_DIR / "DLL" / f"Waarderingsparameters {CHART_YEAR}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Waarderingsparameters file not found. Place '{path.name}' in {path.parent.resolve()}")
    is_2021_format = "2021" in path.name
    context.log.info(f"Parsing postcode mapping from {path.name} (2021 format: {is_2021_format})")

    records: list[dict[str, str]] = []
    in_section = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Postcodes]":
                in_section = True
                continue
            if in_section and stripped.startswith("["):
                break  # next section
            if not in_section or not stripped or stripped.startswith("#") or stripped.startswith("Postcode") or stripped.startswith("Checksum"):
                continue
            parts = stripped.split()
            if is_2021_format and len(parts) >= 7:
                # 2021: Postcode  COROP-gebied  Indexatiegebied  Regio  Gemeentecode  Gemeente  Krimp/aardbeving
                records.append(
                    {
                        "postcode": parts[0],
                        "corop_gebied": parts[1],
                        "regio": parts[3],
                        "gemeentecode": parts[4],
                        "gemeente": parts[5],
                        "categorie": parts[6],
                    }
                )
            elif len(parts) >= 6:
                records.append(
                    {
                        "postcode": parts[0],
                        "corop_gebied": parts[1],
                        "regio": parts[2],
                        "gemeentecode": parts[3],
                        "gemeente": parts[4],
                        "categorie": parts[5],
                    }
                )

    df = pd.DataFrame(records)
    df = normalize_column_names(df)
    context.log.info(f"Loaded {len(df)} postcode mappings from {path.name}")
    return df


# ---------------------------------------------------------------------------
# 2.3.2  CBS COROP regions – 85755NED
# ---------------------------------------------------------------------------
@dg.asset(
    key_prefix=["external"],
    name="cbs_corop_regions",
    compute_kind="cbs_api",
    group_name="external_sources",
    description="CBS 85755NED – Regionale kerncijfers Nederland (COROP geographic data).",
)
def cbs_corop_regions(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Fetch CBS table 85755NED (regionale kerncijfers) via the CBS Open Data API.

    Post-processing:
      - Renames coded columns to readable Dutch names
      - Strips whitespace from string columns
      - Caches results locally (reuses cache if < 30 days old)
      - Returns only the relevant geographic mapping columns
    """
    table_id = "85755NED"
    cache_dir = DATA_DIR / "cbs" / table_id
    cache_file = cache_dir / "TypedDataset.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse local cache if it exists and is less than 30 days old
    if cache_file.exists() and cache_file.stat().st_mtime >= (pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp():
        context.log.info(f"Loading CBS {table_id} from local cache: {cache_file}")
        df = pd.read_json(cache_file)
    else:
        context.log.info(f"Fetching CBS table {table_id} via cbsodata...")
        df = pd.DataFrame(cbsodata.get_data(table_id, dir=str(cache_dir)))

    # Rename coded columns to readable names
    rename_dict = {
        "Code_1": "Gemeentecode",
        "Code_14": "COROP-plusgebieden Code",
        "Naam_15": "COROP-plusgebieden Naam",
        "Code_28": "Provincies Code",
        "Naam_29": "Provincies Naam",
    }
    df.rename(columns=rename_dict, inplace=True)

    # Strip whitespace from all string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Keep only the relevant geographic columns
    df = df[list(rename_dict.values())]

    context.log.info(f"Loaded CBS {table_id}: {len(df)} rows, columns: {list(df.columns)}")
    # normalize_column_names() maps COROP-plusgebieden Code → corop_code, etc.
    df = normalize_column_names(df)
    return df


# ---------------------------------------------------------------------------
# 2.3.3  CBS house price index – 85773NED
# ---------------------------------------------------------------------------
@dg.asset(
    key_prefix=["external"],
    name="cbs_house_price_index",
    compute_kind="cbs_api",
    group_name="external_sources",
    description="CBS 85773NED – Bestaande koopwoningen; prijsindex with Dec-to-Dec price mutation calculation.",
)
def cbs_house_price_index(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Fetch CBS table 85773NED (house price index) and calculate year-over-year mutations.

    Calculates December-to-December price mutations for each available year:
        mutatie = (PrijsindexVerkoopprijzen[year] - PrijsindexVerkoopprijzen[year-1]) / PrijsindexVerkoopprijzen[year-1]

    Returns the full price index DataFrame enriched with a ``mutatie_yoy`` column.
    """
    table_id = "85773NED"
    cache_dir = DATA_DIR / "cbs" / table_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    context.log.info(f"Fetching CBS table {table_id} via cbsodata...")
    df = pd.DataFrame(cbsodata.get_data(table_id, dir=str(cache_dir)))

    # Extract December rows and calculate year-over-year mutation
    december_mask = df["Perioden"].str.contains("december", case=False, na=False)
    df_dec = df.loc[december_mask].copy()
    df_dec["jaar"] = df_dec["Perioden"].str.extract(r"(\d{4})").astype(int)
    df_dec = df_dec.sort_values("jaar").reset_index(drop=True)

    # Calculate mutation: (value_year - value_year-1) / value_year-1
    prijsindex_col = "PrijsindexVerkoopprijzen_1"
    if prijsindex_col in df_dec.columns:
        df_dec["mutatie_yoy"] = df_dec[prijsindex_col].pct_change()
        context.log.info(f"Calculated YoY mutations for {len(df_dec)} December periods")
        for _, row in df_dec.dropna(subset=["mutatie_yoy"]).tail(5).iterrows():
            context.log.info(f"  {int(row['jaar'])}: {row['mutatie_yoy']:.4f} ({row['mutatie_yoy'] * 100:.2f}%)")
    else:
        context.log.warning(f"Column '{prijsindex_col}' not found in CBS data. Available: {list(df.columns)}")

    # Merge mutation back into the full DataFrame
    mutation_lookup = df_dec.set_index("jaar")["mutatie_yoy"].to_dict() if "mutatie_yoy" in df_dec.columns else {}
    df["jaar"] = df["Perioden"].str.extract(r"(\d{4})")
    df["jaar"] = pd.to_numeric(df["jaar"], errors="coerce")
    df["mutatie_yoy"] = df["jaar"].map(mutation_lookup)

    context.log.info(f"Loaded CBS {table_id}: {len(df)} rows, {len(df.columns)} columns")
    df = normalize_column_names(df)
    return df


# ---------------------------------------------------------------------------
# 2.3.4  dVi housing corporation data
# ---------------------------------------------------------------------------
@dg.asset(
    key_prefix=["external"],
    name="dvi_housing_data",
    compute_kind="api",
    group_name="external_sources",
    description="dVi (deelnemersinformatie) housing corporation data – downloaded from data.overheid.nl.",
)
def dvi_housing_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Download dVi housing corporation data (dVi2024 H1-H5) from data.overheid.nl.

    Source: https://data.overheid.nl/dataset/verantwoordingsinformatie-woningcorporaties-dvi2024-hfd1-tm-hfd5
    Direct download: dVi2024 H1-H5.xlsx

    All sheets are loaded into a single DataFrame with a '_sheet' column.
    Row-level filtering (residential units, KVK_nummer not null) is done
    downstream in the dbt staging model stg_dvi_housing.
    """

    dvi_url = "https://data.overheid.nl/sites/default/files/dataset/9dceb9da-e2e1-43f9-920f-48fa01831f56/resources/dVi2024%20H1-H5.xlsx"
    context.log.info(f"Downloading dVi data from {dvi_url}")

    response = requests.get(dvi_url, timeout=60)
    response.raise_for_status()

    # Cache locally for reference
    cache_path = DATA_DIR / "dvi"
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file = cache_path / "dVi2024 H1-H5.xlsx"
    local_file.write_bytes(response.content)
    context.log.info(f"Cached dVi file to {local_file} ({len(response.content)} bytes)")

    df, sheet_count = _read_excel_sheets_with_sheet_column(io.BytesIO(response.content), context)
    context.log.info(f"Loaded dVi data: {len(df)} total rows across {sheet_count} sheets")
    return df


# ---------------------------------------------------------------------------
# 2.3.5  Geographic boundaries — COROP-plus regions and provinces (GeoJSON)
# ---------------------------------------------------------------------------
@dg.asset(
    key_prefix=["external"],
    name="geographic_boundaries",
    compute_kind="file",
    group_name="external_sources",
    description="CBS/Cartomap GeoJSON boundaries for COROP-plus regions and provinces (2025).",
)
def geographic_boundaries(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load cartomap GeoJSON boundary files into a flat DataFrame.

    Reads coropplusgebied_2025.geojson and provincie_2025.geojson from
    data/extern/cartomap/. Each GeoJSON Feature becomes one row with the
    geometry stored as a JSON string for downstream use.

    Expected files:
      data/extern/cartomap/coropplusgebied_2025.geojson  (52 COROP-plus regions)
      data/extern/cartomap/provincie_2025.geojson         (12 provinces)
    """
    cartomap_dir = DATA_DIR / "cartomap"
    geojson_files = sorted(cartomap_dir.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in {cartomap_dir.resolve()}")

    rows: list[dict] = []
    for f in geojson_files:
        data = json.loads(f.read_bytes())
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            rows.append(
                {
                    "source_file": f.name,
                    "statcode": props.get("statcode"),
                    "jrstatcode": props.get("jrstatcode"),
                    "statnaam": props.get("statnaam"),
                    "rubriek": props.get("rubriek"),
                    "feature_id": props.get("id"),
                    "geometry_type": (feature.get("geometry") or {}).get("type"),
                    "geometry_json": json.dumps(feature.get("geometry"), ensure_ascii=False),
                }
            )
        context.log.info(f"  {f.name}: {len(data.get('features', []))} features loaded")

    df = pd.DataFrame(rows)
    context.log.info(f"Loaded geographic boundaries: {len(df)} features from {len(geojson_files)} files")
    df = normalize_column_names(df)
    return df


# ---------------------------------------------------------------------------
# 2.3.6  Historical data – Marktpresentatie Historie
# ---------------------------------------------------------------------------


@dg.asset(
    key_prefix=["external"],
    name="historical_data",
    compute_kind="file",
    group_name="external_sources",
    description="Marktpresentatie Historie – historical market presentation data (Waardeontwikkeling, Disconteringsvoet, etc.).",
)
def historical_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load historical market presentation data from 'Marktpresentatie Historie.xlsx'.

    The file contains four sheets:
      - Waardeontwikkeling: value development over time (Basis, Full, MSCI, CBS, etc.)
      - Waardeontwikkeling per model: value development per valuation model (Woningen, Parkeren, BOG/MOG/ZOG)
      - Disconteringsvoet: historical average discount rates (Doorexploiteren / Uitponden)
      - Disconteringsvoet scenarios: median DV DE / DV UP per year

    All sheets are combined into a single DataFrame with a '_sheet' column.
    """
    path = DATA_DIR / "Marktpresentatie Historie.xlsx"

    if not path.exists():
        raise FileNotFoundError(f"Historical data file not found. Place 'Marktpresentatie Historie.xlsx' in {DATA_DIR.resolve()}")

    df, sheet_count = _read_excel_sheets_with_sheet_column(path, context, log_columns=True)
    context.log.info(f"Loaded historical data from {path.name}: {len(df)} total rows across {sheet_count} sheets")
    return df
