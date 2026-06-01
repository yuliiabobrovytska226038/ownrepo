"""Plotly graph helper functions for the Marktpresentatie visualisations.

Provides shared utilities for all chart assets in the pipeline:

- ``of_template()`` — Registers the Ortec Finance corporate Plotly template
  with brand colours (blue, orange, green, yellow, grey) and Fira Sans font.
- ``save_graph()`` — Exports a Plotly figure to HTML and optionally JPEG in
  the ``grafieken/`` directory.
- ``multi_layer_map()`` — Builds two-layer choropleth maps of the Netherlands
  using MapLibre tiles, with grey fills for missing regions and coloured fills
  for data regions.  Supports both COROP-plus and province boundaries.

Chart assets import these helpers to maintain consistent styling and output
locations across all 40+ presentation charts.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import dagster as dg
import geopandas as gpd
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from ..config import DATA_DIR, GRAFIEKEN_DIR

ChartSpec = tuple[str, go.Figure, pd.DataFrame] | tuple[str, go.Figure, pd.DataFrame, pd.DataFrame]


# Sets up template layout callable by 'ortec_finance'
def of_template():
    """Register the Ortec Finance corporate Plotly template.

    Creates and registers a template named ``'ortec_finance'`` based on the
    ``plotly_white`` template, with Ortec Finance brand colours, Fira Sans
    font, Dutch decimal separators, and a default font size of 24px.
    Called once at module import time by ``chart_assets.py``.
    """
    ortec_colors = [
        "#0084CB",  # blue
        "#F58025",  # orange
        "#87BB40",  # green
        "#FCAF43",  # yellow
        "#A8A9AC",  # grey
        "#D0D3D6",  # light grey
        "#6A75A1",
        "#21407A",
    ]
    of_template = pio.templates["plotly_white"]
    of_template.layout.colorway = ortec_colors
    of_template.layout.font.color = "#000000"
    of_template.layout.font.family = "Fira Sans"
    of_template.layout.font.size = 24
    of_template.layout.separators = ",."
    pio.templates["ortec_finance"] = of_template


def save_graph(fig, filename, context: dg.AssetExecutionContext, df: pd.DataFrame | None = None, export_image: bool = True, detail_df: pd.DataFrame | None = None) -> Path:
    """Write a Plotly figure to HTML (and optionally JPEG + XLSX) in the grafieken/ folder.

    Args:
        fig: Plotly Figure object to export.
        filename: Base filename (without extension) for the output files.
        context: Dagster execution context for logging.
        df: Optional DataFrame whose contents are saved alongside the chart as XLSX.
        export_image: If True, also export a JPEG alongside the HTML.
        detail_df: Optional VHE-level detail DataFrame. When provided, the Excel
            file is written with two sheets: 'Samenvatting' (the aggregate df)
            and 'Detail' (the per-VHE/company breakdown).

    Returns:
        Path to the saved HTML file.

    Raises:
        ValueError: If the figure has no data traces (empty source DataFrame).
    """
    if not fig.data:
        raise ValueError(f"Chart '{filename}' has no data traces — the source DataFrame was empty. Check dbt models and year configuration (ISSUE_YEARS in config.py, dbt var('jaar')).")
    if df is not None and len(df) == 0:
        raise ValueError(f"Chart '{filename}' data DataFrame is empty — no rows to display. Check dbt models and year configuration (ISSUE_YEARS in config.py, dbt var('jaar')).")
    html_path = GRAFIEKEN_DIR / f"{filename}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(html_path))
    if export_image:
        try:
            image_path = html_path.with_suffix(".jpeg")
            pio.write_image(fig, str(image_path), width=1920, height=1080, scale=2)
        except Exception as e:
            context.log.error("Could not export image (%s). HTML file was saved successfully.", e)
    if df is not None:
        try:
            xlsx_path = html_path.with_suffix(".xlsx")
            if detail_df is not None and len(detail_df) > 0:
                with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Samenvatting", index=False)
                    detail_df.to_excel(writer, sheet_name="Detail", index=False)
            else:
                df.to_excel(str(xlsx_path), index=False)
        except Exception as e:
            context.log.error("Could not export xlsx (%s).", e)
    context.log.info("Saved: %s", filename)
    return html_path


def set_template_axis_format(fig: go.Figure, *, axis: str = "y", tickformat: str = "0%", hoverformat: str = ".2%") -> go.Figure:
    """Set the template axis formats used by the legacy chart assets."""
    layout_axis = getattr(fig.layout.template.layout, f"{axis}axis")
    layout_axis.tickformat = tickformat
    layout_axis.hoverformat = hoverformat
    return fig


def set_bar_text_horizontal(fig: go.Figure) -> go.Figure:
    """Keep Plotly bar labels horizontal."""
    fig.update_traces(textangle=0)
    return fig


def set_line_marker_style(fig: go.Figure, *, width: int = 4, size: int = 8) -> go.Figure:
    """Apply the line and marker style repeated across chart assets."""
    fig.update_traces(line={"width": width}, marker={"size": size})
    return fig


def chart_appender(charts: list[ChartSpec], prefix: str) -> Callable[..., None]:
    """Return an appender for chart triples that share an output folder prefix.

    Accepts an optional fourth argument: a detail DataFrame for multi-sheet Excel export.
    """

    def add(label: str, fig: go.Figure, df: pd.DataFrame, detail_df: pd.DataFrame | None = None) -> None:
        if detail_df is not None:
            charts.append((f"{prefix}/{label}", fig, df, detail_df))
        else:
            charts.append((f"{prefix}/{label}", fig, df))

    return add


def save_charts(charts: Iterable[ChartSpec], context: dg.AssetExecutionContext, *, save_fn=save_graph) -> None:
    """Save a sequence of ``(path, figure, dataframe[, detail_df])`` chart specs."""
    for spec in charts:
        if len(spec) == 4:
            path, fig, df, detail_df = spec
            save_fn(fig, path, context, df=df, detail_df=detail_df)
        else:
            path, fig, df = spec
            save_fn(fig, path, context, df=df)


def multi_layer_map(
    df: pd.DataFrame,
    location_col: str,
    value_col: str,
    ls_hover_data: list[str] | None = None,
    use_percent: bool = True,
    overrule_colorscale: str | None = None,
    colorbar_title: str | None = None,
    hoverformat: str | None = None,
):
    """Create a two-layer choropleth map of the Netherlands.

    Uses ``go.Choropleth`` with ``fitbounds='locations'`` to render only the
    Netherlands shape on a blank background, matching the legacy presentation
    style.  The first layer shows areas without data in grey; the second layer
    colours areas with values using the specified colour scale.

    Args:
        df: DataFrame with at least ``location_col`` and ``value_col`` columns.
        location_col: Column containing CBS statcode identifiers for geographic
            matching (e.g. ``'COROPPLUSCODE'`` or ``'Provincies Code'``).
        value_col: Column containing the numeric values to visualise.
        ls_hover_data: Additional columns to include in hover tooltips.
        use_percent: If True, format values and colour bar as percentages.
        overrule_colorscale: Plotly colour scale name to override the default
            (``'RdBu'`` for mixed, ``'Blues'`` for positive-only data).
        colorbar_title: Title text for the colour bar legend.

    Returns:
        Plotly Figure with the choropleth map.
    """
    geojson_file = "provincie_2025.geojson" if location_col == "Provincies Code" else "coropplusgebied_2025.geojson"
    nl_geojson = json.loads((DATA_DIR / "cartomap" / geojson_file).read_text(encoding="utf-8"))

    gdf = gpd.GeoDataFrame.from_features(nl_geojson)
    all_areas = pd.DataFrame({location_col: gdf["statcode"]})
    merged_df = all_areas.merge(df, on=location_col, how="left")

    df_nan = merged_df[merged_df[value_col].isna()]
    df_nonnan = merged_df[merged_df[value_col].notna()]

    colorscale = overrule_colorscale if overrule_colorscale else ("RdBu" if df_nonnan[value_col].min() < 0 else "Blues")

    hovertemplate = "%{z:.2%}<extra></extra>" if use_percent else (hoverformat if hoverformat else "%{z:,.0f}<extra></extra>")
    colorbar = {"tickformat": ".0%"} if use_percent else {}
    if colorbar_title:
        colorbar["title"] = {"text": colorbar_title, "side": "right"}

    traces = []
    if not df_nan.empty:
        traces.append(
            go.Choropleth(
                geojson=nl_geojson,
                featureidkey="properties.statcode",
                locations=df_nan[location_col],
                z=df_nan[value_col].fillna(0),
                name="Geen data",
                autocolorscale=False,
                colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                showscale=False,
                hovertemplate=hovertemplate,
                marker={"line": {"color": "white", "width": 0.5}},
            )
        )

    traces.append(
        go.Choropleth(
            geojson=nl_geojson,
            featureidkey="properties.statcode",
            locations=df_nonnan[location_col],
            z=df_nonnan[value_col],
            name="Data",
            autocolorscale=False,
            colorscale=colorscale,
            showscale=True,
            hovertemplate=hovertemplate,
            colorbar=colorbar,
            zmid=0 if colorscale == "RdBu" else None,
            marker={"line": {"color": "white", "width": 0.5}},
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        geo={
            "scope": "europe",
            "showframe": False,
            "showcountries": False,
            "showcoastlines": False,
            "showland": False,
            "fitbounds": "locations",
        },
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
    )
    return fig
