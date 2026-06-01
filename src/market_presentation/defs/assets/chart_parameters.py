"""charts_parameters — Marktwaardeparameter charts: COROP maps and trendlines."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.assets.chart_constants import CHART_YEAR, CHART_YEAR_M1
from market_presentation.defs.utils.graph_utils import chart_appender, multi_layer_map, of_template, save_charts, save_graph, set_line_marker_style, set_template_axis_format

of_template()

LOC = "COROP-plusgebieden Code"


def _build_parameters_charts(
    *,
    verschil_basis_full_corop: pd.DataFrame,
    uitponden_corop: pd.DataFrame,
    uitponden_ontwikkeling_corop: pd.DataFrame,
    dv_corop: pd.DataFrame,
    dv_ontwikkeling_corop: pd.DataFrame,
    trendlijnen: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build parameter analysis charts."""
    p = "nieuwe figuren"
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []
    add = chart_appender(charts, p)

    # Pre-compute detail for COROP map Excel exports (VHE-level per COROP)
    _detail_cols = [
        "Corporatie",
        "VHE-nr",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        "Waarderingstype",
        "Marktwaarde",
        "Marktwaarde basis",
        "DV doorexploiteren",
        "EY doorexploiteren",
        "MU jaar 1-15",
        "Beleidswaarde",
        "Beleidshuur",
        "Beleidsbeheer_bp",
        "Beleidsonderhoud",
    ]
    _available = [c for c in _detail_cols if c in basis_source.columns]
    _detail_base = basis_source[_available]
    _group_cols = ["COROP-plusgebieden Code", "COROP-plusgebieden Naam", "Waarderingstype"]
    _value_cols = [
        c
        for c in ["Marktwaarde", "Marktwaarde basis", "DV doorexploiteren", "EY doorexploiteren", "MU jaar 1-15", "Beleidswaarde", "Beleidshuur", "Beleidsbeheer_bp", "Beleidsonderhoud"]
        if c in _detail_base.columns
    ]
    _agg_dict: dict = {c: "median" for c in _value_cols}
    _agg_dict["VHE-nr"] = "count"
    _detail_agg = _detail_base.groupby([c for c in _group_cols if c in _detail_base.columns], dropna=False).agg(_agg_dict).reset_index()
    _detail_agg = _detail_agg.rename(columns={"VHE-nr": "Aantal VHEs"})

    def _detail_for(agg_df: pd.DataFrame, wtype: str | None = None) -> pd.DataFrame:
        codes = set(agg_df[LOC].dropna().unique()) if LOC in agg_df.columns else set()
        out = _detail_agg.loc[_detail_agg[LOC].isin(codes)].copy() if codes else _detail_agg.copy()
        if wtype and "Waarderingstype" in out.columns:
            out = out.loc[out["Waarderingstype"] == wtype]
        return out

    _classificaties = ["DAEB", "Niet-DAEB", "Totaal"]

    # --- Item 2: Verschil marktwaarde basis-full per COROP (per Classificatie) ---
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        df_verschil = verschil_basis_full_corop.loc[verschil_basis_full_corop["Classificatie"] == classificatie].copy()
        if not df_verschil.empty:
            fig_verschil = multi_layer_map(df_verschil, location_col=LOC, value_col="Verschil basis-full", use_percent=True)
            add(f"Verschil marktwaarde basis-full per COROP-plusgebied{suffix}", fig_verschil, df_verschil, _detail_for(df_verschil))

    # --- Item 4: Percentage uitponden per COROP (basis/full) (per Classificatie) ---
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        for wtype in ["Basis", "Full"]:
            df_subset = uitponden_corop.loc[(uitponden_corop["Waarderingstype"] == wtype) & (uitponden_corop["Classificatie"] == classificatie)].copy()
            if not df_subset.empty:
                fig = multi_layer_map(df_subset, location_col=LOC, value_col="Percentage uitponden", use_percent=True)
                add(f"Percentage uitponden per COROP-plusgebied {wtype}{suffix}", fig, df_subset, _detail_for(df_subset, wtype))

    # --- Item 5: Ontwikkeling percentage uitponden per COROP (basis/full) (per Classificatie) ---
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        for wtype in ["Basis", "Full"]:
            df_subset = uitponden_ontwikkeling_corop.loc[(uitponden_ontwikkeling_corop["Waarderingstype"] == wtype) & (uitponden_ontwikkeling_corop["Classificatie"] == classificatie)].copy()
            if not df_subset.empty:
                fig = multi_layer_map(df_subset, location_col=LOC, value_col="Ontwikkeling uitponden", use_percent=True)
                add(f"Ontwikkeling percentage uitponden per COROP-plusgebied {wtype}{suffix}", fig, df_subset, _detail_for(df_subset, wtype))

    # --- Item 7: Gehanteerd DV per COROP (basis/full/totaal) (per Classificatie) ---
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        for wtype in ["Basis", "Full", "Totaal"]:
            df_subset = dv_corop.loc[(dv_corop["Waarderingstype"] == wtype) & (dv_corop["Classificatie"] == classificatie)].copy()
            if not df_subset.empty:
                fig = multi_layer_map(df_subset, location_col=LOC, value_col="Median DV", use_percent=True)
                add(f"Gehanteerde disconteringsvoet per COROP-plusgebied {wtype}{suffix}", fig, df_subset, _detail_for(df_subset, wtype if wtype != "Totaal" else None))

    # --- Item 6: Ontwikkeling DV per COROP (basis/full/totaal) (per Classificatie) ---
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        for wtype in ["Basis", "Full", "Totaal"]:
            df_subset = dv_ontwikkeling_corop.loc[(dv_ontwikkeling_corop["Waarderingstype"] == wtype) & (dv_ontwikkeling_corop["Classificatie"] == classificatie)].copy()
            if not df_subset.empty:
                fig = multi_layer_map(df_subset, location_col=LOC, value_col="Ontwikkeling DV", use_percent=True)
                add(f"Ontwikkeling disconteringsvoet per COROP-plusgebied {wtype}{suffix}", fig, df_subset, _detail_for(df_subset, wtype if wtype != "Totaal" else None))

    # --- Item 8: Trendlijnen parameters (basis/full) ---
    param_metrics = [
        ("Median LW", "Leegwaarde", False),
        ("Median Markthuur", "Markthuur", False),
        ("Median EY", "Exit yield", True),
        ("Median DV", "Disconteringsvoet", True),
    ]
    for col, label, use_pct in param_metrics:
        df_plot = trendlijnen[["Jaar", "Waarderingstype", col]].dropna(subset=[col]).copy()
        fig = px.line(df_plot, x="Jaar", y=col, color="Waarderingstype", markers=True, template="ortec_finance")
        set_line_marker_style(fig)
        fig.update_layout(xaxis_title="Jaar", yaxis_title=label)
        fig.update_xaxes(tickmode="linear", dtick=1)
        if use_pct:
            set_template_axis_format(fig)
        add(f"Trendlijn {label} per waarderingstype", fig, df_plot)

    # Basis-full difference trendlines
    pivot = trendlijnen.pivot_table(index="Jaar", columns="Waarderingstype", values=["Median LW", "Median Markthuur", "Median EY", "Median DV"]).reset_index()
    if "Basis" in trendlijnen["Waarderingstype"].values and "Full" in trendlijnen["Waarderingstype"].values:
        for col, label, use_pct in param_metrics:
            basis_vals = trendlijnen.loc[trendlijnen["Waarderingstype"] == "Basis", ["Jaar", col]].set_index("Jaar")
            full_vals = trendlijnen.loc[trendlijnen["Waarderingstype"] == "Full", ["Jaar", col]].set_index("Jaar")
            diff = (basis_vals[col] - full_vals[col]).reset_index()
            diff.columns = ["Jaar", "Verschil basis-full"]
            fig = px.line(diff, x="Jaar", y="Verschil basis-full", markers=True, template="ortec_finance")
            set_line_marker_style(fig)
            fig.update_layout(xaxis_title="Jaar", yaxis_title=f"Verschil {label} (basis - full)")
            fig.update_xaxes(tickmode="linear", dtick=1)
            fig.update_traces(line_color="#146EB4")
            if use_pct:
                set_template_axis_format(fig)
            add(f"Trendlijn verschil basis-full {label}", fig, diff)

    return charts


@dg.asset(
    group_name="charts",
    name="charts_parameters",
    description="Marktwaardeparameter charts: COROP maps for uitponden/DV/basis-full difference, plus parameter trendlines.",
    ins={
        "verschil_basis_full_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__verschil_basis_full_corop"])),
        "uitponden_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__uitponden_corop"])),
        "uitponden_ontwikkeling_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__uitponden_ontwikkeling_corop"])),
        "dv_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__dv_corop"])),
        "dv_ontwikkeling_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__dv_ontwikkeling_corop"])),
        "trendlijnen": dg.AssetIn(key=dg.AssetKey(["models", "chart_parameters__trendlijnen"])),
        "basis_source": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__basis_source"])),
    },
)
def charts_parameters(
    context: dg.AssetExecutionContext,
    verschil_basis_full_corop: pd.DataFrame,
    uitponden_corop: pd.DataFrame,
    uitponden_ontwikkeling_corop: pd.DataFrame,
    dv_corop: pd.DataFrame,
    dv_ontwikkeling_corop: pd.DataFrame,
    trendlijnen: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> None:
    """Generate parameter analysis charts: COROP maps + trendlines."""
    save_charts(
        _build_parameters_charts(
            verschil_basis_full_corop=verschil_basis_full_corop,
            uitponden_corop=uitponden_corop,
            uitponden_ontwikkeling_corop=uitponden_ontwikkeling_corop,
            dv_corop=dv_corop,
            dv_ontwikkeling_corop=dv_ontwikkeling_corop,
            trendlijnen=trendlijnen,
            basis_source=basis_source,
        ),
        context,
        save_fn=save_graph,
    )
