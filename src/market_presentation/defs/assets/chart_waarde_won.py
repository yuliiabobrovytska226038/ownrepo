"""charts_analyse_waarde_won — woningen valuation chart assets."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import chart_appender, multi_layer_map, of_template, save_charts, save_graph, set_line_marker_style, set_template_axis_format

of_template()

CLASSIFICATIES = ("Totaal", "DAEB", "Niet-DAEB")


def _classificatie_label(classificatie: str) -> str:
    return "niet-DAEB" if classificatie == "Niet-DAEB" else classificatie.lower()


def _add_boxplot_mean_annotation(fig: go.Figure, values: pd.Series, *, is_percent: bool = False) -> None:
    """Add a text annotation showing the mean value on a horizontal boxplot."""
    mean_val = values.mean()
    text = f"Gem: {mean_val:.1%}" if is_percent else f"Gem: €{mean_val:,.0f}".replace(",", ".")
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(x=mean_val, y=1.05, yref="paper", text=text, showarrow=False, font={"size": 18, "color": "red"})


def _boxplot_median_waarde(median_waarde: pd.DataFrame, waarde: str, classificatie: str) -> tuple[go.Figure, pd.DataFrame]:
    col = waarde.lower()
    t = median_waarde.loc[(median_waarde["Waarde"] == waarde) & (median_waarde["Classificatie"] == classificatie)].copy()
    t = t.rename(columns={"Median waarde": f"median_{col}"})
    fig = px.box(t, x=f"median_{col}", hover_name="Corporatie", template="ortec_finance")
    fig.update_xaxes(title_text=f"Mediane {waarde.lower()} per VHE {classificatie}")
    _add_boxplot_mean_annotation(fig, t[f"median_{col}"])
    return fig, t


def _boxplot_efg(efg: pd.DataFrame, classificatie: str) -> tuple[go.Figure, pd.DataFrame]:
    t = efg.loc[efg["Classificatie"] == classificatie].copy()
    fig = px.box(t, x="pct_EFG", hover_name="Corporatie", template="ortec_finance")
    fig.update_xaxes(title_text=f"Percentage EFG-labels per Corporatie {classificatie}", tickformat="0%")
    _add_boxplot_mean_annotation(fig, t["pct_EFG"], is_percent=True)
    return fig, t


def _waterfall_blw(waterfall: pd.DataFrame, classificatie: str) -> tuple[go.Figure, pd.DataFrame]:
    t = waterfall.loc[waterfall["Classificatie"] == classificatie].sort_values("Volgorde").copy()
    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=t["Measure"].to_list(),
            x=t["Stap"].to_list(),
            text=t["Text"].to_list(),
            y=t["Procent"].to_list(),
            connector=None,
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "#0084CB"}},
        )
    )
    fig.update_layout(template="ortec_finance")
    fig.update_yaxes(range=[0, 1.05], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.05], tickformat=".0%", showgrid=True)
    fig.add_hline(y=1.05, line_dash="solid", line_color="lightgrey", line_width=1)
    set_template_axis_format(fig, hoverformat=".1%")
    return fig, t


def _waterfall_steps_blw(waterfall_steps: pd.DataFrame) -> dict[str, tuple[go.Figure, pd.DataFrame]]:
    """Build per-step boxplots from pre-filtered long-format waterfall data."""
    graphs: dict[str, tuple[go.Figure, pd.DataFrame]] = {}
    for step, group in waterfall_steps.groupby("Stap", sort=False):
        t_plot = group.reset_index(drop=True)
        fig = px.box(t_plot, x="Value", hover_name="Corporatie", template="ortec_finance")
        if t_plot["Value"].max() < 1:
            fig.layout.template.layout.xaxis.tickformat = "0%"
            fig.layout.template.layout.xaxis.hoverformat = ".1%"
        graphs[f"Boxplot blw waterval {step.lower()}"] = fig, t_plot
    return graphs


def _build_waarde_won_beleidswaarde_charts(
    *,
    median_waarde: pd.DataFrame,
    efg: pd.DataFrame,
    ratio: pd.DataFrame,
    waterfall: pd.DataFrame,
    waterfall_steps: pd.DataFrame,
    huurstijging: pd.DataFrame,
    perc_full_coropplus: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build beleidswaarde wonanalyse charts without writing files."""
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []
    add = chart_appender(charts, "waarde_won")

    # Pre-compute detail subset for COROP map exports (VHE-level per COROP)
    _detail_cols = [
        "Corporatie",
        "VHE-nr",
        "Waarderingscomplex",
        "COROP-plusgebieden Code",
        "COROP-plusgebieden Naam",
        "Postcode",
        "Gemeente",
        "Classificatie",
        "Waarderingstype",
        "% Full",
        "Marktwaarde",
        "Beleidswaarde",
        "LW waarde",
        "DV doorexploiteren",
        "Beleidshuur",
        "Beleidsbeheer_bp",
        "Beleidsonderhoud",
    ]
    _available_cols = [c for c in _detail_cols if c in basis_source.columns]
    _detail_base = basis_source[_available_cols]

    # Aggregate to one row per COROP region (VHE-level median + count)
    _group_cols = ["COROP-plusgebieden Code", "COROP-plusgebieden Naam"]
    _value_cols = [c for c in ["Marktwaarde", "Beleidswaarde", "LW waarde", "DV doorexploiteren", "Beleidshuur", "Beleidsbeheer_bp", "Beleidsonderhoud", "% Full"] if c in _detail_base.columns]
    _agg_dict = {c: "median" for c in _value_cols}
    _agg_dict["VHE-nr"] = "count"
    _detail_agg = _detail_base.groupby([c for c in _group_cols if c in _detail_base.columns], dropna=False).agg(_agg_dict).reset_index()
    _detail_agg = _detail_agg.rename(columns={"VHE-nr": "Aantal VHEs"})

    def _detail_for(agg_df: pd.DataFrame) -> pd.DataFrame:
        """Return detail rows filtered to COROP codes present in the aggregate."""
        corop_col = "COROP-plusgebieden Code"
        if corop_col not in agg_df.columns:
            return _detail_agg
        codes = set(agg_df[corop_col].dropna().unique())
        return _detail_agg.loc[_detail_agg[corop_col].isin(codes)].copy()

    for waarde in ["Beleidswaarde", "Marktwaarde"]:
        for classificatie in CLASSIFICATIES:
            fig, t = _boxplot_median_waarde(median_waarde, waarde, classificatie)
            add(f"Boxplot Mediane {waarde.lower()} per VHE {_classificatie_label(classificatie)}", fig, t)

    for classificatie in CLASSIFICATIES:
        fig, t = _boxplot_efg(efg, classificatie)
        add(f"Boxplot percentage EFG-labels per corporatie {_classificatie_label(classificatie)}", fig, t)

    fig_ratio = px.box(ratio, x="ratio", hover_name="Corporatie", template="ortec_finance", color_discrete_sequence=["#F58025"])
    fig_ratio.update_traces(marker_color=fig_ratio.layout.template.layout.colorway[1])
    fig_ratio.update_xaxes(title_text="Verhouding beleidswaarde / marktwaarde", tickformat="0%", hoverformat="0%")
    _add_boxplot_mean_annotation(fig_ratio, ratio["ratio"], is_percent=True)
    add("Verhouding beleidswaarde marktwaarde", fig_ratio, ratio)

    for classificatie in CLASSIFICATIES:
        fig, t = _waterfall_blw(waterfall, classificatie)
        add(f"Beleidswaarde waterval {classificatie}", fig, t)

    for name, (fig, t) in _waterfall_steps_blw(waterfall_steps).items():
        add(name, fig, t)

    fig_hs = px.line(huurstijging, x="Jaar", y="Huurstijging", color="Classificatie", markers=True, template="ortec_finance")
    set_line_marker_style(fig_hs)
    fig_hs.update_layout(xaxis_title="Jaar", yaxis_title="Gemiddelde huurstijging (%)")
    set_template_axis_format(fig_hs, tickformat=".1%")
    add("Huurstijging Beleidswaarde", fig_hs, huurstijging)

    # Percentage full per COROP (per Classificatie)
    _classificaties_map = ["DAEB", "Niet-DAEB", "Totaal"]
    for classificatie in _classificaties_map:
        suffix = f" ({classificatie})"
        df_full = perc_full_coropplus.loc[perc_full_coropplus["Classificatie"] == classificatie].copy()
        if not df_full.empty:
            add(f"Percentage full per COROP-plusgebied{suffix}", multi_layer_map(df_full, location_col="COROP-plusgebieden Code", value_col="% Full"), df_full, _detail_for(df_full))
    return charts


def _build_waarde_won_marktwaarde_charts(dekking_woningen: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build marktwaarde wonanalyse charts without writing files."""
    fig = multi_layer_map(dekking_woningen, location_col="Provincies Code", value_col="Dekking naar woningen")
    return [("waarde_won/Dekking woningen", fig, dekking_woningen)]


@dg.asset(
    group_name="charts",
    name="charts_analyse_waarde_won_beleidswaarde",
    description="Beleidswaarde wonanalyse: boxplots, waterfall, huurstijging, COROP maps.",
    ins={
        "median_waarde": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__median_waarde"])),
        "efg": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__efg"])),
        "ratio": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__ratio"])),
        "waterfall": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__waterfall"])),
        "waterfall_steps": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__waterfall_steps"])),
        "huurstijging": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__huurstijging"])),
        "perc_full_coropplus": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__perc_full_coropplus"])),
        "basis_source": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__basis_source"])),
    },
)
def charts_analyse_waarde_won_beleidswaarde(
    context: dg.AssetExecutionContext,
    median_waarde: pd.DataFrame,
    efg: pd.DataFrame,
    ratio: pd.DataFrame,
    waterfall: pd.DataFrame,
    waterfall_steps: pd.DataFrame,
    huurstijging: pd.DataFrame,
    perc_full_coropplus: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> None:
    """Beleidswaarde charts: shared boxplots + waterfall + huurstijging + COROP maps."""
    save_charts(
        _build_waarde_won_beleidswaarde_charts(
            median_waarde=median_waarde,
            efg=efg,
            ratio=ratio,
            waterfall=waterfall,
            waterfall_steps=waterfall_steps,
            huurstijging=huurstijging,
            perc_full_coropplus=perc_full_coropplus,
            basis_source=basis_source,
        ),
        context,
        save_fn=save_graph,
    )


@dg.asset(
    group_name="charts",
    name="charts_analyse_waarde_won_marktwaarde",
    description="Marktwaarde wonanalyse: dekking woningen map.",
    ins={
        "dekking_woningen": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde_won__dekking_woningen"])),
    },
)
def charts_analyse_waarde_won_marktwaarde(context: dg.AssetExecutionContext, dekking_woningen: pd.DataFrame) -> None:
    """Marktwaarde charts: dekking woningen per provincie map."""
    save_charts(_build_waarde_won_marktwaarde_charts(dekking_woningen), context, save_fn=save_graph)
