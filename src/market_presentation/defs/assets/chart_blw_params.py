"""charts_blw_params — Beleidswaardeparameter charts: COROP maps, distributions, and realisatie boxplots."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import chart_appender, multi_layer_map, of_template, save_charts, save_graph, set_template_axis_format

of_template()

LOC = "COROP-plusgebieden Code"


def _add_boxplot_mean_annotation(fig: go.Figure, values: pd.Series, *, is_percent: bool = False) -> None:
    """Add a text annotation showing the mean value on a horizontal boxplot."""
    mean_val = values.mean()
    text = f"Gem: {mean_val:.1%}" if is_percent else f"Gem: {mean_val:.4f}"
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(x=mean_val, y=1.05, yref="paper", text=text, showarrow=False, font={"size": 18, "color": "red"})


def _build_blw_params_charts(
    *,
    beleidsonderhoud_ontwikkeling_corop: pd.DataFrame,
    beleidsbeheer_ontwikkeling_corop: pd.DataFrame,
    beleidshuur_ontwikkeling_corop: pd.DataFrame,
    verdeling_huurcategorieen: pd.DataFrame,
    verdeling_huurcategorieen_ontwikkeling: pd.DataFrame,
    lengte_mjob: pd.DataFrame,
    realisatie_huurstijging: pd.DataFrame,
    realisatie_beleidshuur: pd.DataFrame,
    realisatie_mjob: pd.DataFrame,
    detail_source: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build beleidswaardeparameter charts."""
    p = "nieuwe figuren"
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []
    add = chart_appender(charts, p)

    # Pre-compute detail for COROP map Excel exports (VHE-level per COROP)
    _group_cols = ["COROP-plusgebieden Code", "COROP-plusgebieden Naam"]
    _value_cols = [
        c
        for c in ["Beleidsonderhoud", "Beleidsonderhoud vorig jaar", "Beleidsbeheer", "Beleidsbeheer vorig jaar", "Beleidshuur", "Beleidshuur vorig jaar", "Beleidswaarde", "Marktwaarde", "Netto huur"]
        if c in detail_source.columns
    ]
    _agg_dict: dict = {c: "median" for c in _value_cols}
    if "VHE-nr" in detail_source.columns:
        _agg_dict["VHE-nr"] = "count"
    _detail_agg = detail_source.groupby([c for c in _group_cols if c in detail_source.columns], dropna=False).agg(_agg_dict).reset_index()
    if "VHE-nr" in _detail_agg.columns:
        _detail_agg = _detail_agg.rename(columns={"VHE-nr": "Aantal VHEs"})

    def _detail_for(agg_df: pd.DataFrame) -> pd.DataFrame:
        codes = set(agg_df[LOC].dropna().unique()) if LOC in agg_df.columns else set()
        return _detail_agg.loc[_detail_agg[LOC].isin(codes)].copy() if codes else _detail_agg.copy()

    # --- Items 10-12: Ontwikkeling beleidsonderhoud/beheer/huur per COROP (per Classificatie) ---
    _classificaties = ["DAEB", "Niet-DAEB", "Totaal"]

    for classificatie in _classificaties:
        suffix = f" ({classificatie})"

        # --- Ontwikkeling beleidsonderhoud ---
        df_bo = beleidsonderhoud_ontwikkeling_corop.loc[beleidsonderhoud_ontwikkeling_corop["Classificatie"] == classificatie].copy()
        if not df_bo.empty:
            fig = multi_layer_map(df_bo, location_col=LOC, value_col="Ontwikkeling beleidsonderhoud")
            add(f"Ontwikkeling beleidsonderhoud per COROP-plusgebied{suffix}", fig, df_bo, _detail_for(df_bo))

        # --- Ontwikkeling beleidsbeheer ---
        df_bb = beleidsbeheer_ontwikkeling_corop.loc[beleidsbeheer_ontwikkeling_corop["Classificatie"] == classificatie].copy()
        if not df_bb.empty:
            fig = multi_layer_map(df_bb, location_col=LOC, value_col="Ontwikkeling beleidsbeheer")
            add(f"Ontwikkeling beleidsbeheer per COROP-plusgebied{suffix}", fig, df_bb, _detail_for(df_bb))

        # --- Ontwikkeling beleidshuur ---
        df_bh = beleidshuur_ontwikkeling_corop.loc[beleidshuur_ontwikkeling_corop["Classificatie"] == classificatie].copy()
        if not df_bh.empty:
            fig = multi_layer_map(df_bh, location_col=LOC, value_col="Ontwikkeling beleidshuur")
            add(f"Ontwikkeling beleidshuur per COROP-plusgebied{suffix}", fig, df_bh, _detail_for(df_bh))

    # --- Item 13: Verdeling beleidshuren over huurcategorieën ---
    fig = px.pie(
        verdeling_huurcategorieen,
        names="Segment huurregime",
        values="Aantal VHE",
        template="ortec_finance",
    )
    fig.update_traces(textinfo="percent+label")
    add("Verdeling beleidshuren over huurcategorieen", fig, verdeling_huurcategorieen)

    # --- Item 14: Ontwikkeling verdeling beleidshuren over huurcategorieën ---
    df_ontw = verdeling_huurcategorieen_ontwikkeling.copy()
    fig = px.bar(
        df_ontw,
        x="Segment huurregime",
        y=["Percentage huidig", "Percentage vorig jaar"],
        barmode="group",
        template="ortec_finance",
    )
    set_template_axis_format(fig, tickformat=".1%")
    fig.update_layout(xaxis_title="", yaxis_title="Aandeel", legend_title="Jaar")
    add("Ontwikkeling verdeling beleidshuren over huurcategorieen", fig, df_ontw)

    # --- Item 15: Lengte MJOB ---
    fig = px.bar(
        lengte_mjob.sort_values("sort_order"),
        x="Lengte MJOB groep",
        y="Aantal VHE",
        template="ortec_finance",
    )
    fig.update_layout(xaxis_title="Lengte MJOB (jaren)", yaxis_title="Aantal VHE")
    add("Lengte MJOB verdeling", fig, lengte_mjob)

    # --- Item 16: Boxplot realisatie reguliere huurstijging ---
    fig = px.box(
        realisatie_huurstijging,
        x="Mediaan verschil realisatie",
        hover_name="Corporatie",
        template="ortec_finance",
    )
    fig.update_xaxes(title_text="Verschil realisatie - verwachte huurstijging")
    set_template_axis_format(fig, tickformat=".2%")
    _add_boxplot_mean_annotation(fig, realisatie_huurstijging["Mediaan verschil realisatie"], is_percent=True)
    add("Boxplot realisatie reguliere huurstijging", fig, realisatie_huurstijging)

    # --- Item 17: Boxplot realisatie beleidshuur ---
    fig = px.box(
        realisatie_beleidshuur,
        x="Percentage netto huur = beleidshuur",
        hover_name="Corporatie",
        template="ortec_finance",
    )
    fig.update_xaxes(title_text="% woningen netto huur = beleidshuur (mutaties)")
    set_template_axis_format(fig, tickformat=".0%")
    _add_boxplot_mean_annotation(fig, realisatie_beleidshuur["Percentage netto huur = beleidshuur"], is_percent=True)
    add("Boxplot realisatie beleidshuur", fig, realisatie_beleidshuur)

    # --- Item 18: Boxplot realisatie MJOB ---
    fig = px.box(
        realisatie_mjob,
        x="Mediaan verschil realisatie MJOB",
        hover_name="Corporatie",
        template="ortec_finance",
    )
    fig.update_xaxes(title_text="Verschil realisatie - verwachte beleidsonderhoud (MJOB)")
    set_template_axis_format(fig, tickformat=".1%")
    _add_boxplot_mean_annotation(fig, realisatie_mjob["Mediaan verschil realisatie MJOB"], is_percent=True)
    add("Boxplot realisatie MJOB", fig, realisatie_mjob)

    return charts


@dg.asset(
    group_name="charts",
    name="charts_blw_params",
    description="Beleidswaardeparameter charts: COROP ontwikkeling maps, huurcategorieën, lengte MJOB, realisatie boxplots.",
    ins={
        "beleidsonderhoud_ontwikkeling_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__beleidsonderhoud_ontwikkeling_corop"])),
        "beleidsbeheer_ontwikkeling_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__beleidsbeheer_ontwikkeling_corop"])),
        "beleidshuur_ontwikkeling_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__beleidshuur_ontwikkeling_corop"])),
        "verdeling_huurcategorieen": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__verdeling_huurcategorieen"])),
        "verdeling_huurcategorieen_ontwikkeling": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__verdeling_huurcategorieen_ontwikkeling"])),
        "lengte_mjob": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__lengte_mjob"])),
        "realisatie_huurstijging": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__realisatie_huurstijging"])),
        "realisatie_beleidshuur": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__realisatie_beleidshuur"])),
        "realisatie_mjob": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__realisatie_mjob"])),
        "detail_source": dg.AssetIn(key=dg.AssetKey(["models", "chart_blw_params__detail_source"])),
    },
)
def charts_blw_params(
    context: dg.AssetExecutionContext,
    beleidsonderhoud_ontwikkeling_corop: pd.DataFrame,
    beleidsbeheer_ontwikkeling_corop: pd.DataFrame,
    beleidshuur_ontwikkeling_corop: pd.DataFrame,
    verdeling_huurcategorieen: pd.DataFrame,
    verdeling_huurcategorieen_ontwikkeling: pd.DataFrame,
    lengte_mjob: pd.DataFrame,
    realisatie_huurstijging: pd.DataFrame,
    realisatie_beleidshuur: pd.DataFrame,
    realisatie_mjob: pd.DataFrame,
    detail_source: pd.DataFrame,
) -> None:
    """Generate beleidswaardeparameter charts."""
    save_charts(
        _build_blw_params_charts(
            beleidsonderhoud_ontwikkeling_corop=beleidsonderhoud_ontwikkeling_corop,
            beleidsbeheer_ontwikkeling_corop=beleidsbeheer_ontwikkeling_corop,
            beleidshuur_ontwikkeling_corop=beleidshuur_ontwikkeling_corop,
            verdeling_huurcategorieen=verdeling_huurcategorieen,
            verdeling_huurcategorieen_ontwikkeling=verdeling_huurcategorieen_ontwikkeling,
            lengte_mjob=lengte_mjob,
            realisatie_huurstijging=realisatie_huurstijging,
            realisatie_beleidshuur=realisatie_beleidshuur,
            realisatie_mjob=realisatie_mjob,
            detail_source=detail_source,
        ),
        context,
        save_fn=save_graph,
    )
