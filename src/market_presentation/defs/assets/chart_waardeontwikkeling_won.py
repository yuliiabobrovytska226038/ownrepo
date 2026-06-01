"""charts_waardeontwikkeling_won — woningen value-development chart asset."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.assets.chart_constants import CHART_YEAR, CHART_YEAR_M1
from market_presentation.defs.utils.graph_utils import chart_appender, multi_layer_map, of_template, save_charts, save_graph

of_template()


def _style_percent_bar(fig: go.Figure) -> None:
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat="0%", hoverformat=".2%")


def _g_waardeontwikkeling_t0(t: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        t.reset_index(),
        x="Waarderingstype",
        y="% Mutatie",
        text="Text",
        color="Waarderingstype",
        template="ortec_finance",
    )
    _style_percent_bar(fig)
    return fig


def _g_waardeontwikkeling_bs_fl(t: pd.DataFrame) -> go.Figure:
    plot = t.reset_index(drop=False).copy()
    plot["Waarde - Waarderingstype"] = plot["Waarde"].astype(str) + " - " + plot["Waarderingstype"].astype(str)
    fig = px.bar(
        plot,
        x="Waarde - Waarderingstype",
        y="% Mutatie",
        text="Text",
        color="Waarde - Waarderingstype",
        template="ortec_finance",
    )
    _style_percent_bar(fig)
    return fig


def _g_waardeontwikkeling_meerjarig(t: pd.DataFrame, start_year: int) -> go.Figure:
    base_col = f"Indexcijfer_base_{start_year}"
    plot = t.loc[t["Jaar"] >= start_year].copy()
    plot = plot.rename(columns={base_col: "Index"})
    fig = px.line(
        plot,
        x="Jaar",
        y="Index",
        line_group="Waarderingstype",
        color="Waarderingstype",
        template="ortec_finance",
    )
    fig.update_layout(xaxis={"tickvals": list(range(start_year, CHART_YEAR + 1))}, showlegend=False)
    fig.update_traces(line={"width": 4})
    fig.update_xaxes(range=[plot["Jaar"].min(), plot["Jaar"].max() + 0.7])
    fig.update_yaxes(title_text="Index")
    annotations = [
        {
            "xref": "paper",
            "x": 0.91,
            "y": index,
            "xanchor": "left",
            "yanchor": "top" if waardtype == "Beleggers (MSCI)" else "bottom",
            "text": waardtype,
            "font": {"family": "Fira Sans", "size": 14},
            "showarrow": False,
        }
        for index, waardtype in plot.loc[plot["Jaar"] == CHART_YEAR, ["Index", "Waarderingstype"]].itertuples(index=False, name=None)
    ]
    fig.update_layout(annotations=annotations)
    return fig


def _g_waardeontwikkeling_meerjarig_beleidswaardes(t: pd.DataFrame) -> go.Figure:
    fig = _g_waardeontwikkeling_meerjarig(t, 2018)
    if fig.data:
        fig.data[0].line.color = "#87BB40"
    beleidswaarde_nieuw_index = 5306459.09 / 3975011.91 * 114.9017
    beleidswaarde_2024 = 155.2502
    # Get the actual 2025 value from the data to connect the dashed line smoothly
    blw_2025 = t.loc[t["Jaar"] == CHART_YEAR, "Indexcijfer_base_2018"]
    beleidswaarde_2025 = blw_2025.iloc[0] if len(blw_2025) > 0 else beleidswaarde_2024
    fig.add_trace(
        go.Scatter(
            x=[2022, 2023, 2024, 2025],
            y=[132.4564341, beleidswaarde_nieuw_index, beleidswaarde_2024, beleidswaarde_2025],
            mode="lines",
            line={"color": "#87BB40", "dash": "dash", "width": 4},
        )
    )
    fig.layout.annotations = []
    fig.add_annotation(
        xref="paper",
        x=0.91,
        y=beleidswaarde_2025,
        text="Nieuwe beleidswaarde",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"family": "Fira Sans", "size": 14, "color": "black"},
    )
    return fig


def _g_waardeontwikkeling_spreiding(t: pd.DataFrame, waarde: str) -> go.Figure:
    fig = px.box(t.reset_index(), x="% Mutatie", hover_name="Corporatie", template="ortec_finance")
    fig.update_xaxes(title_text=f"{waarde}ontwikkeling")
    if t["% Mutatie"].max() < 2:
        fig.update_xaxes(tickformat="0%", hoverformat=".1%")
    # Add mean annotation
    mean_val = t["% Mutatie"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(x=mean_val, y=1.05, yref="paper", text=f"Gem: {mean_val:.1%}", showarrow=False, font={"size": 18, "color": "red"})
    return fig


def _build_waardeontwikkeling_won_charts(
    *,
    t0_marktwaarde: pd.DataFrame,
    t0_beleidswaarde: pd.DataFrame,
    bs_fl: pd.DataFrame,
    meerjarig: pd.DataFrame,
    coropplus: pd.DataFrame,
    spreiding: pd.DataFrame,
    corop_medians: pd.DataFrame,
    corop_ratio: pd.DataFrame,
    corop_ontwikkeling: pd.DataFrame,
    efg_corop: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build waardeontwikkeling-woningen charts without writing files."""
    p = "waardeontwikkeling_won"
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []
    loc = "COROP-plusgebieden Code"
    add = chart_appender(charts, p)

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
        if loc not in agg_df.columns:
            return _detail_agg
        codes = set(agg_df[loc].dropna().unique())
        return _detail_agg.loc[_detail_agg[loc].isin(codes)].copy()

    add(f"Marktwaardeontwikkeling {CHART_YEAR_M1} - {CHART_YEAR}", _g_waardeontwikkeling_t0(t0_marktwaarde), t0_marktwaarde)
    add(f"Beleidswaardeontwikkeling {CHART_YEAR_M1} - {CHART_YEAR}", _g_waardeontwikkeling_t0(t0_beleidswaarde), t0_beleidswaarde)
    add("Waardeontwikkeling per waarderingstype", _g_waardeontwikkeling_bs_fl(bs_fl), bs_fl)

    add("Waardeontwikkeling meerjarig", _g_waardeontwikkeling_meerjarig(meerjarig, 2018), meerjarig)

    t_excl = meerjarig.loc[meerjarig["Waarderingstype"] != "Beleidswaarde"].copy()
    add("Waardeontwikkeling meerjarig excl. beleidswaarde", _g_waardeontwikkeling_meerjarig(t_excl, 2016), t_excl)

    t_blw_mj = meerjarig.loc[meerjarig["Waarderingstype"] == "Beleidswaarde"].copy()
    add("Waardeontwikkeling meerjarig beleidswaarde oud", _g_waardeontwikkeling_meerjarig(t_blw_mj, 2018), t_blw_mj)
    add("Waardeontwikkeling meerjarig beleidswaardes", _g_waardeontwikkeling_meerjarig_beleidswaardes(t_blw_mj), t_blw_mj)

    t_mw_corop = coropplus.loc[(coropplus["Waarde"] == "Marktwaarde") & (coropplus["Classificatie"] == "Totaal")].copy()
    add("Marktwaardeontwikkeling COROP-plusgebieden", multi_layer_map(t_mw_corop, location_col=loc, value_col="% Mutatie"), t_mw_corop, _detail_for(t_mw_corop))

    t_blw_corop = coropplus.loc[(coropplus["Waarde"] == "Beleidswaarde") & (coropplus["Classificatie"] == "Totaal")].copy()
    add("Beleidswaardeontwikkeling COROP-plusgebieden", multi_layer_map(t_blw_corop, location_col=loc, value_col="% Mutatie"), t_blw_corop, _detail_for(t_blw_corop))

    # Absolute BLW COROP map — convert to thousands for hover display
    t_blw_abs = t_blw_corop.copy()
    t_blw_abs["€ Mutatie (k)"] = t_blw_abs["€ Mutatie"] / 1000
    add(
        "Beleidswaardeontwikkeling COROP-plusgebieden absoluut",
        multi_layer_map(t_blw_abs, location_col=loc, value_col="€ Mutatie (k)", use_percent=False, hoverformat="%{z:,.1f}k<extra></extra>"),
        t_blw_abs,
        _detail_for(t_blw_abs),
    )

    # COROP maps per Classificatie (DAEB / Niet-DAEB / Totaal)
    _classificaties = ["DAEB", "Niet-DAEB", "Totaal"]

    for classificatie in _classificaties:
        suffix = f" ({classificatie})"

        # Marktwaarde- and beleidswaardeontwikkeling per Classificatie
        t_mw_c = coropplus.loc[(coropplus["Waarde"] == "Marktwaarde") & (coropplus["Classificatie"] == classificatie)].copy()
        if not t_mw_c.empty:
            add(f"Marktwaardeontwikkeling COROP-plusgebieden{suffix}", multi_layer_map(t_mw_c, location_col=loc, value_col="% Mutatie"), t_mw_c, _detail_for(t_mw_c))

        t_blw_c = coropplus.loc[(coropplus["Waarde"] == "Beleidswaarde") & (coropplus["Classificatie"] == classificatie)].copy()
        if not t_blw_c.empty:
            add(f"Beleidswaardeontwikkeling COROP-plusgebieden{suffix}", multi_layer_map(t_blw_c, location_col=loc, value_col="% Mutatie"), t_blw_c, _detail_for(t_blw_c))

    add("Boxplot Beleidswaardeontwikkeling", _g_waardeontwikkeling_spreiding(spreiding, "Beleidswaarde"), spreiding)

    # COROP median maps — all pre-computed in SQL (per Classificatie)
    # BLW/MW: convert to thousands (rounded to whole k)
    # OH/beheer: round to whole numbers
    median_maps = [
        ("median gehanteerd beleidsbeheer COROP-plusgebieden", "median_beleidsbeheer", False, "int"),
        ("median gehanteerd beleidshuur COROP-plusgebieden", "median_beleidshuur", False, None),
        ("median beleidswaarde per COROP-plusgebied", "median_beleidswaarde", False, "k"),
        ("median marktwaarde per COROP-plusgebied", "median_marktwaarde", False, "k"),
        ("median leegwaarde per COROP-plusgebied", "median_LW", False, None),
        ("median beleidsonderhoud per COROP-plusgebied", "median_beleidsonderhoud", False, "int"),
        ("median gehanteerde disconteringsvoet per COROP-plusgebied", "median_DV", True, None),
    ]
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        for label, value_col, use_percent, rounding in median_maps:
            df_map = corop_medians.loc[corop_medians["Classificatie"] == classificatie].dropna(subset=[value_col]).copy()
            if df_map.empty:
                continue
            hover = None
            if rounding == "k":
                df_map[value_col] = (df_map[value_col] / 1000).round(0)
                hover = "%{z:,.0f}k<extra></extra>"
            elif rounding == "int":
                df_map[value_col] = df_map[value_col].round(0)
            fig = multi_layer_map(df_map, location_col=loc, value_col=value_col, use_percent=use_percent, hoverformat=hover)
            if rounding == "k":
                for trace in fig.data:
                    if hasattr(trace, "colorbar") and trace.colorbar is not None:
                        trace.colorbar.ticksuffix = "k"
            add(f"{label}{suffix}", fig, df_map, _detail_for(df_map))

    # Ratio map — pre-computed in SQL (per Classificatie)
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        df_ratio = corop_ratio.loc[corop_ratio["Classificatie"] == classificatie].copy()
        if not df_ratio.empty:
            fig_ratio = multi_layer_map(df_ratio, location_col=loc, value_col="median_ratio", use_percent=True)
            add(f"median verhouding Marktwaarde Beleidswaarde per COROP-plusgebied{suffix}", fig_ratio, df_ratio, _detail_for(df_ratio))

    # DV delta map — pre-computed in SQL (per Classificatie)
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        df_dv_map = corop_ontwikkeling.loc[corop_ontwikkeling["Classificatie"] == classificatie].dropna(subset=["median_dv_delta"]).copy()
        if not df_dv_map.empty:
            fig_dv = multi_layer_map(df_dv_map, location_col=loc, value_col="median_dv_delta", use_percent=True, overrule_colorscale="reds_r")
            for trace in fig_dv.data:
                if hasattr(trace, "colorbar") and trace.colorbar is not None:
                    trace.colorbar.tickformat = ".1%"
            add(f"median DV ontwikkelings beleidswaarde per COROP-plusgebied{suffix}", fig_dv, df_dv_map, _detail_for(df_dv_map))

    # Beleidsonderhoud development map — pre-computed in SQL (per Classificatie)
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        df_onderhoud_map = corop_ontwikkeling.loc[corop_ontwikkeling["Classificatie"] == classificatie].dropna(subset=["ontwikkeling_beleidsonderhoud"]).copy()
        if not df_onderhoud_map.empty:
            fig_onderhoud = multi_layer_map(
                df_onderhoud_map,
                location_col=loc,
                value_col="ontwikkeling_beleidsonderhoud",
                use_percent=True,
            )
            add(f"median Ontwikkeling beleidsonderhoud per COROP-plusgebied{suffix}", fig_onderhoud, df_onderhoud_map, _detail_for(df_onderhoud_map))

    # EFG map (per Classificatie)
    for classificatie in _classificaties:
        suffix = f" ({classificatie})"
        df_efg = efg_corop.loc[efg_corop["Classificatie"] == classificatie].copy()
        if not df_efg.empty:
            fig_efg = multi_layer_map(df_efg, location_col=loc, value_col="perc_EFG", use_percent=True)
            add(f"Percentage EFG Labels per COROP-plusgebied{suffix}", fig_efg, df_efg, _detail_for(df_efg))

    return charts


@dg.asset(
    group_name="charts",
    name="charts_waardeontwikkeling_won",
    description="Value-development charts for woningen: activated legacy bars/lines/boxplots plus COROP-plus maps.",
    ins={
        "t0_marktwaarde": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__t0_marktwaarde"])),
        "t0_beleidswaarde": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__t0_beleidswaarde"])),
        "bs_fl": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__bs_fl"])),
        "meerjarig": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__meerjarig"])),
        "coropplus": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__coropplus"])),
        "spreiding": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__spreiding"])),
        "corop_medians": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__corop_medians"])),
        "corop_ratio": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__corop_ratio"])),
        "corop_ontwikkeling": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__corop_ontwikkeling"])),
        "efg_corop": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__efg_corop"])),
        "basis_source": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling_won__basis_source"])),
    },
)
def charts_waardeontwikkeling_won(
    context: dg.AssetExecutionContext,
    t0_marktwaarde: pd.DataFrame,
    t0_beleidswaarde: pd.DataFrame,
    bs_fl: pd.DataFrame,
    meerjarig: pd.DataFrame,
    coropplus: pd.DataFrame,
    spreiding: pd.DataFrame,
    corop_medians: pd.DataFrame,
    corop_ratio: pd.DataFrame,
    corop_ontwikkeling: pd.DataFrame,
    efg_corop: pd.DataFrame,
    basis_source: pd.DataFrame,
) -> None:
    """Generate value-development charts for woningen, including the reactivated legacy outputs."""
    save_charts(
        _build_waardeontwikkeling_won_charts(
            t0_marktwaarde=t0_marktwaarde,
            t0_beleidswaarde=t0_beleidswaarde,
            bs_fl=bs_fl,
            meerjarig=meerjarig,
            coropplus=coropplus,
            spreiding=spreiding,
            corop_medians=corop_medians,
            corop_ratio=corop_ratio,
            corop_ontwikkeling=corop_ontwikkeling,
            efg_corop=efg_corop,
            basis_source=basis_source,
        ),
        context,
        save_fn=save_graph,
    )
