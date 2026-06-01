"""charts_vhg_ontwikkeling — vrijheidsgraad (VHG) development chart assets."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.assets.chart_constants import (
    BELEIDSWAARDE_VRIJHEIDSGRADEN,
    CHART_YEAR,
    CHART_YEAR_M1,
    MARKTWAARDE_VRIJHEIDSGRADEN,
)
from market_presentation.defs.utils.graph_utils import chart_appender, of_template, save_charts, save_graph, set_template_axis_format

of_template()


# ---------------------------------------------------------------------------
# Rendering functions (Plotly only — no data transformation)
# ---------------------------------------------------------------------------


def _g_indexatiegebied_value(t: pd.DataFrame, vhg_dict: dict) -> px.bar:
    vv = vhg_dict["value_vhg"]
    if t[vv].max() < 0.1:
        t["Text"] = t[vv].map(lambda n: f"{n:,.2%}".replace(".", ",") if pd.notna(n) else "")
    else:
        t["Text"] = t[vv].map(lambda n: f"{n:,.1%}".replace(".", ",") if pd.notna(n) else "")

    plot = t.reset_index()
    fig = px.bar(plot, x="Indexatiegebied", y=vv, text="Text", color="Waarderingstype", barmode="group", template="ortec_finance")
    set_template_axis_format(fig)
    fig.layout.showlegend = plot["Waarderingstype"].nunique() > 1
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_traces(textposition="outside", textfont_size=10)
    return fig


def _g_corporatie_value_boxplot(t: pd.DataFrame, vhg_dict: dict, b_calc_diff: bool) -> px.box:
    t_r = t.reset_index()
    color_map = {"Full": "#F58025", "Full (handboek)": "#87BB40"}
    fig = px.box(t_r, x=vhg_dict["value_vhg"], color="Waarderingstype", hover_name="Corporatie", template="ortec_finance", color_discrete_map=color_map)
    title = ("Aanpassing " + vhg_dict["name"]).capitalize() if b_calc_diff else vhg_dict["name"]
    fig.update_layout(xaxis_title=title)
    if t_r[vhg_dict["value_vhg"]].max() < 1:
        set_template_axis_format(fig, axis="x")
    if t_r["Waarderingstype"].nunique() == 1:
        fig.layout.showlegend = False
        fig.update_traces(marker_color="#F58025")
    return fig


def _g_vhg_corporatie(t: pd.DataFrame, vhg_dict: dict, b_diff: bool) -> px.box:
    vv = vhg_dict["value_vhg"]
    vb = vhg_dict.get("value_basic", "")
    if b_diff:
        vrijheidsgraad = "Verschil"
        title = f"Aanpassing {vhg_dict['name']}".capitalize()
    else:
        vrijheidsgraad = vv
        title = vhg_dict["name"]

    if b_diff:
        t_plot = t[t["Vrijheidsgraad"] == vrijheidsgraad]
    else:
        # Include only Full and Full (handboek) for niveau charts
        t_plot = t[(t["Vrijheidsgraad"].isin([vv, vb])) & (t["Waarderingstype"] != "Basis")]

    fig = px.box(t_plot, x="Waarde", color="Waarderingstype", hover_name="Corporatie", template="ortec_finance")
    fig.update_xaxes(title_text=title)
    max_val = t_plot["Waarde"].max()
    if pd.notna(max_val) and max_val < 1:
        set_template_axis_format(fig, axis="x")
    return fig


def _g_corporatie_wozvalue_boxplot(woz_corporatie: pd.DataFrame) -> px.box:
    fig = px.box(woz_corporatie, x="Stijging WOZ-waarde", hover_name="Corporatie", template="ortec_finance")
    fig.layout.template.layout.xaxis.tickformat = "0%"
    return fig


def _g_disconteringsvoet_historie(df: pd.DataFrame) -> px.line:
    fig = px.line(
        df,
        x="Jaar",
        y="Percentage",
        line_group="Disconteringsvoet",
        color="Disconteringsvoet",
        template="ortec_finance",
    )
    fig.layout.showlegend = False
    fig.update_traces(line={"width": 4})
    fig.update_xaxes(range=[df["Jaar"].min(), df["Jaar"].max() + 0.7])
    annotations = [
        {
            "xref": "paper",
            "x": 0.9,
            "y": index,
            "xanchor": "left",
            "yanchor": "middle",
            "text": dv,
            "font": {"family": "Fira Sans", "size": 26},
            "showarrow": False,
        }
        for index, dv in zip(
            df.loc[df["Jaar"] == df["Jaar"].max(), "Percentage"],
            df.loc[df["Jaar"] == CHART_YEAR, "Disconteringsvoet"],
            strict=False,
        )
    ]
    fig.update_layout(annotations=annotations)
    set_template_axis_format(fig)
    return fig


def _g_gebruik_vhg(gebruik_df: pd.DataFrame) -> px.bar:
    fig = px.bar(
        gebruik_df,
        x="Vrijheidsgraad",
        y="Totaal",
        text="Text",
        template="ortec_finance",
    )
    fig.update_traces(textposition="outside", insidetextanchor="middle", textangle=0)
    set_template_axis_format(fig, tickformat=".0%", hoverformat=".1%")
    return fig


def _g_blw_yoy_boxplot(t: pd.DataFrame, vhg_dict: dict) -> px.box:
    vv = vhg_dict["value_vhg"]
    fig = px.box(t, y="Jaar", x=vv, color="Jaar", hover_name="Corporatie", template="ortec_finance")
    if t[vv].max() < 1:
        fig.update_xaxes(tickformat=",.1%", hoverformat=".1%")
    fig.layout.showlegend = False
    return fig


def _g_blw_corporatie_boxplot(t: pd.DataFrame, vhg_dict: dict) -> px.box:
    vv = vhg_dict["value_vhg"]
    fig = px.box(t, x=vv, hover_name="Corporatie", color="Jaar", template="ortec_finance")
    fig.update_xaxes(title_text=vv.replace("_", " "))
    if t[vv].max() < 1:
        set_template_axis_format(fig, axis="x", hoverformat=".1%")
    return fig


def _g_oh_basis_full_boxplot(t: pd.DataFrame, vhg_dict: dict) -> px.box:
    fig = px.box(t, x=vhg_dict["name"], color="Waarderingstype", hover_name="Corporatie", template="ortec_finance")
    return fig


def _indexatiegebied_chart(path: str, source: pd.DataFrame, vhg_key: str, vhg_dict: dict) -> tuple[str, go.Figure, pd.DataFrame]:
    vv = vhg_dict["value_vhg"]
    table = source[source["vhg_key"] == vhg_key].rename(columns={"value": vv}).copy()
    table = table.sort_values(["Waarderingstype", "Indexatiegebied"]).set_index(["Indexatiegebied", "Waarderingstype"])
    fig = _g_indexatiegebied_value(table, vhg_dict)
    return path, fig, table


def _corporatie_value_charts(prefix: str, source: pd.DataFrame, vhg_key: str, vhg_dict: dict) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    vv = vhg_dict["value_vhg"]
    table = source[source["vhg_key"] == vhg_key].rename(columns={"value": vv}).copy()

    niveau = table[table["calc_type"] == "niveau"].set_index(["Corporatie", "Waarderingstype"])
    fig_niveau = _g_corporatie_value_boxplot(niveau, vhg_dict, b_calc_diff=False)

    aanpassing = table[table["calc_type"] == "aanpassing"].set_index(["Corporatie", "Waarderingstype"])
    fig_aanpassing = _g_corporatie_value_boxplot(aanpassing, vhg_dict, b_calc_diff=True)

    return [
        (f"{prefix}/Boxplot niveau {vhg_dict['name']} per corporatie", fig_niveau, niveau),
        (f"{prefix}/Boxplot aanpassing {vhg_dict['name']} per corporatie", fig_aanpassing, aanpassing),
    ]


# ---------------------------------------------------------------------------
# Build functions — read pre-computed SQL data, filter by vhg_key, render
# ---------------------------------------------------------------------------


def _build_vhg_ontwikkeling_marktwaarde_charts(
    *,
    woz_corporatie: pd.DataFrame,
    disconteringsvoet_historie: pd.DataFrame,
    gebruik_vhg: pd.DataFrame,
    mw_indexatiegebied: pd.DataFrame,
    mw_corporatie: pd.DataFrame,
    mw_vhg_corporatie: pd.DataFrame,
    mw_oh_basis_full: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build Marktwaarde VHG charts from pre-computed SQL data."""
    woz_fig = _g_corporatie_wozvalue_boxplot(woz_corporatie)
    history_fig = _g_disconteringsvoet_historie(disconteringsvoet_historie)
    gebruik_fig = _g_gebruik_vhg(gebruik_vhg)

    charts: list[tuple[str, go.Figure, pd.DataFrame]] = [
        ("vhg_ontwikkeling/Boxplot ontwikkeling WOZ-waarde per corporatie", woz_fig, woz_corporatie),
        ("vhg_ontwikkeling/Historie disconteringsvoet", history_fig, disconteringsvoet_historie),
        ("vhg_ontwikkeling/Gebruik vrijheidsgraden full waarderingen", gebruik_fig, gebruik_vhg),
    ]
    add = chart_appender(charts, "vhg_ontwikkeling")

    for vhg_key, vhg_dict in MARKTWAARDE_VRIJHEIDSGRADEN.items():
        vv = vhg_dict["value_vhg"]

        # VHG corporatie boxplots (Basis/Full/Full handboek)
        t_vhg = mw_vhg_corporatie[mw_vhg_corporatie["vhg_key"] == vhg_key].copy()
        t_vhg["Waarderingstype"] = pd.Categorical(t_vhg["Waarderingstype"], ["Basis", "Full", "Full (handboek)"])
        t_vhg = t_vhg.sort_values("Waarderingstype")

        fig_niveau = _g_vhg_corporatie(t_vhg, vhg_dict, b_diff=False)
        add(f"Boxplots niveau {vhg_dict['name']} per corporatie", fig_niveau, t_vhg)

        fig_aanp = _g_vhg_corporatie(t_vhg, vhg_dict, b_diff=True)
        add(f"Boxplots aanpassing {vhg_dict['name']} per corporatie", fig_aanp, t_vhg)

        charts.append(_indexatiegebied_chart(f"vhg_ontwikkeling/Aanpassing {vhg_dict['name']} per indexatiegebied", mw_indexatiegebied, vhg_key, vhg_dict))
        charts.extend(_corporatie_value_charts("vhg_ontwikkeling", mw_corporatie, vhg_key, vhg_dict))

        # Special OH uitponden basis/full comparison
        if vv == "OH uitponden":
            t_oh = mw_oh_basis_full.rename(columns={"value": vhg_dict["name"]}).copy()
            t_oh = t_oh.sort_values("Waarderingstype")
            fig_oh = _g_oh_basis_full_boxplot(t_oh, vhg_dict)
            add("Onderhoud uitponden vergelijking basis full boxplot", fig_oh, t_oh)

    return charts


def _build_vhg_ontwikkeling_beleidswaarde_charts(
    *,
    woz_corporatie: pd.DataFrame,
    blw_indexatiegebied: pd.DataFrame,
    blw_corporatie: pd.DataFrame,
    blw_yoy: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build Beleidswaarde VHG charts from pre-computed SQL data."""
    woz_fig = _g_corporatie_wozvalue_boxplot(woz_corporatie)

    charts: list[tuple[str, go.Figure, pd.DataFrame]] = [("vhg_ontwikkeling_blw/Boxplot ontwikkeling WOZ-waarde per corporatie", woz_fig, woz_corporatie)]
    add = chart_appender(charts, "vhg_ontwikkeling_blw")

    for vhg_key, vhg_dict in BELEIDSWAARDE_VRIJHEIDSGRADEN.items():
        vv = vhg_dict["value_vhg"]

        charts.append(_indexatiegebied_chart(f"vhg_ontwikkeling_blw/Aanpassing {vhg_dict['name']} per indexatiegebied", blw_indexatiegebied, vhg_key, vhg_dict))
        charts.extend(_corporatie_value_charts("vhg_ontwikkeling_blw", blw_corporatie, vhg_key, vhg_dict))

        # Current / Prior / YoY boxplots
        t_yoy = blw_yoy[blw_yoy["vhg_key"] == vhg_key].rename(columns={"value": vv}).copy()
        t_yoy = t_yoy.sort_values("Jaar")

        t_current = t_yoy[t_yoy["Jaar"] == str(CHART_YEAR)].copy()
        t_current["Jaar"] = CHART_YEAR
        fig_current = _g_blw_corporatie_boxplot(t_current, vhg_dict)
        add(f"Boxplot {vhg_dict['name']} per corporatie", fig_current, t_current)

        t_previous = t_yoy[t_yoy["Jaar"] == str(CHART_YEAR_M1)].copy()
        t_previous["Jaar"] = CHART_YEAR_M1
        fig_previous = _g_blw_corporatie_boxplot(t_previous, vhg_dict)
        add(f"Boxplot {vhg_dict['name']} {CHART_YEAR_M1} per corporatie", fig_previous, t_previous)

        fig_yoy = _g_blw_yoy_boxplot(t_yoy, vhg_dict)
        add(f"Boxplot {vhg_dict['name']} YoY per corporatie", fig_yoy, t_yoy)

    return charts


# ---------------------------------------------------------------------------
# Dagster assets
# ---------------------------------------------------------------------------


@dg.asset(
    group_name="charts",
    name="charts_vhg_ontwikkeling_marktwaarde",
    description="Marktwaarde VHG charts: gebruik VHG bar + 5 charts per 10 vrijheidsgraden + WOZ and history charts.",
    ins={
        "woz_corporatie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__woz_corporatie"])),
        "disconteringsvoet_historie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__disconteringsvoet_historie"])),
        "gebruik_vhg": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__gebruik_vhg_parity"])),
        "mw_indexatiegebied": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__mw_indexatiegebied"])),
        "mw_corporatie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__mw_corporatie"])),
        "mw_vhg_corporatie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__mw_vhg_corporatie"])),
        "mw_oh_basis_full": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__mw_oh_basis_full"])),
    },
)
def charts_vhg_ontwikkeling_marktwaarde(
    context: dg.AssetExecutionContext,
    woz_corporatie: pd.DataFrame,
    disconteringsvoet_historie: pd.DataFrame,
    gebruik_vhg: pd.DataFrame,
    mw_indexatiegebied: pd.DataFrame,
    mw_corporatie: pd.DataFrame,
    mw_vhg_corporatie: pd.DataFrame,
    mw_oh_basis_full: pd.DataFrame,
) -> None:
    """Generate 53+ charts for Marktwaarde VHG development: gebruik, per-VHG boxplots and bars, WOZ."""
    save_charts(
        _build_vhg_ontwikkeling_marktwaarde_charts(
            woz_corporatie=woz_corporatie,
            disconteringsvoet_historie=disconteringsvoet_historie,
            gebruik_vhg=gebruik_vhg,
            mw_indexatiegebied=mw_indexatiegebied,
            mw_corporatie=mw_corporatie,
            mw_vhg_corporatie=mw_vhg_corporatie,
            mw_oh_basis_full=mw_oh_basis_full,
        ),
        context,
        save_fn=save_graph,
    )


@dg.asset(
    group_name="charts",
    name="charts_vhg_ontwikkeling_beleidswaarde",
    description="Beleidswaarde VHG charts: activated indexatiegebied/current/prior/YoY boxplots + WOZ boxplot.",
    ins={
        "woz_corporatie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__woz_corporatie"])),
        "blw_indexatiegebied": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__blw_indexatiegebied"])),
        "blw_corporatie": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__blw_corporatie"])),
        "blw_yoy": dg.AssetIn(key=dg.AssetKey(["models", "chart_vhg__blw_yoy"])),
    },
)
def charts_vhg_ontwikkeling_beleidswaarde(
    context: dg.AssetExecutionContext,
    woz_corporatie: pd.DataFrame,
    blw_indexatiegebied: pd.DataFrame,
    blw_corporatie: pd.DataFrame,
    blw_yoy: pd.DataFrame,
) -> None:
    """Generate activated Beleidswaarde VHG charts."""
    save_charts(
        _build_vhg_ontwikkeling_beleidswaarde_charts(
            woz_corporatie=woz_corporatie,
            blw_indexatiegebied=blw_indexatiegebied,
            blw_corporatie=blw_corporatie,
            blw_yoy=blw_yoy,
        ),
        context,
        save_fn=save_graph,
    )
