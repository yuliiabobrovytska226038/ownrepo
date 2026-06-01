"""charts_va — value driver chart asset."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from market_presentation.defs.utils.graph_utils import of_template, save_charts, save_graph, set_template_axis_format

of_template()


def _g_value_drivers(t: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        t,
        x="Stap",
        y="Mutatie",
        color="Waarderingstype",
        text="Text Mutatie",
        barmode="group",
        template="ortec_finance",
    )
    fig.update_traces(textangle=0, textposition="outside")

    waard_list = t["Waarderingstype"].unique().tolist()
    if waard_list == ["Beleidswaarde"]:
        fig.update_traces(marker_color=pio.templates["ortec_finance"].layout.colorway[2])

    set_template_axis_format(fig, tickformat=".1%")
    fig.layout.xaxis.title.text = ""
    fig.layout.yaxis.title.text = ""
    fig.layout.legend = {"x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"}
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="show")

    # Add cumulative effect line
    colorway = pio.templates["ortec_finance"].layout.colorway
    for i, wtype in enumerate(waard_list):
        sub = t[t["Waarderingstype"] == wtype]
        # Position text below on first point (overlaps bar text), above on rest
        positions = ["bottom center"] + ["top center"] * (len(sub) - 1)
        # Match bar color: Beleidswaarde-only uses green, otherwise use colorway order
        if waard_list == ["Beleidswaarde"]:
            line_color = colorway[2]
        else:
            line_color = colorway[i % len(colorway)]
        fig.add_trace(
            go.Scatter(
                x=sub["Stap"],
                y=sub["Cumulatief"],
                mode="lines+markers+text",
                text=sub["Text Cumulatief"],
                textposition=positions,
                cliponaxis=False,
                name=f"Cumulatief ({wtype})" if len(waard_list) > 1 else "Cumulatief",
                line={"dash": "dot", "width": 2, "color": line_color},
                marker={"size": 6, "color": line_color},
                showlegend=True,
            )
        )

    return fig


def _g_value_drivers_bubble(t: pd.DataFrame) -> go.Figure:
    """Bubble chart showing cumulative effect of each driver."""
    t_plot = t.copy()
    t_plot["abs_mutatie"] = t_plot["Mutatie"].abs()
    fig = px.scatter(
        t_plot,
        x="Cumulatief",
        y="Stap",
        size="abs_mutatie",
        color="Waarderingstype",
        text="Text Mutatie",
        template="ortec_finance",
        size_max=50,
    )
    fig.update_traces(textposition="top center")
    set_template_axis_format(fig, tickformat=".1%", axis="x")
    fig.update_layout(xaxis_title="Cumulatief effect", yaxis_title="")
    return fig


def _build_va_charts(mw_drivers: pd.DataFrame, blw_drivers: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build verschillenanalyse value driver charts without writing files."""
    charts = [
        (path, _g_value_drivers(drivers), drivers)
        for path, drivers in (
            ("va/Marktwaardedrijvers", mw_drivers),
            ("va/Beleidswaardedrijvers", blw_drivers),
        )
    ]
    # Add bubble charts (item 3: waardedrijvers wolk - cumulatief effect)
    charts.append(("nieuwe figuren/Marktwaardedrijvers wolk cumulatief effect", _g_value_drivers_bubble(mw_drivers), mw_drivers))
    charts.append(("nieuwe figuren/Beleidswaardedrijvers wolk cumulatief effect", _g_value_drivers_bubble(blw_drivers), blw_drivers))
    return charts


@dg.asset(
    group_name="charts",
    name="charts_va",
    description="Verschillenanalyse value-driver charts for Marktwaarde and Beleidswaarde.",
    ins={
        "mw_drivers": dg.AssetIn(key=dg.AssetKey(["models", "chart_va__marketvalue_drivers"])),
        "blw_drivers": dg.AssetIn(key=dg.AssetKey(["models", "chart_va__policyvalue_drivers"])),
    },
)
def charts_va(context: dg.AssetExecutionContext, mw_drivers: pd.DataFrame, blw_drivers: pd.DataFrame) -> None:
    """Generate value-driver charts for Marktwaarde and Beleidswaarde."""
    save_charts(_build_va_charts(mw_drivers, blw_drivers), context, save_fn=save_graph)
