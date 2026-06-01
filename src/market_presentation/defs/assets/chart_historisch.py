"""charts_historisch — historical basis/full and uitponden charts."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import of_template, save_charts, save_graph

of_template()


def _style_historical_line(fig: go.Figure) -> None:
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_xaxes(tickmode="linear", tick0=2021, dtick=1)


def _build_historisch_charts(basis_full_verschil: pd.DataFrame, uitponden_percentage: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build historical charts without writing files."""
    fig1 = px.line(basis_full_verschil, x="Jaar", y="Verschil basis-full", markers=True, template="ortec_finance")
    _style_historical_line(fig1)
    fig1.update_traces(line_color="#146EB4")
    fig1.update_yaxes(tickformat=".1%", zeroline=True, zerolinewidth=2, zerolinecolor="#6d6865")

    fig2 = px.line(
        uitponden_percentage, x="Jaar", y="Percentage", color="Waarderingstype", markers=True,
        template="ortec_finance", color_discrete_map={"Basis": "#146EB4", "Full": "#f58025"},
    )
    _style_historical_line(fig2)
    fig2.update_yaxes(tickformat=".1%", range=[0, 1])
    fig2.update_layout(
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
        yaxis_title="Percentage uitponden",
    )

    return [
        ("historisch/Historisch basis-full verschil", fig1, basis_full_verschil),
        ("historisch/Historisch uitponden percentage", fig2, uitponden_percentage),
    ]


@dg.asset(
    group_name="charts",
    name="charts_historisch",
    description="Historical charts for basis/full difference and uitponden percentage.",
    ins={
        "basis_full_verschil": dg.AssetIn(key=dg.AssetKey(["models", "chart_historisch__basis_full_verschil"])),
        "uitponden_percentage": dg.AssetIn(key=dg.AssetKey(["models", "chart_historisch__uitponden_percentage"])),
    },
)
def charts_historisch(context: dg.AssetExecutionContext, basis_full_verschil: pd.DataFrame, uitponden_percentage: pd.DataFrame) -> None:
    """Generate historical basis/full and uitponden charts."""
    save_charts(_build_historisch_charts(basis_full_verschil, uitponden_percentage), context, save_fn=save_graph)
