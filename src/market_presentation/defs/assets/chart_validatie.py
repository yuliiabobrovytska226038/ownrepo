"""charts_validatie — validation charts comparing basis and full valuation portfolios."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import of_template, save_charts, save_graph, set_template_axis_format

of_template()


def _validatie_bar(plot: pd.DataFrame, *, y: str, text: str, colors: list[str], shapes: list[dict]) -> go.Figure:
    fig = px.bar(plot, x="Afwijking", y=y, text=text, barmode="group", template="ortec_finance")
    set_template_axis_format(fig, tickformat=".0%", hoverformat=".1%")
    fig.data[0].marker.color = colors
    fig.layout.shapes = shapes

    # Format text as integers (no .0) and place on top of bars
    text_values = [str(int(float(v))) if str(v).replace(".", "").replace("-", "").isdigit() else str(v) for v in fig.data[0].text]
    fig.data[0].text = text_values
    fig.update_traces(textposition="outside", textangle=0)
    return fig


def _build_validatie_charts(basis_full_portefeuilles: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build validation charts without writing files."""
    plot = basis_full_portefeuilles.reset_index()

    seq_col = px.colors.sequential.Blues[3::1][::-1] + px.colors.sequential.Blues[3::1]
    ref_lines = [
        {"x0": 3.5, "x1": 3.5, "color": "rgb(245, 128, 37)", "width": 1.5, "dash": "dash"},
        {"x0": 5.5, "x1": 5.5, "color": "rgb(245, 128, 37)", "width": 4, "dash": "solid"},
        {"x0": 7.5, "x1": 7.5, "color": "rgb(245, 128, 37)", "width": 1.5, "dash": "dash"},
    ]
    shapes = [{"type": "line", "xref": "x", "yref": "paper", "x0": r["x0"], "y0": 0, "x1": r["x1"], "y1": 1, "line": {"color": r["color"], "width": r["width"], "dash": r["dash"]}} for r in ref_lines]

    fig1 = _validatie_bar(plot, y="Procent", text="Aantal portefeuilles", colors=seq_col, shapes=shapes)
    fig1.update_layout(yaxis_title="Percentage deelportefeuilles")

    fig2 = _validatie_bar(plot, y="Naar woning gewogen", text="Naar woning gewogen text", colors=seq_col, shapes=shapes)

    return [
        ("validatie/Validatie deelportefeuilles", fig1, plot),
        ("validatie/Validatie deelportefeuilles naar woningen gewogen", fig2, plot),
    ]


@dg.asset(
    group_name="charts",
    name="charts_validatie",
    description="Validation charts for basis/full deelportefeuilles.",
    ins={
        "basis_full_portefeuilles": dg.AssetIn(key=dg.AssetKey(["models", "chart_validatie__basis_full_portefeuilles"])),
    },
)
def charts_validatie(context: dg.AssetExecutionContext, basis_full_portefeuilles: pd.DataFrame) -> None:
    """Generate validation charts for basis-full comparison buckets."""
    save_charts(_build_validatie_charts(basis_full_portefeuilles), context, save_fn=save_graph)
