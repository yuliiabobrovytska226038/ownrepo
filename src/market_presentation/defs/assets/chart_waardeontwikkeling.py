"""charts_waardeontwikkeling — market-value development charts."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import of_template, save_charts, save_graph, set_bar_text_horizontal, set_line_marker_style, set_template_axis_format

of_template()


def _style_mutation_axis(fig: go.Figure, *, hide_x_title: bool = False) -> None:
    set_template_axis_format(fig)
    if hide_x_title:
        fig.update_xaxes(title_text="")


def _style_mutation_bar(fig: go.Figure, *, hide_x_title: bool = False) -> None:
    set_bar_text_horizontal(fig)
    _style_mutation_axis(fig, hide_x_title=hide_x_title)


def _build_waardeontwikkeling_charts(model_current: pd.DataFrame, type_current: pd.DataFrame, model_combined: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build waardeontwikkeling charts without writing files."""
    fig1 = px.bar(model_current, x="Waarderingsmodel", y="% Mutatie", color="Waarderingsmodel", text="Text", template="ortec_finance")
    _style_mutation_axis(fig1)

    fig2 = px.bar(model_combined, x="Jaar", y="% Mutatie", color="Waarderingsmodel", barmode="group", text="Text", template="ortec_finance")
    _style_mutation_bar(fig2, hide_x_title=True)
    fig2.layout.legend.orientation = "h"

    # Add cumulative growth line per Waarderingsmodel
    colorway = ["#0084CB", "#F58025", "#87BB40"]
    for i, model in enumerate(model_combined["Waarderingsmodel"].unique()):
        sub = model_combined[model_combined["Waarderingsmodel"] == model].sort_values("Jaar")
        cumul = (1 + sub["% Mutatie"]).cumprod() - 1
        # BOG/MOG/ZOG: place 2021 and 2022 labels below the line to avoid overlap
        if model == "BOG/MOG/ZOG":
            positions = ["bottom center" if j in (2021, 2022) else "top center" for j in sub["Jaar"]]
        else:
            positions = "top center"
        fig2.add_trace(
            go.Scatter(
                x=sub["Jaar"],
                y=cumul,
                mode="lines+markers+text",
                text=[f"{v:.1%}" for v in cumul],
                textposition=positions,
                name=f"{model} (cum.)",
                line={"dash": "dot", "width": 2, "color": colorway[i % len(colorway)]},
                marker={"size": 5, "color": colorway[i % len(colorway)]},
                showlegend=True,
            )
        )

    t_lijn = model_combined.sort_values(["Waarderingsmodel", "Jaar"], ascending=[False, True])
    fig3 = px.line(t_lijn, x="Jaar", y="% Mutatie", color="Waarderingsmodel", markers=True, template="ortec_finance")
    _style_mutation_axis(fig3, hide_x_title=True)
    set_line_marker_style(fig3)
    fig3.update_layout(legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99})

    fig4 = px.bar(type_current, x="Handboektype", y="% Mutatie", color="Waarderingsmodel", text="Text", template="ortec_finance")
    _style_mutation_bar(fig4)

    return [
        ("waardeontwikkeling/Marktwaardeontwikkeling per model bar", fig1, model_current),
        ("waardeontwikkeling/Marktwaardeontwikkeling per waarderingsmodel meerjarig", fig2, model_combined),
        ("waardeontwikkeling/Marktwaardeontwikkeling per waarderingsmodel meerjarig lijn", fig3, t_lijn),
        ("waardeontwikkeling/Marktwaardeontwikkeling per type bar", fig4, type_current),
    ]


@dg.asset(
    group_name="charts",
    name="charts_waardeontwikkeling",
    description="Marktwaardeontwikkeling charts by valuation model and handboek type.",
    ins={
        "model_current": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling__model_current"])),
        "type_current": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling__type_current"])),
        "model_combined": dg.AssetIn(key=dg.AssetKey(["models", "chart_waardeontwikkeling__model_combined"])),
    },
)
def charts_waardeontwikkeling(context: dg.AssetExecutionContext, model_current: pd.DataFrame, type_current: pd.DataFrame, model_combined: pd.DataFrame) -> None:
    """Generate marktwaardeontwikkeling charts."""
    save_charts(_build_waardeontwikkeling_charts(model_current, type_current, model_combined), context, save_fn=save_graph)
