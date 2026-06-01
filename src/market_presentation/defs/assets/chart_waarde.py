"""charts_analyse_waarde — pie chart and bar chart for market/policy value breakdown."""

import dagster as dg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from market_presentation.defs.utils.graph_utils import of_template, save_charts, save_graph, set_bar_text_horizontal, set_template_axis_format

of_template()

WAARDE_TYPES = ("Marktwaarde", "Beleidswaarde")


def _build_waarde_charts(waarde_per_model_perc: pd.DataFrame, gebruik_full_per_type: pd.DataFrame) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build waarde charts without writing files."""
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []

    for waarde in WAARDE_TYPES:
        t = waarde_per_model_perc.loc[waarde_per_model_perc["Waarde"] == waarde]
        fig = px.pie(t, values="% van totaal", names="Waarderingsmodel", template="ortec_finance")
        fig.update_traces(texttemplate="%{percent:.1%}")
        fig.layout.legend.orientation = "h"
        fig.update_layout(font={"size": 48})
        fig.layout.legend.font.size = 40
        charts.append((f"waarde/{waarde.lower()} per model percentage pie", fig, t))

    fig = px.bar(gebruik_full_per_type, x="Handboektype", y="% Full", color="Waarderingsmodel", text="Text", barmode="group", template="ortec_finance")
    set_template_axis_format(fig)
    set_bar_text_horizontal(fig)
    charts.append(("waarde/gebruik full per type bar", fig, gebruik_full_per_type))
    return charts


@dg.asset(
    group_name="charts",
    name="charts_analyse_waarde",
    description="Waarde per model percentage pie + Gebruik full per type bar (charts 1-2).",
    ins={
        "waarde_per_model_perc": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde__per_model_perc"])),
        "gebruik_full_per_type": dg.AssetIn(key=dg.AssetKey(["models", "chart_waarde__gebruik_full_per_type"])),
    },
)
def charts_analyse_waarde(context: dg.AssetExecutionContext, waarde_per_model_perc: pd.DataFrame, gebruik_full_per_type: pd.DataFrame) -> None:
    """Generate waarde-per-model pie charts (Marktwaarde + Beleidswaarde) and gebruik-full bar chart."""
    save_charts(_build_waarde_charts(waarde_per_model_perc, gebruik_full_per_type), context, save_fn=save_graph)
