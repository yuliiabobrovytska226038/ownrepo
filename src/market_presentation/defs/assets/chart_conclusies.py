"""charts_conclusies — Visual conclusion dashboards for Marktwaarde and Beleidswaarde."""

import dagster as dg
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market_presentation.defs.utils.graph_utils import chart_appender, of_template, save_charts, save_graph

of_template()

# Ortec Finance color palette
OF_BLUE = "#146EB4"
OF_GREEN = "#2E8B57"
OF_RED = "#C0392B"
OF_ORANGE = "#E67E22"
OF_GRAY = "#7F8C8D"
OF_LIGHT_BLUE = "#5DADE2"


def _conclusion_marktwaarde(df: pd.DataFrame) -> go.Figure:
    """Create Marktwaarde conclusion dashboard: KPI tiles + driver waterfall."""
    # Get summary metrics per type
    basis = df[df["Waarderingstype"] == "Basis"].iloc[0]
    full = df[df["Waarderingstype"] == "Full"].iloc[0]

    # Build subplot grid: Row 1 = KPIs (4 indicators), Row 2 = drivers bar
    fig = make_subplots(
        rows=2,
        cols=4,
        row_heights=[0.35, 0.65],
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar", "colspan": 4}, None, None, None],
        ],
        vertical_spacing=0.15,
    )

    # --- Row 1: KPI indicator tiles ---
    # 1. Totale waardestijging Basis
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=basis["pct_change_median"] * 100,
            number={"suffix": "%", "font": {"size": 40, "color": OF_BLUE}},
            delta={"reference": 0, "relative": False, "suffix": "%", "increasing": {"color": OF_GREEN}},
            title={"text": "<b>Basis</b><br>Waardestijging", "font": {"size": 14}},
        ),
        row=1,
        col=1,
    )

    # 2. Totale waardestijging Full
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=full["pct_change_median"] * 100,
            number={"suffix": "%", "font": {"size": 40, "color": OF_LIGHT_BLUE}},
            delta={"reference": 0, "relative": False, "suffix": "%", "increasing": {"color": OF_GREEN}},
            title={"text": "<b>Full</b><br>Waardestijging", "font": {"size": 14}},
        ),
        row=1,
        col=2,
    )

    # 3. Median DV
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=basis["median_dv"] * 100,
            number={"suffix": "%", "font": {"size": 36, "color": OF_ORANGE}},
            title={"text": "<b>Mediaan DV</b><br>Basis", "font": {"size": 14}},
        ),
        row=1,
        col=3,
    )

    # 4. Median EY
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=basis["median_ey"] * 100,
            number={"suffix": "%", "font": {"size": 36, "color": OF_ORANGE}},
            title={"text": "<b>Mediaan EY</b><br>Basis", "font": {"size": 14}},
        ),
        row=1,
        col=4,
    )

    # --- Row 2: Drivers waterfall (basis) ---
    drivers_basis = df[df["Waarderingstype"] == "Basis"][["driver_stap", "driver_mutatie"]].drop_duplicates().dropna()
    drivers_basis = drivers_basis.sort_values("driver_mutatie", ascending=True)

    colors = [OF_GREEN if v >= 0 else OF_RED for v in drivers_basis["driver_mutatie"]]

    fig.add_trace(
        go.Bar(
            x=drivers_basis["driver_mutatie"] * 100,
            y=drivers_basis["driver_stap"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in drivers_basis["driver_mutatie"] * 100],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="ortec_finance",
        title={"text": "<b>Conclusie Marktwaarde Woningen</b>", "x": 0.5, "font": {"size": 20}},
        height=650,
        margin={"t": 80, "b": 40, "l": 200, "r": 60},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mutatie (%)", row=2, col=1, zeroline=True, zerolinewidth=2, zerolinecolor=OF_GRAY)
    fig.update_yaxes(title_text="", row=2, col=1)

    return fig


def _conclusion_beleidswaarde(df: pd.DataFrame) -> go.Figure:
    """Create Beleidswaarde conclusion dashboard: KPI tiles + drivers + parameter changes."""
    # Get first row for scalar metrics (all rows share the same scalar values)
    row0 = df.iloc[0]

    # Build subplot grid: Row 1 = KPIs, Row 2 = drivers, Row 3 = parameter changes
    fig = make_subplots(
        rows=3,
        cols=4,
        row_heights=[0.25, 0.45, 0.30],
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar", "colspan": 4}, None, None, None],
            [{"type": "bar", "colspan": 4}, None, None, None],
        ],
        vertical_spacing=0.12,
    )

    # --- Row 1: KPI indicator tiles ---
    # 1. Totale waardestijging BLW
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=row0["pct_change_median"] * 100,
            number={"suffix": "%", "font": {"size": 38, "color": OF_BLUE}},
            delta={"reference": 0, "relative": False, "suffix": "%", "increasing": {"color": OF_GREEN}},
            title={"text": "<b>Waardestijging</b><br>Mediaan BLW", "font": {"size": 13}},
        ),
        row=1,
        col=1,
    )

    # 2. Ratio BLW/MW
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=row0["median_ratio_blw_mw"] * 100,
            number={"suffix": "%", "font": {"size": 38, "color": OF_ORANGE}},
            title={"text": "<b>Ratio BLW/MW</b><br>Mediaan", "font": {"size": 13}},
        ),
        row=1,
        col=2,
    )

    # 3. Beleidsonderhoud change
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=row0["median_bo"],
            number={"prefix": "€", "font": {"size": 32, "color": OF_BLUE}},
            delta={"reference": float(row0["median_bo_prev"]), "relative": True, "valueformat": ".1%"},
            title={"text": "<b>Beleidsonderhoud</b><br>Mediaan/mnd", "font": {"size": 12}},
        ),
        row=1,
        col=3,
    )

    # 4. Beleidshuur change
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=row0["median_bh"],
            number={"prefix": "€", "font": {"size": 32, "color": OF_BLUE}},
            delta={"reference": float(row0["median_bh_prev"]), "relative": True, "valueformat": ".1%"},
            title={"text": "<b>Beleidshuur</b><br>Mediaan/mnd", "font": {"size": 12}},
        ),
        row=1,
        col=4,
    )

    # --- Row 2: Drivers waterfall ---
    drivers = df[["driver_stap", "driver_mutatie"]].drop_duplicates().dropna()
    drivers = drivers.sort_values("driver_mutatie", ascending=True)

    colors = [OF_GREEN if v >= 0 else OF_RED for v in drivers["driver_mutatie"]]

    fig.add_trace(
        go.Bar(
            x=drivers["driver_mutatie"] * 100,
            y=drivers["driver_stap"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in drivers["driver_mutatie"] * 100],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Row 3: Parameter changes (BO, BB, BH) ---
    param_names = ["Beleidsonderhoud", "Beleidsbeheer", "Beleidshuur"]
    param_changes = [
        float(row0["pct_change_bo"]) * 100,
        float(row0["pct_change_bb"]) * 100,
        float(row0["pct_change_bh"]) * 100,
    ]
    param_colors = [OF_GREEN if v >= 0 else OF_RED for v in param_changes]

    fig.add_trace(
        go.Bar(
            x=param_changes,
            y=param_names,
            orientation="h",
            marker_color=param_colors,
            text=[f"{v:+.1f}%" for v in param_changes],
            textposition="outside",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="ortec_finance",
        title={"text": "<b>Conclusie Beleidswaarde Woningen</b>", "x": 0.5, "font": {"size": 20}},
        height=800,
        margin={"t": 80, "b": 40, "l": 200, "r": 80},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mutatie (%)", row=2, col=1, zeroline=True, zerolinewidth=2, zerolinecolor=OF_GRAY)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="Ontwikkeling parameter (%)", row=3, col=1, zeroline=True, zerolinewidth=2, zerolinecolor=OF_GRAY)
    fig.update_yaxes(title_text="", row=3, col=1)

    return fig


def _build_conclusie_charts(
    *,
    conclusie_mw: pd.DataFrame,
    conclusie_blw: pd.DataFrame,
) -> list[tuple[str, go.Figure, pd.DataFrame]]:
    """Build conclusion dashboard charts."""
    p = "nieuwe figuren"
    charts: list[tuple[str, go.Figure, pd.DataFrame]] = []
    add = chart_appender(charts, p)

    fig_mw = _conclusion_marktwaarde(conclusie_mw)
    add("Conclusie Marktwaarde dashboard", fig_mw, conclusie_mw)

    fig_blw = _conclusion_beleidswaarde(conclusie_blw)
    add("Conclusie Beleidswaarde dashboard", fig_blw, conclusie_blw)

    return charts


@dg.asset(
    group_name="charts",
    name="charts_conclusies",
    description="Visual conclusion dashboards for Marktwaarde and Beleidswaarde — KPI scorecards with driver waterfall.",
    ins={
        "conclusie_mw": dg.AssetIn(key=dg.AssetKey(["models", "chart_conclusie__marktwaarde"])),
        "conclusie_blw": dg.AssetIn(key=dg.AssetKey(["models", "chart_conclusie__beleidswaarde"])),
    },
)
def charts_conclusies(
    context: dg.AssetExecutionContext,
    conclusie_mw: pd.DataFrame,
    conclusie_blw: pd.DataFrame,
) -> None:
    """Generate visual conclusion dashboards for MW and BLW."""
    save_charts(
        _build_conclusie_charts(conclusie_mw=conclusie_mw, conclusie_blw=conclusie_blw),
        context,
        save_fn=save_graph,
    )
