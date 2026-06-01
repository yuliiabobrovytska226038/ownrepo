"""Unit tests for Plotly graph generation utilities.

Tests cover:
- of_template registration
- save_graph HTML/JPEG output
- save_graph error handling (empty figure, empty DataFrame)
- set_template_axis_format
- set_bar_text_horizontal
- set_line_marker_style
- chart_appender
- save_charts iteration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import pytest

from market_presentation.defs.utils.graph_utils import (
    chart_appender,
    of_template,
    save_charts,
    save_graph,
    set_bar_text_horizontal,
    set_line_marker_style,
    set_template_axis_format,
)


# ---------------------------------------------------------------------------
# of_template
# ---------------------------------------------------------------------------


class TestOfTemplate:
    def test_registers_ortec_finance_template(self):
        of_template()
        assert "ortec_finance" in pio.templates

    def test_template_has_brand_colors(self):
        of_template()
        template = pio.templates["ortec_finance"]
        assert "#0084CB" in template.layout.colorway  # Ortec blue

    def test_template_uses_fira_sans(self):
        of_template()
        template = pio.templates["ortec_finance"]
        assert template.layout.font.family == "Fira Sans"

    def test_template_uses_dutch_separators(self):
        of_template()
        template = pio.templates["ortec_finance"]
        assert template.layout.separators == ",."


# ---------------------------------------------------------------------------
# save_graph
# ---------------------------------------------------------------------------


class TestSaveGraph:
    def test_raises_on_empty_figure(self, tmp_path):
        fig = go.Figure()  # No data traces
        context = MagicMock()

        with patch("market_presentation.defs.utils.graph_utils.GRAFIEKEN_DIR", tmp_path):
            with pytest.raises(ValueError, match="no data traces"):
                save_graph(fig, "empty_chart", context)

    def test_raises_on_empty_dataframe(self, tmp_path):
        fig = go.Figure(data=[go.Bar(x=[1], y=[2])])
        context = MagicMock()
        empty_df = pd.DataFrame()

        with patch("market_presentation.defs.utils.graph_utils.GRAFIEKEN_DIR", tmp_path):
            with pytest.raises(ValueError, match="empty"):
                save_graph(fig, "empty_df_chart", context, df=empty_df)

    def test_saves_html_file(self, tmp_path):
        fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[1, 2])])
        context = MagicMock()

        with patch("market_presentation.defs.utils.graph_utils.GRAFIEKEN_DIR", tmp_path):
            result = save_graph(fig, "test_chart", context, export_image=False)

        assert result.exists()
        assert result.suffix == ".html"
        assert "test_chart" in result.name

    def test_saves_with_dataframe_xlsx(self, tmp_path):
        fig = go.Figure(data=[go.Bar(x=["A"], y=[1])])
        context = MagicMock()
        df = pd.DataFrame({"col": [1, 2, 3]})

        with patch("market_presentation.defs.utils.graph_utils.GRAFIEKEN_DIR", tmp_path):
            result = save_graph(fig, "with_data", context, df=df, export_image=False)

        xlsx_path = result.with_suffix(".xlsx")
        assert xlsx_path.exists()

    def test_logs_on_image_export_failure(self, tmp_path):
        fig = go.Figure(data=[go.Bar(x=["A"], y=[1])])
        context = MagicMock()

        with patch("market_presentation.defs.utils.graph_utils.GRAFIEKEN_DIR", tmp_path):
            with patch("plotly.io.write_image", side_effect=Exception("kaleido not found")):
                result = save_graph(fig, "no_image", context, export_image=True)

        # HTML still saved despite image failure
        assert result.exists()
        context.log.error.assert_called()


# ---------------------------------------------------------------------------
# set_template_axis_format
# ---------------------------------------------------------------------------


class TestSetTemplateAxisFormat:
    def test_sets_y_axis_format(self):
        of_template()
        fig = go.Figure(data=[go.Bar(x=[1], y=[0.5])])
        fig.update_layout(template="ortec_finance")
        result = set_template_axis_format(fig, axis="y", tickformat="0%", hoverformat=".2%")
        assert result is fig  # Returns same figure

    def test_sets_x_axis_format(self):
        of_template()
        fig = go.Figure(data=[go.Bar(x=[1], y=[1])])
        fig.update_layout(template="ortec_finance")
        result = set_template_axis_format(fig, axis="x", tickformat=".0f", hoverformat=".1f")
        assert result is fig


# ---------------------------------------------------------------------------
# set_bar_text_horizontal / set_line_marker_style
# ---------------------------------------------------------------------------


class TestChartHelpers:
    def test_set_bar_text_horizontal(self):
        fig = go.Figure(data=[go.Bar(x=[1, 2], y=[3, 4], text=["a", "b"])])
        result = set_bar_text_horizontal(fig)
        assert result is fig

    def test_set_line_marker_style(self):
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        result = set_line_marker_style(fig, width=3, size=10)
        assert result is fig


# ---------------------------------------------------------------------------
# chart_appender
# ---------------------------------------------------------------------------


class TestChartAppender:
    def test_appends_chart_triple(self):
        charts = []
        add = chart_appender(charts, "waarde")
        fig = go.Figure(data=[go.Bar(x=[1], y=[2])])
        df = pd.DataFrame({"x": [1]})
        add("per_model", fig, df)
        assert len(charts) == 1
        assert charts[0][0] == "waarde/per_model"
        assert charts[0][1] is fig
        assert charts[0][2] is df

    def test_multiple_appends(self):
        charts = []
        add = chart_appender(charts, "prefix")
        for i in range(3):
            add(f"chart_{i}", go.Figure(data=[go.Bar(x=[i], y=[i])]), pd.DataFrame())
        assert len(charts) == 3


# ---------------------------------------------------------------------------
# save_charts
# ---------------------------------------------------------------------------


class TestSaveCharts:
    def test_calls_save_fn_for_each_chart(self):
        calls = []

        def mock_save(fig, path, context, df=None):
            calls.append((path, df))

        charts = [
            ("chart_a", go.Figure(data=[go.Bar(x=[1], y=[1])]), pd.DataFrame({"a": [1]})),
            ("chart_b", go.Figure(data=[go.Bar(x=[2], y=[2])]), pd.DataFrame({"b": [2]})),
        ]
        context = MagicMock()
        save_charts(charts, context, save_fn=mock_save)
        assert len(calls) == 2
        assert calls[0][0] == "chart_a"
        assert calls[1][0] == "chart_b"
