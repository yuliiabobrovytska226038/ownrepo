"""Helpers for materializing dbt chart views in validation DuckDB connections."""

from __future__ import annotations

from pathlib import Path

import duckdb
from jinja2 import Environment, StrictUndefined

from ..config import CHART_YEAR

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DBT_MODELS = PROJECT_ROOT / "dbt" / "models"
DBT_CHART_MODELS = DBT_MODELS / "charts"
DBT_MACROS = PROJECT_ROOT / "dbt" / "macros"
DEFAULT_DBT_JAAR = CHART_YEAR


def _macro_sources() -> list[str]:
    if not DBT_MACROS.exists():
        return []
    return [
        macro_file.read_text(encoding="utf-8")
        for macro_file in sorted(DBT_MACROS.glob("*.sql"))
        if macro_file.name != "generate_schema_name.sql"
    ]


def render_chart_model_sql(model_name: str, jaar: int = DEFAULT_DBT_JAAR) -> str:
    """Render one dbt chart model SQL file for ad-hoc validation."""
    model_path = DBT_CHART_MODELS / f"{model_name}.sql"
    if not model_path.exists():
        matches = sorted(DBT_MODELS.rglob(f"{model_name}.sql"))
        if not matches:
            raise FileNotFoundError(f"dbt model not found: {model_path}")
        model_path = matches[0]

    def var(name: str) -> int:
        if name != "jaar":
            raise KeyError(f"Unsupported dbt var in chart validation helper: {name}")
        return jaar

    def ref(name: str) -> str:
        return f'models."{name}"'

    def source(source_name: str, table_name: str) -> str:
        return f'{source_name}."{table_name}"'

    model_source = model_path.read_text(encoding="utf-8")
    combined_source = "\n".join(_macro_sources() + [model_source])

    env = Environment(undefined=StrictUndefined)
    template = env.from_string(combined_source)
    return template.render(var=var, ref=ref, source=source)


def materialize_chart_view(con: duckdb.DuckDBPyConnection, model_name: str, jaar: int = DEFAULT_DBT_JAAR) -> None:
    """Create or replace one rendered dbt chart model as ``models.<model_name>``."""
    con.execute("CREATE SCHEMA IF NOT EXISTS models")
    sql = render_chart_model_sql(model_name, jaar)
    con.execute(f'CREATE OR REPLACE VIEW models."{model_name}" AS {sql}')
