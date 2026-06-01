"""Performance benchmark for dbt models and DuckDB queries.

Measures execution times for:
- Individual dbt model rebuilds
- Key analytical queries on the models schema
- Comparison with legacy tms.db (if available)

Usage:
    uv run python scripts/benchmark_performance.py
    uv run python scripts/benchmark_performance.py --compare-legacy
"""

import argparse
import statistics
import time
from pathlib import Path

import duckdb

NEW_DB = Path("database/local.duckdb")
LEGACY_DB_DEFAULT = Path(r"C:\Users\YuliiaB\OneDrive - ORTEC Finance\Desktop\Marktpresentatie\data\tms.db")

# Key queries that represent typical analytical workloads
BENCHMARK_QUERIES = {
    "dataset_basis_full_scan": "SELECT COUNT(*) FROM models.dataset_basis",
    "dataset_basis_filter_corporatie": "SELECT COUNT(*) FROM models.dataset_basis WHERE Corporatie = '3B Wonen'",
    "dataset_basis_aggregate_marktwaarde": """
        SELECT Corporatie, Waarderingsmodel, SUM(Marktwaarde) AS total_marktwaarde, COUNT(*) AS n
        FROM models.dataset_basis
        GROUP BY Corporatie, Waarderingsmodel
    """,
    "dataset_validatie_full_scan": "SELECT COUNT(*) FROM models.dataset_validatie",
    "dataset_ontwikkeling_yoy": """
        SELECT Corporatie, COUNT(*) AS n,
               AVG(Marktwaarde) AS avg_mw_current
        FROM models.dataset_ontwikkeling
        GROUP BY Corporatie
    """,
    "int_vastgoedgegevens_join": """
        SELECT COUNT(*) FROM models.int_vastgoedgegevens
        WHERE Corporatie IS NOT NULL AND "VHE-nummer" IS NOT NULL
    """,
    "int_parameteroverzicht_scan": "SELECT COUNT(*) FROM models.int_parameteroverzicht",
    "geographic_lookup": """
        SELECT "COROP-gebied", "Provincies Naam", COUNT(*) AS n, SUM(Marktwaarde) AS total
        FROM models.dataset_basis
        WHERE "COROP-gebied" IS NOT NULL
        GROUP BY "COROP-gebied", "Provincies Naam"
        ORDER BY total DESC
    """,
    "pct_full_analysis": """
        SELECT Waarderingsmodel, "% Full", COUNT(*) AS n
        FROM models.dataset_basis
        GROUP BY Waarderingsmodel, "% Full"
    """,
    "complex_aggregation": """
        SELECT Waarderingscomplex, Corporatie,
               COUNT(*) AS n_vhe,
               SUM(Marktwaarde) AS total_mw,
               AVG(Disconteringsvoet) AS avg_dv
        FROM models.dataset_basis
        GROUP BY Waarderingscomplex, Corporatie
        HAVING COUNT(*) > 1
    """,
}

# Legacy-equivalent queries (same logic, different table names)
LEGACY_QUERIES = {
    "dataset_basis_full_scan": "SELECT COUNT(*) FROM dataset_basis",
    "dataset_basis_filter_corporatie": "SELECT COUNT(*) FROM dataset_basis WHERE Corporatie = '3B Wonen'",
    "dataset_basis_aggregate_marktwaarde": """
        SELECT Corporatie, Waarderingsmodel, SUM(Marktwaarde) AS total_marktwaarde, COUNT(*) AS n
        FROM dataset_basis
        GROUP BY Corporatie, Waarderingsmodel
    """,
    "dataset_validatie_full_scan": "SELECT COUNT(*) FROM dataset_validatie",
    "dataset_ontwikkeling_yoy": """
        SELECT Corporatie, COUNT(*) AS n,
               AVG(Marktwaarde) AS avg_mw_current
        FROM dataset_ontwikkeling
        GROUP BY Corporatie
    """,
}


def _time_query(con: duckdb.DuckDBPyConnection, sql: str, warmup: int = 1, runs: int = 5) -> dict:
    """Execute a query multiple times and return timing stats."""
    # Warmup
    for _ in range(warmup):
        try:
            con.execute(sql).fetchall()
        except Exception:
            return {"error": True, "median_ms": -1, "min_ms": -1, "max_ms": -1, "runs": 0}

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        con.execute(sql).fetchall()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "error": False,
        "median_ms": round(statistics.median(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "runs": runs,
    }


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def benchmark_new_db() -> dict[str, dict]:
    """Benchmark queries on the new DuckDB database."""
    if not NEW_DB.exists():
        print(f"ERROR: {NEW_DB} not found. Materialize assets first.")
        return {}

    con = duckdb.connect(str(NEW_DB), read_only=True)
    results = {}

    print("\n=== NEW PIPELINE BENCHMARKS ===\n")
    print(f"{'Query':<45} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 80)

    for name, sql in BENCHMARK_QUERIES.items():
        table_ref = sql.split("FROM")[-1].strip().split()[0] if "FROM" in sql else ""
        if table_ref and not _table_exists(con, table_ref):
            results[name] = {"error": True, "median_ms": -1, "note": f"table {table_ref} not found"}
            print(f"{name:<45} {'SKIP':>12} (table not found)")
            continue

        stats = _time_query(con, sql)
        results[name] = stats
        if stats["error"]:
            print(f"{name:<45} {'ERROR':>12}")
        else:
            print(f"{name:<45} {stats['median_ms']:>10.2f}ms {stats['min_ms']:>10.2f}ms {stats['max_ms']:>10.2f}ms")

    con.close()
    return results


def benchmark_legacy_db(legacy_path: Path) -> dict[str, dict]:
    """Benchmark queries on the legacy DuckDB database."""
    if not legacy_path.exists():
        print(f"\nLegacy DB not found at {legacy_path} — skipping legacy comparison.")
        return {}

    con = duckdb.connect(str(legacy_path), read_only=True)
    results = {}

    print("\n=== LEGACY PIPELINE BENCHMARKS ===\n")
    print(f"{'Query':<45} {'Median (ms)':>12} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 80)

    for name, sql in LEGACY_QUERIES.items():
        stats = _time_query(con, sql)
        results[name] = stats
        if stats["error"]:
            print(f"{name:<45} {'ERROR':>12}")
        else:
            print(f"{name:<45} {stats['median_ms']:>10.2f}ms {stats['min_ms']:>10.2f}ms {stats['max_ms']:>10.2f}ms")

    con.close()
    return results


def compare_results(new_results: dict, legacy_results: dict) -> None:
    """Compare new vs legacy performance and flag regressions."""
    if not legacy_results:
        return

    print("\n=== PERFORMANCE COMPARISON (new vs legacy) ===\n")
    print(f"{'Query':<45} {'New (ms)':>10} {'Legacy (ms)':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 90)

    regressions = 0
    for name in LEGACY_QUERIES:
        new = new_results.get(name, {})
        legacy = legacy_results.get(name, {})

        if new.get("error") or legacy.get("error"):
            print(f"{name:<45} {'N/A':>10} {'N/A':>12} {'N/A':>8} {'SKIP':>8}")
            continue

        new_ms = new["median_ms"]
        legacy_ms = legacy["median_ms"]
        ratio = new_ms / legacy_ms if legacy_ms > 0 else float("inf")
        status = "OK" if ratio <= 1.10 else "WARN" if ratio <= 1.50 else "FAIL"
        if ratio > 1.10:
            regressions += 1

        print(f"{name:<45} {new_ms:>8.2f}ms {legacy_ms:>10.2f}ms {ratio:>7.2f}x {status:>8}")

    print(f"\nRegressions (>10% slower): {regressions}/{len(LEGACY_QUERIES)}")
    if regressions == 0:
        print("✓ All queries within 10% of legacy performance.")


def benchmark_dbt_build_time() -> None:
    """Report dbt model build times from the target/run_results.json if available."""
    run_results_path = Path("dbt/target/run_results.json")
    if not run_results_path.exists():
        print("\nNo dbt run_results.json found. Run `dbt build` first to collect build times.")
        return

    import json

    with open(run_results_path) as f:
        data = json.load(f)

    print("\n=== DBT MODEL BUILD TIMES ===\n")
    print(f"{'Model':<55} {'Time (s)':>10} {'Status':>8}")
    print("-" * 75)

    results = sorted(data.get("results", []), key=lambda r: r.get("execution_time", 0), reverse=True)
    for result in results:
        node = result.get("unique_id", "unknown")
        model_name = node.split(".")[-1] if "." in node else node
        exec_time = result.get("execution_time", 0)
        status = result.get("status", "unknown")
        if "test" in node:
            continue  # Skip test results
        print(f"{model_name:<55} {exec_time:>8.2f}s {status:>8}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DuckDB query performance")
    parser.add_argument("--compare-legacy", action="store_true", help="Also benchmark legacy DB and compare")
    parser.add_argument("--legacy-db", type=Path, default=LEGACY_DB_DEFAULT, help="Path to legacy DuckDB/tms.db file")
    args = parser.parse_args()

    new_results = benchmark_new_db()
    benchmark_dbt_build_time()

    if args.compare_legacy:
        legacy_results = benchmark_legacy_db(args.legacy_db)
        compare_results(new_results, legacy_results)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
