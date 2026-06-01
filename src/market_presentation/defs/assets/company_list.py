"""Dagster asset for company list verification from the TMS API.

Fetches the full company list from ``/api/private/company-list`` and provides
an asset check that compares the API result against the statically defined
``INCLUDED_COMPANIES`` partition list, flagging mismatches.

Assets:
    tms/company_list  (unpartitioned)
        All companies known to the TMS API.
"""

from pathlib import Path

import dagster as dg
import pandas as pd

from ..partitions import INCLUDED_COMPANIES
from ..resources.oauth2_api import Oauth2ApiResource
from ..schemas import CompanyListSchema
from ..utils.tms_report_utils import RawReportConfig, add_raw_metadata, build_api_cache_path, load_or_fetch_json, validate_dataframe

COVERAGE_EXCEL_PATH = build_api_cache_path("company_list_coverage", extension=".xlsx")


def _company_coverage(api_df: pd.DataFrame, included_companies: list[str]) -> tuple[pd.Series, dict[str, list[str]]]:
    result = api_df.copy()
    result["company_id"] = result["company_id"].astype("string").str.strip()
    result = result[result["company_id"].notna() & (result["company_id"] != "")]
    result["is_customer"] = result["is_customer"].fillna(False).astype(bool)

    api_customers = result.groupby("company_id")["is_customer"].max()
    api_companies = set(api_customers.index)
    customer_companies = set(api_customers[api_customers].index)
    non_customer_companies = api_companies - customer_companies
    included = set(included_companies)

    return api_customers, {
        "included_missing_from_api": sorted(included - api_companies),
        "api_customer_missing_from_included": sorted(customer_companies - included),
        "included_not_customer": sorted(included & non_customer_companies),
        "not_included_not_customer": sorted(non_customer_companies - included),
    }


def write_company_coverage_excel(api_df: pd.DataFrame, included_companies: list[str], path: Path) -> None:
    """Write company coverage tables to an Excel workbook with Excel-safe sheet names."""
    api_customers, coverage_lists = _company_coverage(api_df, included_companies)
    included = set(included_companies)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for category, sheet_name in [
            ("included_missing_from_api", "included_missing_api"),
            ("api_customer_missing_from_included", "api_customer_not_included"),
            ("included_not_customer", "included_not_customer"),
            ("not_included_not_customer", "not_included_not_customer"),
        ]:
            rows = [
                {
                    "category": category,
                    "company_id": company_id,
                    "included": company_id in included,
                    "in_api": company_id in api_customers.index,
                    "is_customer": api_customers.get(company_id, pd.NA),
                }
                for company_id in coverage_lists[category]
            ]
            pd.DataFrame(rows, columns=["category", "company_id", "included", "in_api", "is_customer"]).to_excel(writer, sheet_name=sheet_name, index=False)


def evaluate_company_list_coverage(api_df: pd.DataFrame, included_companies: list[str]) -> dict[str, object]:
    """Compare TMS company-list rows against the configured partition list."""
    api_customers, coverage_lists = _company_coverage(api_df, included_companies)

    return {
        "passed": not coverage_lists["included_missing_from_api"] and not coverage_lists["included_not_customer"],
        "api_company_count": len(api_customers),
        "api_customer_company_count": int(api_customers.sum()),
        "included_company_count": len(set(included_companies)),
        **coverage_lists,
    }


@dg.asset(
    key_prefix=["tms"],
    name="company_list",
    compute_kind="api",
    group_name="tms_assets",
    description="Full company list from the TMS API /api/private/company-list endpoint.",
)
def company_list(context: dg.AssetExecutionContext, config: RawReportConfig, tms_api: Oauth2ApiResource) -> pd.DataFrame:
    """Fetch all companies from the TMS API."""
    cache = load_or_fetch_json(
        build_api_cache_path("company_list"),
        tms_api.get_company_list,
        force_download=config.force_download,
    )
    companies = cache.value
    if not isinstance(companies, list):
        raise dg.Failure(description="Invalid company_list JSON cache: expected list")
    if not companies:
        raise dg.Failure(description="No companies returned from /api/private/company-list")
    context.log.info(f"Retrieved {len(companies)} companies from TMS API")
    df = validate_dataframe(pd.DataFrame(companies), CompanyListSchema)
    write_company_coverage_excel(df, INCLUDED_COMPANIES, COVERAGE_EXCEL_PATH)
    add_raw_metadata(context, paths=[str(cache.path), str(COVERAGE_EXCEL_PATH)], sources=[cache.source])
    context.add_output_metadata({"company_coverage_excel": dg.MetadataValue.path(str(COVERAGE_EXCEL_PATH))})
    return df


@dg.asset_check(asset=dg.AssetKey(["tms", "company_list"]))
def check_company_list_coverage(context: dg.AssetCheckExecutionContext, company_list: pd.DataFrame) -> dg.AssetCheckResult:
    """Verify that every included company exists in TMS and is marked as a customer."""
    coverage = evaluate_company_list_coverage(company_list, INCLUDED_COMPANIES)
    metadata = {
        "api_company_count": dg.MetadataValue.int(int(coverage["api_company_count"])),
        "api_customer_company_count": dg.MetadataValue.int(int(coverage["api_customer_company_count"])),
        "included_company_count": dg.MetadataValue.int(int(coverage["included_company_count"])),
        "included_missing_from_api": dg.MetadataValue.json(coverage["included_missing_from_api"]),
        "api_customer_missing_from_included": dg.MetadataValue.json(coverage["api_customer_missing_from_included"]),
        "included_not_customer": dg.MetadataValue.json(coverage["included_not_customer"]),
        "not_included_not_customer": dg.MetadataValue.json(coverage["not_included_not_customer"]),
    }

    if coverage["api_customer_missing_from_included"]:
        context.log.warning(f"Customer companies in API but not in INCLUDED_COMPANIES: {coverage['api_customer_missing_from_included'][:10]}")
    if coverage["included_missing_from_api"]:
        context.log.error(f"Companies in INCLUDED_COMPANIES but not in API: {coverage['included_missing_from_api'][:10]}")
    if coverage["included_not_customer"]:
        context.log.error(f"Companies in INCLUDED_COMPANIES that are not marked as customer: {coverage['included_not_customer'][:10]}")

    return dg.AssetCheckResult(
        passed=bool(coverage["passed"]),
        metadata=metadata,
        description=(
            "All INCLUDED_COMPANIES are present in the TMS API and marked as customer"
            if coverage["passed"]
            else f"{len(coverage['included_missing_from_api'])} missing and {len(coverage['included_not_customer'])} non-customer included companies"
        ),
    )
