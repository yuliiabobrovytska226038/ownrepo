"""Dagster resource for OAuth2-authenticated REST API access.

Provides ``Oauth2ApiResource``, a configurable Dagster resource that manages
OAuth2 ROPC (Resource Owner Password Credentials) token acquisition and
automatic refresh on 401 responses. It wraps an ``httpx.Client`` with a
custom auth handler and exposes raw HTTP methods plus domain-specific helper
methods for the TMS and VGR APIs.

This resource is used as both ``tms_api`` and ``vgr_api`` in the pipeline,
each configured with different base URLs but sharing the same OAuth2 token
endpoint and credentials.

Classes:
    _BearerTokenRefreshAuth: httpx Auth handler with transparent 401 retry.
    Oauth2ApiResource: Dagster ConfigurableResource for authenticated API calls.
"""

import io
import logging
import os
import time
import urllib.parse
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import dagster as dg
import httpx
import pandas as pd
import urllib3
from oauthlib.oauth2 import LegacyApplicationClient
from pydantic import Field, PrivateAttr
from requests_oauthlib import OAuth2Session

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _issue_date_to_iso(issue_date: dict[str, Any]) -> str | None:
    year = issue_date.get("year")
    quarter = issue_date.get("quarter")
    if year is None or quarter != "Q4":
        return None
    return f"{int(year)}-12-31"


def normalize_company_valuation_data(
    company_id: str,
    *,
    valuation_rounds: list[dict[str, Any]],
    valuations_by_round_id: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten TMS valuation-round and valuation payloads."""
    rows: list[dict[str, Any]] = []

    for valuation_round in valuation_rounds:
        round_id = int(valuation_round["id"])
        issue_date = valuation_round.get("issueDate") or {}
        valuations = valuations_by_round_id.get(round_id, [])
        round_common = {
            "company_id": company_id,
            "valuation_round_id": round_id,
            "round_data_set_id": valuation_round.get("dataSetId"),
            "issue_year": issue_date.get("year"),
            "issue_quarter": issue_date.get("quarter"),
            "issue_date": _issue_date_to_iso(issue_date),
            "round_status": valuation_round.get("status"),
            "round_is_free": valuation_round.get("isFree"),
        }

        if not valuations:
            rows.append(round_common | _empty_valuation_columns())
            continue

        for valuation in valuations:
            sub_portfolio = valuation.get("subPortfolio") or {}
            rows.append(
                round_common
                | {
                    "valuation_id": valuation.get("valuationId"),
                    "valuation_data_set_id": valuation.get("dataSetId"),
                    "valuation_name": valuation.get("name"),
                    "model_year": valuation.get("modelYear"),
                    "archived": valuation.get("archived"),
                    "archived_date_time": valuation.get("archivedDateTime"),
                    "creation_date_time": valuation.get("creationDateTime"),
                    "data_set_update_date_time": valuation.get("dataSetUpdateDateTime"),
                    "avm_issue_date": valuation.get("avmIssueDate"),
                    "is_old_data_set": valuation.get("isOldDataSet"),
                    "has_changed_parameters": valuation.get("hasChangedParameters"),
                    "is_new_calculation_rules": valuation.get("isNewCalculationRules"),
                    "main_valuation": valuation.get("mainValuation"),
                    "sub_portfolio_id": sub_portfolio.get("id") or sub_portfolio.get("subPortfolioId"),
                    "sub_portfolio_name": sub_portfolio.get("name"),
                    "sub_portfolio_valuation_level": sub_portfolio.get("valuationLevel"),
                }
            )
    return rows


def _empty_valuation_columns() -> dict[str, Any]:
    return {
        "valuation_id": None,
        "valuation_data_set_id": None,
        "valuation_name": None,
        "model_year": None,
        "archived": None,
        "archived_date_time": None,
        "creation_date_time": None,
        "data_set_update_date_time": None,
        "avm_issue_date": None,
        "is_old_data_set": None,
        "has_changed_parameters": None,
        "is_new_calculation_rules": None,
        "main_valuation": None,
        "sub_portfolio_id": None,
        "sub_portfolio_name": None,
        "sub_portfolio_valuation_level": None,
    }


class _BearerTokenRefreshAuth(httpx.Auth):
    """httpx Auth handler: injects a Bearer token and retries exactly once on 401.

    The token is read via *get_token_fn* on every request so it is always
    up-to-date after *refresh_fn* has updated it — no client rebuild needed.
    """

    def __init__(self, get_token_fn: Callable[[], str], refresh_fn: Callable[[], None]) -> None:
        self._get_token = get_token_fn
        self._refresh = refresh_fn

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Inject the bearer token; on 401 refresh the token and retry once."""
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        response = yield request
        if response.status_code == 401:
            self._refresh()
            request.headers["Authorization"] = f"Bearer {self._get_token()}"
            yield request


class Oauth2ApiResource(dg.ConfigurableResource):
    """Dagster resource providing an OAuth2-authenticated API client.

    Auth flow
    ---------
    On ``setup_for_execution`` the resource acquires an access token via
    OAuth2 ROPC and injects a ``_BearerTokenRefreshAuth`` handler into the
    underlying ``httpx.Client``. Any HTTP call that returns 401 is
    transparently retried once after re-acquiring the token — **no retry
    code is needed in the calling assets**.

    Usage in assets
    ---------------
    ``api.request(method, url, **kwargs)`` — plain GET/POST calls.
    ``api.client`` — optional low-level access to the underlying
    ``httpx.Client`` when a raw client is still useful.
    """

    base_url: str = Field(description="Base URL for the TMS/VGR API (e.g. https://host/tms)")
    token_url: str = Field(description="OAuth2 token endpoint URL")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    username: str = Field(description="OAuth2 username (ROPC flow)")
    password: str = Field(description="OAuth2 password (ROPC flow)")

    _token: dict[str, Any] = PrivateAttr(default_factory=dict)
    _httpx_client: httpx.Client | None = PrivateAttr(default=None)
    _logger: logging.Logger | None = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """Acquire the initial OAuth2 token and build the authenticated client."""
        self._logger = logging.getLogger(__name__)
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        self._fetch_token()
        self._build_client()

    def _fetch_token(self) -> None:
        """(Re-)acquire the OAuth2 access token via ROPC and store it in ``_token``."""
        session = OAuth2Session(client=LegacyApplicationClient(client_id=self.client_id))
        self._token = session.fetch_token(
            token_url=self.token_url,
            username=self.username,
            password=self.password,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        if self._logger:
            self._logger.info("OAuth token acquired for %s", self.base_url)

    def _build_client(self) -> None:
        """Build the underlying ``httpx.Client`` with the token-refresh auth handler."""
        if self._httpx_client is not None:
            self._httpx_client.close()

        auth = _BearerTokenRefreshAuth(
            get_token_fn=lambda: self._token["access_token"],
            refresh_fn=self._fetch_token,
        )
        self._httpx_client = httpx.Client(base_url=self.base_url, auth=auth, verify=False, timeout=httpx.Timeout(120.0))

    @property
    def client(self) -> httpx.Client:
        """Return the underlying ``httpx.Client`` for low-level access."""
        if self._httpx_client is None:
            raise RuntimeError("Oauth2ApiResource has not been initialised. Call setup_for_execution first.")
        return self._httpx_client

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Make an authenticated HTTP request with automatic 401 retry.

        Token refresh and one retry are handled transparently by the underlying
        ``_BearerTokenRefreshAuth`` handler — no additional try/except or refresh
        logic is needed at the call site.
        """
        return self.client.request(method, url, **kwargs)

    def request_with_retry(
        self,
        method: str,
        url: str,
        *,
        timeout: float = 60.0,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> httpx.Response | None:
        """Make an API request with exponential-backoff retry on transient errors.

        Retries on ``httpx.TimeoutException``, ``httpx.RemoteProtocolError``,
        and HTTP 500/503 status codes.  Returns the response on success, ``None``
        when all attempts are exhausted.
        """
        for attempt in range(max_attempts):
            try:
                r = self.request(method, url, timeout=timeout, **kwargs)
            except (httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                wait = 2**attempt
                if self._logger:
                    self._logger.warning("%s attempt %d failed: %r - retrying in %ds", url, attempt + 1, exc, wait)
                time.sleep(wait)
                continue
            if r.status_code in (500, 503):
                wait = 2**attempt
                if self._logger:
                    self._logger.warning("%s returned %d - retrying in %ds (attempt %d)", url, r.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            return r
        return None

    def get_company_list(self) -> list[dict]:
        """Return all companies from ``/api/private/company-list``.

        Each dict has keys ``company_id`` and ``is_customer``.
        """
        r = self.request("GET", "/api/private/company-list")
        if r.status_code != 200:
            return []
        return [
            {
                "company_id": c.get("companyId") or c.get("companyName") or c.get("name") or c.get("id"),
                "is_customer": c.get("isCustomer", False),
            }
            for c in r.json()
        ]

    def get_raw_company_list(self) -> list[dict]:
        """Return raw company records from ``/api/private/company-list``."""
        r = self.request("GET", "/api/private/company-list")
        if r.status_code != 200:
            return []
        return r.json() or []

    def get_raw_valuation_rounds(self, company_id: str) -> list[dict]:
        """Return raw valuation rounds for *company_id* from the TMS API."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds")
        if r.status_code != 200:
            return []
        return r.json() or []

    def get_valuation_round_valuations(self, company_id: str, valuation_round_id: int) -> list[dict]:
        """Return raw valuations for a TMS valuation round."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/valuations")
        if r.status_code != 200:
            return []
        return r.json() or []

    def get_company_valuation_data(self, company_id: str, issue_years: list[int]) -> list[dict[str, Any]]:
        """Return joined valuation-round and valuation records for *company_id*."""
        valuation_rounds = []
        for valuation_round in self.get_raw_valuation_rounds(company_id):
            issue_date = valuation_round.get("issueDate") or {}
            if issue_date.get("year") in issue_years and issue_date.get("quarter") == "Q4":
                valuation_rounds.append(valuation_round)

        valuations_by_round_id = {
            int(valuation_round["id"]): self.get_valuation_round_valuations(company_id, int(valuation_round["id"]))
            for valuation_round in valuation_rounds
            if valuation_round.get("id") is not None
        }
        return normalize_company_valuation_data(
            company_id,
            valuation_rounds=valuation_rounds,
            valuations_by_round_id=valuations_by_round_id,
        )

    def get_vgr_datasets(self, company_id: str, issue_years: list[int]) -> list[dict]:
        """Return published Q4 VGR datasets for *company_id*.

        Calls ``/api/private/companies/{id}/data-sets`` and returns a list of
        dicts with keys ``data_set_id`` and ``issue_year`` for each published
        Q4 dataset matching *issue_years*.
        """
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("GET", f"/api/private/companies/{cid}/data-sets")
        if r is None or r.status_code != 200:
            return []
        return [
            {"data_set_id": ds.get("dataSetId") or ds.get("datasetId") or ds.get("id"), "issue_year": (ds.get("issueDate") or {}).get("year")}
            for ds in r.json()
            if ds.get("isPublished") and (ds.get("issueDate") or {}).get("quarter") == "Q4" and (ds.get("issueDate") or {}).get("year") in issue_years
        ]

    def download_vgr_import_file(self, company_id: str, data_set_id: str) -> pd.DataFrame | None:
        """Download and parse the VGR import file for a single dataset.

        Tries to parse the response as Excel first, then CSV.
        Returns a DataFrame on success, None on failure.
        """
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("GET", f"/api/private/companies/{cid}/data-sets/{data_set_id}/download-import-file", timeout=120.0)
        if r is None or r.status_code != 200:
            return None
        try:
            return pd.read_excel(io.BytesIO(r.content))
        except Exception:
            try:
                return pd.read_csv(io.BytesIO(r.content))
            except Exception:
                if self._logger:
                    self._logger.warning("Could not parse %s/ds=%s as Excel or CSV", company_id, data_set_id)
                return None

    def download_vgr_import_file_bytes(self, company_id: str, data_set_id: str) -> bytes | None:
        """Download the VGR import file as raw bytes for multi-sheet parsing."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("GET", f"/api/private/companies/{cid}/data-sets/{data_set_id}/download-import-file", params={"datagrouping": "VALUATION_DATA"}, timeout=120.0)
        if r is None or r.status_code != 200:
            return None
        return r.content

    def fetch_vgr_dataset_json(self, company_id: str, data_set_id: str) -> dict | None:
        """Fetch VGR dataset as JSON (includes rental unit internalId + rentalUnitNumber)."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("GET", f"/api/private/companies/{cid}/data-sets/{data_set_id}", timeout=120.0)
        if r is None or r.status_code != 200:
            return None
        return r.json()

    def fetch_vgr_valuation_complexes(self, company_id: str, data_set_id: str) -> dict | None:
        """Fetch VGR valuation complexes for a dataset, matching the legacy subsidiary lookup."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry(
            "POST",
            f"/api/private/companies/{cid}/data-sets/{data_set_id}/valuation-complexes-query",
            json={"valuationComplexInternalIds": [], "includeRentalUnits": False},
            timeout=120.0,
        )
        if r is None or r.status_code != 200:
            return None
        return r.json()

    # ------------------------------------------------------------------
    # TMS report fetch methods
    # ------------------------------------------------------------------

    def fetch_market_value_parameters(self, company_id: str, valuation_round_id: int) -> bytes:
        """Download marktwaardeparameters Excel export. Returns raw bytes."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("POST", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/export-market-value-parameters", json=[], timeout=120.0)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"marktwaardeparameters returned {getattr(r, 'status_code', 'no response')} for {company_id}/{valuation_round_id}")
        return r.content

    def fetch_policy_value_parameters(self, company_id: str, valuation_round_id: int) -> bytes:
        """Download beleidswaardeparameters Excel export. Returns raw bytes."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry(
            "POST",
            f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/export-policy-value-parameters",
            json={"complexInternalIds": []},
            timeout=300.0,
            max_attempts=2,
        )
        if r is None or r.status_code != 200:
            raise RuntimeError(f"beleidswaardeparameters returned {getattr(r, 'status_code', 'no response')} for {company_id}/{valuation_round_id}")
        return r.content

    def fetch_complex_references(self, company_id: str, valuation_round_id: int) -> bytes:
        """Download complexreferenties Excel export. Returns raw bytes."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("POST", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/export-complexes-references", json=[], timeout=120.0)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"complexreferenties returned {getattr(r, 'status_code', 'no response')} for {company_id}/{valuation_round_id}")
        return r.content

    def fetch_difference_analysis(self, company_id: str, valuation_round_id: int, subsidiary: str) -> dict:
        """Fetch verschilanalyse JSON for a specific subsidiary. Returns response dict (may be empty)."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/difference-analysis", params={"subsidiary": subsidiary, "include-result": "true"})
        return r.json() if r.status_code == 200 else {}

    def fetch_property_info(self, company_id: str, valuation_round_id: int) -> dict:
        """Fetch DVI report JSON. Returns response dict (may be empty)."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("POST", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/dvi-report", json=[])
        return r.json() if r.status_code == 200 else {}

    def fetch_energy_performance(self, company_id: str, valuation_round_id: int) -> list[dict]:
        """Fetch energy performance policy-values. Returns list of complex dicts."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/complexes/policy-values")
        if r.status_code != 200:
            raise RuntimeError(f"energy_performance returned {r.status_code} for {company_id}")
        return r.json() or []

    def fetch_ratios(self, company_id: str, valuation_round_id: int) -> list[dict]:
        """Fetch ratio analysis data. Returns list of ratio dicts."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/complexes/ratios")
        if r.status_code != 200:
            raise RuntimeError(f"ratios returned {r.status_code} for {company_id}")
        return r.json() or []

    def fetch_ratios_report(self, company_id: str, valuation_round_id: int, output_path: Path) -> bool:
        """Trigger, poll, download ratio rapport Excel (RATIOS report type)."""
        cid = urllib.parse.quote(str(company_id), safe="")
        return self._trigger_poll_download(company_id, f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/report", {"complexInternalIds": [], "reportType": "RATIOS"}, output_path)

    def fetch_complex_characteristics_excel(self, company_id: str, valuation_round_id: int) -> bytes:
        """Download complexkenmerken Excel export. Returns raw bytes."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("POST", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/export-complexes-characteristics", json=[], timeout=120.0)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"complex_characteristics_excel returned {getattr(r, 'status_code', 'no response')} for {company_id}/{valuation_round_id}")
        return r.content

    def fetch_waterfall_analysis(self, company_id: str, valuation_round_id: int) -> dict:
        """Fetch policy value waterfall analysis JSON. Returns response dict (may be empty)."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request_with_retry("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/policy-value-waterfall-analysis", params={"subsidiary": company_id})
        return r.json() if r is not None and r.status_code == 200 else {}

    def fetch_complex_characteristics(self, company_id: str, valuation_round_id: int) -> list[dict]:
        """Fetch complexkenmerken. Returns list of dicts."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/complexes-characteristics")
        if r.status_code != 200:
            raise RuntimeError(f"complex_characteristics returned {r.status_code} for {company_id}")
        return r.json() or []

    def fetch_market_value_basis(self, company_id: str, valuation_round_id: int) -> list[dict]:
        """Fetch per-VHE market values via two-step API call. Returns flat list of VHE dicts."""
        cid = urllib.parse.quote(str(company_id), safe="")
        r = self.request("GET", f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/complexes/market-values", timeout=30.0)
        if r.status_code != 200:
            raise RuntimeError(f"complexes/market-values returned {r.status_code} for {company_id}/{valuation_round_id}")
        complexes = r.json()
        if self._logger:
            self._logger.info("%s/%s: %d complexes", company_id, valuation_round_id, len(complexes))
        time.sleep(0.2)

        all_records: list[dict] = []
        for cx in complexes:
            valuation_id = cx.get("valuationId")
            complex_internal_id = cx.get("complexInternalId")
            if not valuation_id or not complex_internal_id:
                continue
            r4 = self.request_with_retry(
                "GET",
                f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/valuations/{valuation_id}/complexes/{complex_internal_id}/rental-units/market-values",
                timeout=30.0,
            )
            if r4 is not None and r4.status_code == 200 and r4.json():
                all_records.extend(r4.json())
            time.sleep(0.1)
        return all_records

    def fetch_parameters_report(self, company_id: str, valuation_round_id: int, output_path: Path) -> bool:
        """Trigger, poll, download parameteroverzicht report."""
        cid = urllib.parse.quote(str(company_id), safe="")
        return self._trigger_poll_download(company_id, f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/parameters-report", [], output_path)

    def fetch_valuation_overview(self, company_id: str, valuation_round_id: int, output_path: Path) -> bool:
        """Trigger, poll, download waardeoverzicht report."""
        cid = urllib.parse.quote(str(company_id), safe="")
        return self._trigger_poll_download(
            company_id, f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/report", {"complexInternalIds": [], "reportType": "MARKET_VALUE_OVERVIEW"}, output_path
        )

    def fetch_policy_value_report(self, company_id: str, valuation_round_id: int, output_path: Path) -> bool:
        """Trigger, poll, download beleidswaarderapport."""
        cid = urllib.parse.quote(str(company_id), safe="")
        return self._trigger_poll_download(
            company_id, f"/api/private/companies/{cid}/valuation-rounds/{valuation_round_id}/report", {"complexInternalIds": [], "reportType": "POLICY_VALUE_WATERFALL"}, output_path
        )

    def _trigger_poll_download(self, company_id: str, trigger_url: str, trigger_json: object, output_path: Path, poll_interval: int = 5, max_polls: int = 60) -> bool:
        """Trigger an async TMS report, poll until ready, download to *output_path*. Returns True on success."""
        cid = urllib.parse.quote(str(company_id), safe="")

        resp = self.request_with_retry("POST", trigger_url, json=trigger_json)
        if resp is None or resp.status_code != 200:
            if self._logger:
                self._logger.warning("Trigger %s returned %s", trigger_url, getattr(resp, "status_code", "no response"))
            return False
        report_meta = resp.json()
        report_id = report_meta["id"]
        if self._logger:
            self._logger.info("Report triggered: id=%s, name=%s", report_id, report_meta.get("name"))

        status_url = f"/api/private/companies/{cid}/reports"
        for i in range(max_polls):
            status_resp = self.request_with_retry("GET", status_url)
            if status_resp is None:
                break
            status_resp.raise_for_status()
            report = next((r for r in status_resp.json() if r["id"] == report_id), None)
            if report is None:
                break
            lock_status = report.get("lockState", {}).get("lockStatus", "")
            if self._logger:
                self._logger.info("Poll %d/%d: lockStatus=%s", i + 1, max_polls, lock_status)
            if lock_status != "PROCESSING":
                break
            time.sleep(poll_interval)

        dl_resp = self.request_with_retry("GET", f"/api/private/companies/{cid}/reports/{report_id}")
        if dl_resp is None:
            return False
        dl_resp.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(dl_resp.content)
        return True

    def teardown_after_execution(self, context: dg.InitResourceContext) -> None:
        """Release the authenticated client and clear token data."""
        if self._httpx_client is not None:
            self._httpx_client.close()
        self._httpx_client = None
        self._token = {}
        self._logger = None
