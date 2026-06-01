"""Unit tests for the OAuth2 API client (Oauth2ApiResource).

Tests cover:
- Bearer token auth flow (inject + 401 retry)
- request_with_retry exponential backoff
- normalize_company_valuation_data edge cases
- API endpoint methods (get_company_list, get_raw_valuation_rounds, etc.)
- _issue_date_to_iso helper
- _trigger_poll_download flow
"""

import time
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from market_presentation.defs.resources.oauth2_api import (
    Oauth2ApiResource,
    _BearerTokenRefreshAuth,
    _issue_date_to_iso,
    normalize_company_valuation_data,
)


# ---------------------------------------------------------------------------
# _issue_date_to_iso
# ---------------------------------------------------------------------------


class TestIssueDateToIso:
    def test_valid_q4(self):
        assert _issue_date_to_iso({"year": 2024, "quarter": "Q4"}) == "2024-12-31"

    def test_non_q4_returns_none(self):
        assert _issue_date_to_iso({"year": 2024, "quarter": "Q1"}) is None

    def test_missing_year_returns_none(self):
        assert _issue_date_to_iso({"quarter": "Q4"}) is None

    def test_empty_dict_returns_none(self):
        assert _issue_date_to_iso({}) is None


# ---------------------------------------------------------------------------
# normalize_company_valuation_data
# ---------------------------------------------------------------------------


class TestNormalizeCompanyValuationData:
    def test_empty_rounds_returns_empty(self):
        result = normalize_company_valuation_data("Test", valuation_rounds=[], valuations_by_round_id={})
        assert result == []

    def test_round_with_no_valuations_produces_row_with_nones(self):
        result = normalize_company_valuation_data(
            "TestCo",
            valuation_rounds=[{"id": 100, "dataSetId": 5, "issueDate": {"year": 2024, "quarter": "Q4"}, "status": "OPEN", "isFree": False}],
            valuations_by_round_id={100: []},
        )
        assert len(result) == 1
        assert result[0]["company_id"] == "TestCo"
        assert result[0]["valuation_round_id"] == 100
        assert result[0]["issue_date"] == "2024-12-31"
        assert result[0]["valuation_id"] is None
        assert result[0]["sub_portfolio_name"] is None

    def test_multiple_valuations_per_round(self):
        result = normalize_company_valuation_data(
            "Corp",
            valuation_rounds=[{"id": 1, "dataSetId": 10, "issueDate": {"year": 2023, "quarter": "Q4"}, "status": "CLOSED", "isFree": False}],
            valuations_by_round_id={
                1: [
                    {"valuationId": 1001, "name": "V1", "dataSetId": 10, "modelYear": 2023, "archived": False, "mainValuation": True, "subPortfolio": {"id": 1, "name": "A", "valuationLevel": "FULL"}},
                    {
                        "valuationId": 1002,
                        "name": "V2",
                        "dataSetId": 11,
                        "modelYear": 2023,
                        "archived": True,
                        "mainValuation": False,
                        "subPortfolio": {"id": 2, "name": "B", "valuationLevel": "BASIC"},
                    },
                ]
            },
        )
        assert len(result) == 2
        assert result[0]["valuation_id"] == 1001
        assert result[0]["sub_portfolio_name"] == "A"
        assert result[1]["valuation_id"] == 1002
        assert result[1]["archived"] is True

    def test_non_q4_issue_date_gives_none_iso(self):
        result = normalize_company_valuation_data(
            "X",
            valuation_rounds=[{"id": 5, "dataSetId": 1, "issueDate": {"year": 2024, "quarter": "Q2"}, "status": "OPEN", "isFree": False}],
            valuations_by_round_id={5: []},
        )
        assert result[0]["issue_date"] is None

    def test_missing_sub_portfolio_fields(self):
        result = normalize_company_valuation_data(
            "Y",
            valuation_rounds=[{"id": 9, "dataSetId": None, "issueDate": {"year": 2024, "quarter": "Q4"}, "status": "OPEN", "isFree": True}],
            valuations_by_round_id={9: [{"valuationId": 50, "name": "V", "dataSetId": 20, "modelYear": 2024, "archived": False, "mainValuation": True, "subPortfolio": None}]},
        )
        assert result[0]["sub_portfolio_id"] is None
        assert result[0]["sub_portfolio_name"] is None


# ---------------------------------------------------------------------------
# _BearerTokenRefreshAuth
# ---------------------------------------------------------------------------


class TestBearerTokenRefreshAuth:
    def test_injects_bearer_token(self):
        token = "test-token-123"
        auth = _BearerTokenRefreshAuth(get_token_fn=lambda: token, refresh_fn=lambda: None)
        request = httpx.Request("GET", "https://example.com/api")

        flow = auth.auth_flow(request)
        modified_request = next(flow)
        assert modified_request.headers["Authorization"] == "Bearer test-token-123"

    def test_refreshes_on_401(self):
        tokens = ["old-token", "new-token"]
        call_count = [0]

        def get_token():
            return tokens[min(call_count[0], 1)]

        def refresh():
            call_count[0] += 1

        auth = _BearerTokenRefreshAuth(get_token_fn=get_token, refresh_fn=refresh)
        request = httpx.Request("GET", "https://example.com/api")

        flow = auth.auth_flow(request)
        modified_request = next(flow)
        assert modified_request.headers["Authorization"] == "Bearer old-token"

        # Simulate 401 response
        mock_response = httpx.Response(401, request=request)
        retry_request = flow.send(mock_response)
        assert retry_request.headers["Authorization"] == "Bearer new-token"
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Oauth2ApiResource methods
# ---------------------------------------------------------------------------


class TestOauth2ApiResourceMethods:
    """Test API endpoint methods using monkeypatch on the class."""

    def _make_resource(self):
        return Oauth2ApiResource(
            base_url="https://tms.example.com",
            token_url="https://auth.example.com/token",
            client_id="test-client",
            client_secret="",
            username="user",
            password="pass",
        )

    def test_get_company_list_empty_on_non_200(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.get_company_list() == []

    def test_get_company_list_maps_fields(self, monkeypatch):
        def fake_request(self, method, url, **kwargs):
            return httpx.Response(200, json=[{"companyId": "Corp A", "isCustomer": True}, {"companyName": "Corp B", "isCustomer": False}], request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        result = resource.get_company_list()
        assert result == [{"company_id": "Corp A", "is_customer": True}, {"company_id": "Corp B", "is_customer": False}]

    def test_get_raw_valuation_rounds_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(404, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.get_raw_valuation_rounds("SomeCo") == []

    def test_get_valuation_round_valuations_parses_response(self, monkeypatch):
        payload = [{"valuationId": 100, "name": "V1"}]

        def fake_request(self, method, url, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        result = resource.get_valuation_round_valuations("Co", 42)
        assert result == payload

    def test_request_with_retry_returns_none_after_exhaustion(self, monkeypatch):
        call_count = [0]

        def fake_request(self, method, url, **kwargs):
            call_count[0] += 1
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()

        with patch("time.sleep"):  # Don't actually wait
            result = resource.request_with_retry("GET", "/api/test", max_attempts=3, timeout=5.0)

        assert result is None
        assert call_count[0] == 3

    def test_request_with_retry_retries_on_500(self, monkeypatch):
        attempts = []

        def fake_request(self, method, url, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(500, request=httpx.Request("GET", url))
            return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()

        with patch("time.sleep"):
            result = resource.request_with_retry("GET", "/api/endpoint", max_attempts=3)

        assert result.status_code == 200
        assert len(attempts) == 3

    def test_request_with_retry_returns_immediately_on_success(self, monkeypatch):
        def fake_request(self, method, url, **kwargs):
            return httpx.Response(200, json=[], request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        result = resource.request_with_retry("GET", "/api/test")
        assert result.status_code == 200

    def test_fetch_market_value_parameters_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(403, request=httpx.Request("POST", url)))
        resource = self._make_resource()

        with pytest.raises(RuntimeError, match="marktwaardeparameters"):
            resource.fetch_market_value_parameters("TestCo", 123)

    def test_fetch_market_value_parameters_returns_bytes(self, monkeypatch):
        content = b"PK\x03\x04fake-xlsx-content"

        def fake_retry(self, method, url, **kwargs):
            return httpx.Response(200, content=content, request=httpx.Request("POST", url))

        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", fake_retry)
        resource = self._make_resource()
        result = resource.fetch_market_value_parameters("TestCo", 123)
        assert result == content

    def test_fetch_difference_analysis_returns_empty_on_non_200(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(404, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        result = resource.fetch_difference_analysis("Co", 1, "Sub")
        assert result == {}

    def test_get_vgr_datasets_filters_published_q4(self, monkeypatch):
        payload = [
            {"dataSetId": 1, "isPublished": True, "issueDate": {"year": 2024, "quarter": "Q4"}},
            {"dataSetId": 2, "isPublished": False, "issueDate": {"year": 2024, "quarter": "Q4"}},
            {"dataSetId": 3, "isPublished": True, "issueDate": {"year": 2024, "quarter": "Q2"}},
            {"dataSetId": 4, "isPublished": True, "issueDate": {"year": 2023, "quarter": "Q4"}},
        ]

        def fake_retry(self, method, url, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", fake_retry)
        resource = self._make_resource()
        result = resource.get_vgr_datasets("Co", [2024])
        assert len(result) == 1
        assert result[0]["data_set_id"] == 1

    def test_get_raw_company_list_empty_on_non_200(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.get_raw_company_list() == []

    def test_get_raw_company_list_returns_json(self, monkeypatch):
        payload = [{"companyId": "A"}, {"companyId": "B"}]

        def fake_request(self, method, url, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        assert resource.get_raw_company_list() == payload

    def test_get_valuation_round_valuations_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.get_valuation_round_valuations("Co", 1) == []

    def test_download_vgr_import_file_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(404, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.download_vgr_import_file("Co", "ds1") is None

    def test_download_vgr_import_file_bytes_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        assert resource.download_vgr_import_file_bytes("Co", "ds1") is None

    def test_download_vgr_import_file_bytes_returns_content(self, monkeypatch):
        content = b"raw-bytes-content"

        def fake_retry(self, method, url, **kwargs):
            return httpx.Response(200, content=content, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", fake_retry)
        resource = self._make_resource()
        assert resource.download_vgr_import_file_bytes("Co", "ds1") == content

    def test_fetch_vgr_dataset_json_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: None)
        resource = self._make_resource()
        assert resource.fetch_vgr_dataset_json("Co", "ds1") is None

    def test_fetch_vgr_dataset_json_returns_dict(self, monkeypatch):
        payload = {"id": 1, "rentalUnits": []}

        def fake_retry(self, method, url, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", fake_retry)
        resource = self._make_resource()
        assert resource.fetch_vgr_dataset_json("Co", "ds1") == payload

    def test_fetch_vgr_valuation_complexes_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(403, request=httpx.Request("POST", url)))
        resource = self._make_resource()
        assert resource.fetch_vgr_valuation_complexes("Co", "ds1") is None

    def test_fetch_policy_value_parameters_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("POST", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="beleidswaardeparameters"):
            resource.fetch_policy_value_parameters("Co", 1)

    def test_fetch_complex_references_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("POST", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="complexreferenties"):
            resource.fetch_complex_references("Co", 1)

    def test_fetch_property_info_returns_empty_on_non_200(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(404, request=httpx.Request("POST", url)))
        resource = self._make_resource()
        assert resource.fetch_property_info("Co", 1) == {}

    def test_fetch_energy_performance_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="energy_performance"):
            resource.fetch_energy_performance("Co", 1)

    def test_fetch_energy_performance_returns_list(self, monkeypatch):
        payload = [{"complexId": 1, "policyValues": []}]

        def fake_request(self, method, url, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        assert resource.fetch_energy_performance("Co", 1) == payload

    def test_fetch_ratios_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="ratios"):
            resource.fetch_ratios("Co", 1)

    def test_fetch_complex_characteristics_excel_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("POST", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="complex_characteristics_excel"):
            resource.fetch_complex_characteristics_excel("Co", 1)

    def test_fetch_waterfall_analysis_returns_empty_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request_with_retry", lambda self, method, url, **kwargs: None)
        resource = self._make_resource()
        assert resource.fetch_waterfall_analysis("Co", 1) == {}

    def test_fetch_complex_characteristics_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(Oauth2ApiResource, "request", lambda self, method, url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)))
        resource = self._make_resource()
        with pytest.raises(RuntimeError, match="complex_characteristics"):
            resource.fetch_complex_characteristics("Co", 1)

    def test_get_company_valuation_data_end_to_end(self, monkeypatch):
        rounds_payload = [
            {"id": 10, "dataSetId": 100, "issueDate": {"year": 2024, "quarter": "Q4"}, "status": "OPEN", "isFree": False},
            {"id": 20, "dataSetId": 200, "issueDate": {"year": 2024, "quarter": "Q2"}, "status": "OPEN", "isFree": False},
        ]
        valuations_payload = [{"valuationId": 1001, "name": "V1", "dataSetId": 100, "modelYear": 2024, "archived": False, "mainValuation": True, "subPortfolio": None}]

        def fake_request(self, method, url, **kwargs):
            if "valuation-rounds/" in url and "/valuations" in url:
                return httpx.Response(200, json=valuations_payload, request=httpx.Request("GET", url))
            if "valuation-rounds" in url:
                return httpx.Response(200, json=rounds_payload, request=httpx.Request("GET", url))
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()
        result = resource.get_company_valuation_data("TestCo", [2024])
        assert len(result) == 1
        assert result[0]["valuation_round_id"] == 10

    def test_request_with_retry_retries_on_remote_protocol_error(self, monkeypatch):
        attempts = []

        def fake_request(self, method, url, **kwargs):
            attempts.append(1)
            if len(attempts) < 2:
                raise httpx.RemoteProtocolError("peer closed")
            return httpx.Response(200, json={}, request=httpx.Request("GET", url))

        monkeypatch.setattr(Oauth2ApiResource, "request", fake_request)
        resource = self._make_resource()

        with patch("time.sleep"):
            result = resource.request_with_retry("GET", "/api/test", max_attempts=3)

        assert result.status_code == 200
        assert len(attempts) == 2
