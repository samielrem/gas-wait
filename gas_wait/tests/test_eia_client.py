"""Tests for the EIA API client (no live API calls)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.datasets import (
    ACTIVE_DAILY_DATASETS,
    ACTIVE_WEEKLY_DATASETS,
    GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY,
    LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY,
    NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY,
    RETAIL_GASOLINE_REGULAR_US_WEEKLY,
    WTI_CRUDE_SPOT_US_DAILY,
)
from data.eia_client import EIAClient, EIAMissingAPIKeyError, EIARequestError
from data.fetch_datasets import _frequency_check, _series_facet, clean_dataset


class EIAClientTests(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = EIAClient(api_key=None)
            with self.assertRaises(EIAMissingAPIKeyError):
                client.fetch_data("petroleum/pri/gnd/data")

    def test_build_query_params_includes_facets_and_data_columns(self) -> None:
        params = EIAClient._build_query_params(
            frequency="weekly",
            data_columns=["value"],
            facets={"product": ["EPMR"], "duoarea": ["NUS"], "process": ["PTE"]},
            start="2000-01-01",
            end="2024-01-01",
            offset=0,
            length=5000,
            sort_column="period",
            sort_direction="asc",
        )

        self.assertEqual(params["frequency"], "weekly")
        self.assertEqual(params["data[]"], ["value"])
        self.assertEqual(params["facets[product][]"], ["EPMR"])
        self.assertEqual(params["facets[duoarea][]"], ["NUS"])
        self.assertEqual(params["facets[process][]"], ["PTE"])
        self.assertEqual(params["sort[0][column]"], "period")

    def test_response_to_dataframe_parses_dates_and_values(self) -> None:
        payload = {
            "response": {
                "total": 2,
                "frequency": "weekly",
                "dateFormat": "YYYY-MM-DD",
                "data": [
                    {"period": "2024-01-01", "value": "3.25"},
                    {"period": "2024-01-08", "value": "3.30"},
                ],
            }
        }

        df = EIAClient.response_to_dataframe(payload)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "value"], 3.25)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["observation_date"]))

    def test_fetch_all_data_paginates(self) -> None:
        client = EIAClient(api_key="test-key")
        page_one = {
            "response": {
                "total": 3,
                "frequency": "weekly",
                "data": [{"period": "2024-01-01", "value": "1.0"}],
            }
        }
        page_two = {
            "response": {
                "total": 3,
                "frequency": "weekly",
                "data": [
                    {"period": "2024-01-08", "value": "2.0"},
                    {"period": "2024-01-15", "value": "3.0"},
                ],
            }
        }

        with patch.object(client, "_request", side_effect=[page_one, page_two]) as mock_request:
            payload = client.fetch_all_data("petroleum/pri/gnd/data", frequency="weekly")

        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(len(payload["response"]["data"]), 3)

    def test_request_retries_on_retryable_status(self) -> None:
        client = EIAClient(api_key="test-key", max_retries=2, backoff_seconds=0)

        success_response = Mock()
        success_response.ok = True
        success_response.status_code = 200
        success_response.json.return_value = {"response": {"data": []}}

        retry_response = Mock()
        retry_response.ok = False
        retry_response.status_code = 503
        retry_response.text = "Service unavailable"
        retry_response.json.return_value = {"error": "Service unavailable"}

        session = Mock()
        session.get.side_effect = [retry_response, success_response]
        client.session = session

        payload = client._request("petroleum/pri/gnd")
        self.assertEqual(payload["response"]["data"], [])
        self.assertEqual(session.get.call_count, 2)

    def test_request_raises_on_client_error(self) -> None:
        client = EIAClient(api_key="test-key", max_retries=0)

        error_response = Mock()
        error_response.ok = False
        error_response.status_code = 400
        error_response.text = "Bad request"
        error_response.json.return_value = {"error": "Invalid facet"}

        session = Mock()
        session.get.return_value = error_response
        client.session = session

        with self.assertRaises(EIARequestError):
            client._request("petroleum/pri/gnd/data")

    def test_save_raw_json_writes_file(self) -> None:
        payload = {"response": {"data": [{"period": "2024-01-01", "value": "1.0"}]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            EIAClient.save_raw_json(payload, str(path))
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["response"]["data"][0]["value"], "1.0")

    def test_save_raw_json_redacts_api_keys(self) -> None:
        payload = {
            "request": {"params": {"api_key": "test-secret-value"}},
            "response": {
                "data": [{"period": "2024-01-01", "value": "1.0"}],
                "nested": [{"api_key": "another-test-secret"}],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.json"
            EIAClient.save_raw_json(payload, str(path))
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["request"]["params"]["api_key"], "[REDACTED]")
        self.assertEqual(loaded["response"]["nested"][0]["api_key"], "[REDACTED]")
        self.assertEqual(loaded["response"]["data"][0]["value"], "1.0")

    def test_clean_dataset_adds_metadata_columns(self) -> None:
        raw_df = pd.DataFrame(
            {
                "period": ["2024-01-01"],
                "observation_date": pd.to_datetime(["2024-01-01"]),
                "value": [3.25],
            }
        )
        cleaned = clean_dataset(raw_df, RETAIL_GASOLINE_REGULAR_US_WEEKLY)

        self.assertEqual(cleaned.loc[0, "dataset_id"], "retail_gasoline_regular_us_weekly")
        self.assertEqual(cleaned.loc[0, "frequency"], "weekly")
        self.assertEqual(cleaned.loc[0, "units"], "dollars per gallon")

    def test_daily_dataset_definitions_are_verified(self) -> None:
        self.assertEqual(len(ACTIVE_DAILY_DATASETS), 4)
        self.assertEqual(WTI_CRUDE_SPOT_US_DAILY.frequency, "daily")
        self.assertEqual(WTI_CRUDE_SPOT_US_DAILY.facets["series"], ["RWTC"])
        self.assertEqual(WTI_CRUDE_SPOT_US_DAILY.legacy_series_id, "PET.RWTC.D")
        self.assertEqual(
            NY_HARBOR_GASOLINE_REGULAR_SPOT_DAILY.facets["series"],
            ["EER_EPMRU_PF4_Y35NY_DPG"],
        )
        self.assertEqual(
            GULF_COAST_GASOLINE_REGULAR_SPOT_DAILY.facets["series"],
            ["EER_EPMRU_PF4_RGC_DPG"],
        )
        self.assertEqual(
            LA_RBOB_GASOLINE_REGULAR_SPOT_DAILY.facets["series"],
            ["EER_EPMRR_PF4_Y05LA_DPG"],
        )
        self.assertEqual(len(ACTIVE_WEEKLY_DATASETS), 3)

    def test_series_facet_prefers_series_facet_value(self) -> None:
        self.assertEqual(_series_facet(WTI_CRUDE_SPOT_US_DAILY), "RWTC")

    def test_frequency_check_detects_daily_series(self) -> None:
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame({"observation_date": dates, "value": range(10)})
        result = _frequency_check(df, "daily")
        self.assertTrue(result["is_native_frequency"])
        self.assertLessEqual(result["median_gap_days"], 3)

    def test_frequency_check_detects_weekly_series(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="7D")
        df = pd.DataFrame({"observation_date": dates, "value": range(8)})
        result = _frequency_check(df, "weekly")
        self.assertTrue(result["is_native_frequency"])
        self.assertEqual(result["median_gap_days"], 7.0)


if __name__ == "__main__":
    unittest.main()
