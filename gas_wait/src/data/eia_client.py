"""EIA Open Data API v2 client.

Documentation: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.eia.gov/v2"
DEFAULT_PAGE_SIZE = 5000
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class EIAClientError(Exception):
    """Base exception for EIA client failures."""


class EIAMissingAPIKeyError(EIAClientError):
    """Raised when EIA_API_KEY is not configured."""


class EIARequestError(EIAClientError):
    """Raised when the EIA API returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class EIAClient:
    """Client for the U.S. Energy Information Administration Open Data API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise EIAMissingAPIKeyError(
                "EIA_API_KEY is not set. Register for a key at "
                "https://www.eia.gov/opendata/register.php and export it as EIA_API_KEY."
            )
        return self.api_key

    @staticmethod
    def _normalize_route(route: str) -> str:
        route = route.strip("/")
        if route.endswith("/data"):
            return route
        return f"{route}/data"

    @staticmethod
    def _build_query_params(
        *,
        frequency: str | None = None,
        data_columns: list[str] | None = None,
        facets: dict[str, list[str]] | None = None,
        start: str | None = None,
        end: str | None = None,
        offset: int | None = None,
        length: int | None = None,
        sort_column: str | None = None,
        sort_direction: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if frequency:
            params["frequency"] = frequency
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if offset is not None:
            params["offset"] = offset
        if length is not None:
            params["length"] = length
        if sort_column:
            params["sort[0][column]"] = sort_column
        if sort_direction:
            params["sort[0][direction]"] = sort_direction

        for column in data_columns or ["value"]:
            params.setdefault("data[]", [])
            if isinstance(params["data[]"], list):
                params["data[]"].append(column)

        for facet_name, facet_values in (facets or {}).items():
            for value in facet_values:
                key = f"facets[{facet_name}][]"
                params.setdefault(key, [])
                if isinstance(params[key], list):
                    params[key].append(value)

        if extra_params:
            params.update(extra_params)

        return params

    def _request(self, route: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        api_key = self._require_api_key()
        url = f"{self.base_url}/{route.lstrip('/')}"
        request_params = {"api_key": api_key, **(params or {})}

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("EIA request failed on attempt %s: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise EIARequestError(f"Network error contacting EIA API: {exc}") from exc

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                logger.warning(
                    "Retryable EIA status %s on attempt %s for route %s",
                    response.status_code,
                    attempt + 1,
                    route,
                )
                time.sleep(self.backoff_seconds * (2**attempt))
                continue

            if not response.ok:
                payload = self._safe_json(response)
                message = self._extract_error_message(payload) or response.text.strip()
                logger.error(
                    "EIA API error %s for route %s: %s",
                    response.status_code,
                    route,
                    message,
                )
                raise EIARequestError(
                    f"EIA API request failed ({response.status_code}): {message}",
                    status_code=response.status_code,
                    payload=payload,
                )

            payload = self._safe_json(response)
            if not isinstance(payload, dict):
                raise EIARequestError("EIA API returned a non-JSON object response.")
            return payload

        raise EIARequestError(f"EIA API request failed after retries: {last_error}")

    @staticmethod
    def _safe_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}

    @staticmethod
    def _extract_error_message(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None

        for key in ("error", "message", "errors"):
            if key in payload and payload[key]:
                return str(payload[key])

        response_obj = payload.get("response")
        if isinstance(response_obj, dict):
            for key in ("error", "message", "errors"):
                if key in response_obj and response_obj[key]:
                    return str(response_obj[key])
        return None

    def get_route_metadata(self, route: str) -> dict[str, Any]:
        """Fetch metadata for a route without requesting /data values."""
        metadata_route = route.strip("/")
        if metadata_route.endswith("/data"):
            metadata_route = metadata_route[: -len("/data")]
        return self._request(metadata_route)

    def fetch_data(
        self,
        route: str,
        *,
        frequency: str | None = None,
        data_columns: list[str] | None = None,
        facets: dict[str, list[str]] | None = None,
        start: str | None = None,
        end: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort_column: str = "period",
        sort_direction: str = "asc",
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch one page of time-series data from an EIA /data route."""
        normalized_route = self._normalize_route(route)
        params = self._build_query_params(
            frequency=frequency,
            data_columns=data_columns,
            facets=facets,
            start=start,
            end=end,
            offset=0,
            length=page_size,
            sort_column=sort_column,
            sort_direction=sort_direction,
            extra_params=extra_params,
        )
        return self._request(normalized_route, params)

    def fetch_all_data(
        self,
        route: str,
        *,
        frequency: str | None = None,
        data_columns: list[str] | None = None,
        facets: dict[str, list[str]] | None = None,
        start: str | None = None,
        end: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort_column: str = "period",
        sort_direction: str = "asc",
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch all pages for a query, respecting EIA's 5,000 row JSON limit."""
        normalized_route = self._normalize_route(route)
        all_rows: list[dict[str, Any]] = []
        offset = 0
        combined_response: dict[str, Any] | None = None

        while True:
            params = self._build_query_params(
                frequency=frequency,
                data_columns=data_columns,
                facets=facets,
                start=start,
                end=end,
                offset=offset,
                length=page_size,
                sort_column=sort_column,
                sort_direction=sort_direction,
                extra_params=extra_params,
            )
            payload = self._request(normalized_route, params)
            response_obj = payload.get("response", {})
            rows = response_obj.get("data", [])
            total_raw = response_obj.get("total", len(rows))
            try:
                total = int(total_raw)
            except (TypeError, ValueError):
                total = len(rows)

            if combined_response is None:
                combined_response = payload
                combined_response["response"]["data"] = all_rows
            all_rows.extend(rows)

            logger.info(
                "Fetched %s rows (%s/%s) from %s",
                len(rows),
                len(all_rows),
                total,
                normalized_route,
            )

            if len(all_rows) >= total or not rows:
                break

            offset += len(rows)

        if combined_response is None:
            return {"response": {"data": [], "total": 0}}

        combined_response["response"]["data"] = all_rows
        combined_response["response"]["total"] = len(all_rows)
        return combined_response

    @staticmethod
    def response_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
        """Convert an EIA time-series response into a pandas DataFrame."""
        response_obj = payload.get("response", {})
        rows = response_obj.get("data", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "period" in df.columns:
            df["observation_date"] = pd.to_datetime(df["period"], errors="coerce")
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        metadata = {
            "frequency": response_obj.get("frequency"),
            "date_format": response_obj.get("dateFormat"),
            "total": response_obj.get("total"),
        }
        for key, value in metadata.items():
            if value is not None:
                df[key] = value

        return df

    @staticmethod
    def save_raw_json(payload: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
