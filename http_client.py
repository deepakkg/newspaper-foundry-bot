from __future__ import annotations

import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import math
from typing import Any

import requests


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def _retry_delay(response: Any, attempt: int, *, max_delay_seconds: float) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after:
        try:
            numeric_delay = float(retry_after)
            if math.isfinite(numeric_delay):
                return min(max(numeric_delay, 0.0), max_delay_seconds)
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return min(max(seconds, 0.0), max_delay_seconds)
        except (TypeError, ValueError, OverflowError):
            pass
    return min(2**attempt, max_delay_seconds)


def request_with_retry(
    request_fn: Callable[..., requests.Response],
    url: str,
    *,
    safe_to_retry: bool,
    max_attempts: int = 3,
    max_delay_seconds: float = 8.0,
    **kwargs: Any,
) -> requests.Response:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")

    for attempt in range(max_attempts):
        try:
            response = request_fn(url, **kwargs)
        except requests.RequestException:
            if not safe_to_retry or attempt == max_attempts - 1:
                raise
            time.sleep(min(2**attempt, max_delay_seconds))
            continue

        if (
            safe_to_retry
            and response.status_code in RETRYABLE_STATUS_CODES
            and attempt < max_attempts - 1
        ):
            time.sleep(_retry_delay(response, attempt, max_delay_seconds=max_delay_seconds))
            continue
        return response

    raise RuntimeError("HTTP retry loop ended unexpectedly")
