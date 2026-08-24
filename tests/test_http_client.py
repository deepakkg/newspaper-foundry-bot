from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from http_client import request_with_retry


class HttpRetryTests(unittest.TestCase):
    def test_retries_safe_transient_response_then_returns_success(self) -> None:
        first = Mock(status_code=503, headers={}, text="busy")
        second = Mock(status_code=200, headers={}, text="ok")
        request = Mock(side_effect=[first, second])

        with patch("http_client.time.sleep") as sleep:
            response = request_with_retry(
                request,
                "https://example.com",
                safe_to_retry=True,
                max_attempts=2,
            )

        self.assertIs(response, second)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()

    def test_does_not_retry_unsafe_operation_after_network_error(self) -> None:
        request = Mock(side_effect=requests.ConnectionError("connection dropped"))

        with patch("http_client.time.sleep") as sleep:
            with self.assertRaises(requests.ConnectionError):
                request_with_retry(
                    request,
                    "https://example.com",
                    safe_to_retry=False,
                    max_attempts=3,
                )

        request.assert_called_once()
        sleep.assert_not_called()

    def test_raises_final_network_error_after_safe_retries(self) -> None:
        request = Mock(side_effect=requests.Timeout("timed out"))

        with patch("http_client.time.sleep"):
            with self.assertRaises(requests.Timeout):
                request_with_retry(
                    request,
                    "https://example.com",
                    safe_to_retry=True,
                    max_attempts=3,
                )

        self.assertEqual(request.call_count, 3)

    def test_clamps_negative_retry_after(self) -> None:
        response = Mock(status_code=429, headers={"Retry-After": "-1"})
        request = Mock(return_value=response)

        with patch("http_client.time.sleep") as sleep:
            request_with_retry(
                request,
                "https://example.com",
                safe_to_retry=True,
                max_attempts=2,
            )

        sleep.assert_called_once_with(0.0)

    def test_ignores_non_finite_retry_after(self) -> None:
        response = Mock(status_code=429, headers={"Retry-After": "NaN"})
        request = Mock(return_value=response)

        with patch("http_client.time.sleep") as sleep:
            request_with_retry(
                request,
                "https://example.com",
                safe_to_retry=True,
                max_attempts=2,
            )

        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
