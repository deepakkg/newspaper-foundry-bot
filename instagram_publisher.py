from __future__ import annotations

from dataclasses import dataclass

import requests

from config import AppConfig


INVALID_TOKEN_MESSAGE = (
    "Instagram access token is invalid or malformed. "
    "Check INSTAGRAM_ACCESS_TOKEN in GitHub Secrets."
)


@dataclass(frozen=True)
class PublishedInstagramPost:
    media_id: str
    url: str | None = None


def _graph_base_url(config: AppConfig) -> str:
    version = config.instagram_graph_api_version.strip().lstrip("/")
    return f"https://graph.facebook.com/{version}"


def _extract_error(payload: dict[str, object]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return str(payload)


def _format_error(message: str) -> str:
    normalized = message.lower()
    if (
        "invalid oauth access token" in normalized
        or "cannot parse access token" in normalized
    ):
        return f"{INVALID_TOKEN_MESSAGE} Original error: {message}"
    return message


def _read_json_response(response: requests.Response, context: str) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Instagram {context} returned a non-JSON response.") from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Instagram {context} failed: {_format_error(_extract_error(payload))}"
        )
    return payload


def publish_instagram_image(
    config: AppConfig,
    *,
    image_url: str,
    caption: str,
) -> PublishedInstagramPost:
    base_url = _graph_base_url(config)
    media_response = requests.post(
        f"{base_url}/{config.instagram_account_id}/media",
        data={
            "image_url": image_url,
            "caption": caption,
            "access_token": config.instagram_access_token,
        },
        timeout=config.timeout_seconds,
    )
    media_payload = _read_json_response(media_response, "media container creation")
    creation_id = media_payload.get("id")
    if not isinstance(creation_id, str) or not creation_id.strip():
        raise RuntimeError("Instagram media container response did not include an ID.")

    publish_response = requests.post(
        f"{base_url}/{config.instagram_account_id}/media_publish",
        data={
            "creation_id": creation_id,
            "access_token": config.instagram_access_token,
        },
        timeout=config.timeout_seconds,
    )
    publish_payload = _read_json_response(publish_response, "media publish")
    media_id = publish_payload.get("id")
    if not isinstance(media_id, str) or not media_id.strip():
        raise RuntimeError("Instagram publish response did not include a media ID.")

    url = publish_payload.get("permalink")
    return PublishedInstagramPost(
        media_id=media_id,
        url=url if isinstance(url, str) and url.strip() else None,
    )
