from __future__ import annotations

from dataclasses import dataclass

from atproto import Client

from config import AppConfig


@dataclass(frozen=True)
class PublishedBlueskyPost:
    uri: str
    cid: str
    url: str


def build_bluesky_post_url(handle: str, uri: str) -> str:
    record_key = uri.rstrip("/").split("/")[-1]
    return f"https://bsky.app/profile/{handle}/post/{record_key}"


def post_to_bluesky(config: AppConfig, post_text: str) -> PublishedBlueskyPost:
    client = Client(base_url=config.bluesky_service_url)
    try:
        client.login(config.bluesky_handle, config.bluesky_app_password)
        post = client.send_post(post_text)
    except Exception as exc:
        raise RuntimeError(f"Bluesky posting failed: {exc}") from exc

    uri = getattr(post, "uri", None)
    cid = getattr(post, "cid", None)
    if not isinstance(uri, str) or not uri.strip():
        raise RuntimeError("Bluesky API response did not include a post URI.")
    if not isinstance(cid, str) or not cid.strip():
        raise RuntimeError("Bluesky API response did not include a post CID.")

    return PublishedBlueskyPost(
        uri=uri,
        cid=cid,
        url=build_bluesky_post_url(config.bluesky_handle or "", uri),
    )
