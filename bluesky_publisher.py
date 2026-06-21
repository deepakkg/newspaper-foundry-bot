from __future__ import annotations

from dataclasses import dataclass

from atproto import Client, models

from config import AppConfig
from link_preview import (
    LinkCardMetadata,
    fetch_link_card_metadata,
    fetch_thumbnail_bytes,
)


@dataclass(frozen=True)
class PublishedBlueskyPost:
    uri: str
    cid: str
    url: str


def build_bluesky_post_url(handle: str, uri: str) -> str:
    record_key = uri.rstrip("/").split("/")[-1]
    return f"https://bsky.app/profile/{handle}/post/{record_key}"


def upload_thumbnail_best_effort(
    client: Client,
    image_url: str | None,
) -> object | None:
    try:
        image_bytes = fetch_thumbnail_bytes(image_url)
        if image_bytes is None:
            return None
        uploaded = client.upload_blob(image_bytes)
        return getattr(uploaded, "blob", None)
    except Exception:
        return None


def build_external_embed(client: Client, metadata: LinkCardMetadata) -> object:
    thumb = upload_thumbnail_best_effort(client, metadata.image_url)
    return models.AppBskyEmbedExternal.Main(
        external=models.AppBskyEmbedExternal.External(
            uri=metadata.url,
            title=metadata.title,
            description=metadata.description,
            thumb=thumb,
        )
    )


def post_to_bluesky(
    config: AppConfig,
    post_text: str,
    *,
    news_url: str | None = None,
    news_title: str | None = None,
    news_summary: str | None = None,
) -> PublishedBlueskyPost:
    client = Client(base_url=config.bluesky_service_url)
    try:
        client.login(config.bluesky_handle, config.bluesky_app_password)
        embed = None
        if news_url:
            metadata = fetch_link_card_metadata(
                news_url,
                fallback_title=news_title or news_url,
                fallback_description=news_summary or news_title or news_url,
            )
            embed = build_external_embed(client, metadata)
        post = client.send_post(post_text, embed=embed)
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
