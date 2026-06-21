from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests
from atproto import Client, models

from config import AppConfig

METADATA_TIMEOUT_SECONDS = 10
USER_AGENT = "gemma-tweet-bot/1.0"
MAX_THUMBNAIL_BYTES = 1_000_000


@dataclass(frozen=True)
class PublishedBlueskyPost:
    uri: str
    cid: str
    url: str


@dataclass(frozen=True)
class LinkCardMetadata:
    url: str
    title: str
    description: str
    image_url: str | None = None


class LinkMetadataParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.in_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True
            return
        if tag.lower() != "meta":
            return

        attr_map = {name.lower(): value or "" for name, value in attrs}
        key = attr_map.get("property") or attr_map.get("name")
        content = attr_map.get("content")
        if key and content:
            self.meta[key.lower()] = content.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)

    def get(self, *keys: str) -> str | None:
        for key in keys:
            value = self.meta.get(key.lower())
            if value:
                return value
        return None

    def title(self) -> str | None:
        meta_title = self.get("og:title", "twitter:title")
        if meta_title:
            return meta_title
        title = " ".join(part.strip() for part in self.title_parts if part.strip())
        return title or None

    def description(self) -> str | None:
        return self.get("og:description", "twitter:description", "description")

    def image_url(self) -> str | None:
        image = self.get("og:image", "twitter:image", "twitter:image:src")
        if not image:
            return None
        return urljoin(self.base_url, image)


def build_bluesky_post_url(handle: str, uri: str) -> str:
    record_key = uri.rstrip("/").split("/")[-1]
    return f"https://bsky.app/profile/{handle}/post/{record_key}"


def clean_card_text(value: str | None, fallback: str) -> str:
    cleaned = " ".join((value or "").split())
    if cleaned:
        return cleaned[:1000]
    return " ".join(fallback.split())[:1000]


def fetch_link_card_metadata(
    url: str,
    *,
    fallback_title: str,
    fallback_description: str,
    timeout_seconds: int = METADATA_TIMEOUT_SECONDS,
) -> LinkCardMetadata:
    title = fallback_title
    description = fallback_description
    image_url: str | None = None
    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        parser = LinkMetadataParser(response.url or url)
        parser.feed(response.text)
        title = parser.title() or fallback_title
        description = parser.description() or fallback_description
        image_url = parser.image_url()
    except Exception:
        pass

    return LinkCardMetadata(
        url=url,
        title=clean_card_text(title, fallback_title or url),
        description=clean_card_text(description, fallback_description or title or url),
        image_url=image_url,
    )


def upload_thumbnail_best_effort(
    client: Client,
    image_url: str | None,
    *,
    timeout_seconds: int = METADATA_TIMEOUT_SECONDS,
) -> object | None:
    if not image_url:
        return None
    try:
        response = requests.get(
            image_url,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type and not content_type.startswith("image/"):
            return None
        image_bytes = response.content
        if not image_bytes or len(image_bytes) > MAX_THUMBNAIL_BYTES:
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
