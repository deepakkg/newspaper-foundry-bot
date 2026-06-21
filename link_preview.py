from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests


METADATA_TIMEOUT_SECONDS = 10
USER_AGENT = "gemma-tweet-bot/1.0"
MAX_THUMBNAIL_BYTES = 1_000_000


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


def fetch_thumbnail_bytes(
    image_url: str | None,
    *,
    timeout_seconds: int = METADATA_TIMEOUT_SECONDS,
) -> bytes | None:
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
        return image_bytes
    except Exception:
        return None
