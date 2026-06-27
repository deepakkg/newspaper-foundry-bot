from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from config import AppConfig
from news_fetcher import NewsItem
from time_formatting import format_datetime_ist


@dataclass(frozen=True)
class ArticleLinkEntry:
    title: str
    source: str
    published_at: str
    url: str
    instagram_media_id: str | None = None
    instagram_url: str | None = None
    added_at: str | None = None


def build_article_link_entry(
    news_item: NewsItem,
    *,
    instagram_media_id: str | None = None,
    instagram_url: str | None = None,
    added_at: str | None = None,
) -> ArticleLinkEntry:
    return ArticleLinkEntry(
        title=" ".join(news_item.title.split()) or "Untitled article",
        source=" ".join(news_item.source.split()) or "Unknown",
        published_at=format_datetime_ist(news_item.published_at),
        url=news_item.link,
        instagram_media_id=instagram_media_id,
        instagram_url=instagram_url,
        added_at=added_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def update_article_links_page(
    config: AppConfig,
    entry: ArticleLinkEntry,
) -> None:
    existing_entries = _read_entries(config.article_links_data_path)
    entries = _upsert_entry(existing_entries, entry, max_items=config.article_links_max_items)

    config.article_links_data_path.parent.mkdir(parents=True, exist_ok=True)
    config.article_links_data_path.write_text(
        json.dumps({"items": [_entry_to_dict(item) for item in entries]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    config.article_links_html_path.write_text(
        _render_html(entries),
        encoding="utf-8",
    )


def _read_entries(path: Path) -> list[ArticleLinkEntry]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries: list[ArticleLinkEntry] = []
    for raw_item in payload.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        url = str(raw_item.get("url", "")).strip()
        if not url:
            continue
        entries.append(
            ArticleLinkEntry(
                title=str(raw_item.get("title", "")).strip() or "Untitled article",
                source=str(raw_item.get("source", "")).strip() or "Unknown",
                published_at=str(raw_item.get("published_at", "")).strip()
                or "Not available",
                url=url,
                instagram_media_id=_optional_text(raw_item.get("instagram_media_id")),
                instagram_url=_optional_text(raw_item.get("instagram_url")),
                added_at=_optional_text(raw_item.get("added_at")),
            )
        )
    return entries


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _upsert_entry(
    existing_entries: list[ArticleLinkEntry],
    new_entry: ArticleLinkEntry,
    *,
    max_items: int,
) -> list[ArticleLinkEntry]:
    seen_new_url = new_entry.url.strip().lower()
    entries = [new_entry]
    for entry in existing_entries:
        if entry.url.strip().lower() == seen_new_url:
            continue
        entries.append(entry)
    return entries[:max_items]


def _entry_to_dict(entry: ArticleLinkEntry) -> dict[str, str]:
    item = {
        "title": entry.title,
        "source": entry.source,
        "published_at": entry.published_at,
        "url": entry.url,
    }
    if entry.instagram_media_id:
        item["instagram_media_id"] = entry.instagram_media_id
    if entry.instagram_url:
        item["instagram_url"] = entry.instagram_url
    if entry.added_at:
        item["added_at"] = entry.added_at
    return item


def _render_html(entries: list[ArticleLinkEntry]) -> str:
    rendered_items = "\n".join(_render_item(entry) for entry in entries)
    if not rendered_items:
        rendered_items = '<p class="empty">No article links yet.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Article Links | Newspaper Foundry</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f4eedf;
      --ink: #171614;
      --muted: #5d7181;
      --line: #23211d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 760px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }}
    header {{
      border-bottom: 2px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    .rule {{
      width: 96px;
      height: 8px;
      background: var(--muted);
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 7vw, 4rem);
      line-height: 1;
      letter-spacing: 0;
    }}
    .lede {{
      margin: 14px 0 0;
      font-family: system-ui, sans-serif;
      color: #494844;
    }}
    ol {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 16px;
    }}
    li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.24);
    }}
    a {{
      color: var(--ink);
      text-decoration-thickness: 1px;
      text-underline-offset: 4px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1.28rem;
      line-height: 1.22;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0;
      color: #55514a;
      font-family: system-ui, sans-serif;
      font-size: 0.92rem;
    }}
    .empty {{
      font-family: system-ui, sans-serif;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="rule" aria-hidden="true"></div>
      <h1>Article Links</h1>
      <p class="lede">News sources behind recent Newspaper Foundry posts.</p>
    </header>
    <ol>
{rendered_items}
    </ol>
  </main>
</body>
</html>
"""


def _render_item(entry: ArticleLinkEntry) -> str:
    title = escape(entry.title)
    source = escape(entry.source)
    published_at = escape(entry.published_at)
    url = escape(entry.url, quote=True)
    return f"""      <li>
        <h2><a href="{url}" rel="noopener noreferrer">{title}</a></h2>
        <p class="meta">Source: {source} · Published At: {published_at}</p>
      </li>"""
