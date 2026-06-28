from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from article_links import build_article_link_entry, update_article_links_page
from bluesky_publisher import post_to_bluesky
from cloudinary_uploader import upload_image_to_cloudinary
from config import AppConfig
from instagram_image import render_instagram_image
from instagram_publisher import publish_instagram_image
from logger import PlatformLogResult
from news_fetcher import NewsItem
from publisher import build_post_text_without_url, post_tweet_to_x


@dataclass(frozen=True)
class PublishOutcome:
    results: list[PlatformLogResult]
    success_count: int


def enabled_platforms(config: AppConfig) -> list[str]:
    platforms: list[str] = []
    if config.post_to_bluesky:
        platforms.append("Bluesky")
    if config.post_to_instagram:
        platforms.append("Instagram")
    if config.post_to_x:
        platforms.append("X")
    return platforms


def image_output_path(config: AppConfig, topic: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(part for part in topic.lower().split() if part) or "post"
    safe_slug = "".join(char for char in slug if char.isalnum() or char == "-")[:60]
    return config.generated_image_dir / f"{timestamp}-{safe_slug}.png"


def format_platform_result(result: PlatformLogResult) -> str:
    detail_parts = [result.status]
    if result.url:
        detail_parts.append(result.url)
    if result.identifier:
        detail_parts.append(result.identifier)
    if result.error:
        detail_parts.append(result.error)
    return f"{result.platform}: {' | '.join(detail_parts)}"


def print_platform_results(results: list[PlatformLogResult]) -> None:
    for result in results:
        print(format_platform_result(result))


def update_article_links_after_instagram_publish(
    config: AppConfig,
    *,
    news_item: NewsItem | None,
    results: list[PlatformLogResult],
) -> None:
    if not config.article_links_enabled or news_item is None:
        return

    instagram_result = next(
        (
            result
            for result in results
            if result.platform == "Instagram" and result.status == "published"
        ),
        None,
    )
    if instagram_result is None:
        return

    try:
        update_article_links_page(
            config,
            build_article_link_entry(
                news_item,
                instagram_media_id=instagram_result.identifier,
                instagram_url=instagram_result.url,
            ),
        )
        print(f"Article link page updated: {news_item.link}")
    except Exception as exc:
        print(f"Warning: Article link page update failed: {exc}")


def publish_enabled_platforms(
    config: AppConfig,
    *,
    topic: str,
    tweet: str,
    final_post_text: str,
    news_item: NewsItem | None,
    instagram_caption: str | None,
) -> PublishOutcome:
    _ = final_post_text
    news_url = news_item.link if news_item else None
    results: list[PlatformLogResult] = []
    success_count = 0

    if config.post_to_bluesky:
        try:
            published = post_to_bluesky(
                config,
                build_post_text_without_url(tweet),
                news_url=news_url,
                news_title=news_item.title if news_item else None,
                news_summary=news_item.summary if news_item else None,
            )
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="Bluesky",
                    status="published",
                    url=published.url,
                    identifier=published.uri,
                )
            )
        except Exception as exc:
            results.append(
                PlatformLogResult(platform="Bluesky", status="failed", error=str(exc))
            )

    if config.post_to_instagram:
        cloudinary_url: str | None = None
        try:
            if instagram_caption is None:
                raise RuntimeError("Instagram caption was not generated.")
            image_path = render_instagram_image(
                tweet,
                image_output_path(config, topic),
            )
            uploaded = upload_image_to_cloudinary(config, image_path)
            cloudinary_url = uploaded.secure_url
            published = publish_instagram_image(
                config,
                image_url=uploaded.secure_url,
                caption=instagram_caption,
            )
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="Instagram",
                    status="published",
                    url=published.url,
                    identifier=published.media_id,
                )
            )
        except Exception as exc:
            error_message = str(exc)
            if cloudinary_url:
                error_message = f"{error_message}; Cloudinary URL: {cloudinary_url}"
            results.append(
                PlatformLogResult(
                    platform="Instagram",
                    status="failed",
                    error=error_message,
                )
            )

    if config.post_to_x:
        try:
            published = post_tweet_to_x(config, tweet, news_url=news_url)
            success_count += 1
            results.append(
                PlatformLogResult(
                    platform="X",
                    status="published",
                    url=published.url,
                    identifier=published.tweet_id,
                )
            )
        except Exception as exc:
            results.append(PlatformLogResult(platform="X", status="failed", error=str(exc)))

    return PublishOutcome(results=results, success_count=success_count)
