from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from article_links import build_article_link_entry, update_article_links_page
from bluesky_publisher import post_to_bluesky
from cloudinary_uploader import upload_image_to_cloudinary
from config import AppConfig
from instagram_infographic import InfographicPlan, render_instagram_infographic
from instagram_image import render_instagram_image
from instagram_publisher import publish_instagram_image
from logger import PlatformLogResult
from news_fetcher import NewsItem
from publisher import build_post_text, build_post_text_without_url, post_tweet_to_x
from run_state import RunStateStore, content_hash


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


def existing_publication_result(
    state_store: RunStateStore | None,
    *,
    fingerprint: str,
    platform: str,
) -> PlatformLogResult | None:
    if state_store is None:
        return None
    state = state_store.find_platform_state(fingerprint, platform)
    return publication_result_from_state(platform, state)


def publication_result_from_state(
    platform: str, state: dict[str, Any] | None
) -> PlatformLogResult | None:
    if not state:
        return None
    if state.get("status") == "published":
        status = "already published"
    elif state.get("status") == "publishing":
        status = "publication uncertain"
    else:
        return None
    return PlatformLogResult(
        platform=platform,
        status=status,
        url=state.get("url"),
        identifier=state.get("identifier"),
        error=(
            "A previous attempt may have reached the platform; reconcile before retrying."
            if status == "publication uncertain"
            else None
        ),
    )


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
    instagram_infographic_plan: InfographicPlan | None = None,
    run_id: str | None = None,
    state_store: RunStateStore | None = None,
) -> PublishOutcome:
    news_url = news_item.link if news_item else None
    results: list[PlatformLogResult] = []
    success_count = 0

    if config.post_to_bluesky:
        bluesky_text = build_post_text_without_url(tweet)
        bluesky_fingerprint = content_hash(
            "Bluesky",
            bluesky_text,
            news_url,
            news_item.title if news_item else None,
            news_item.summary if news_item else None,
        )
        prior = existing_publication_result(
            state_store, fingerprint=bluesky_fingerprint, platform="Bluesky"
        )
        if not prior and state_store and run_id:
            prior = publication_result_from_state(
                "Bluesky",
                state_store.claim_platform(
                    run_id, platform="Bluesky", fingerprint=bluesky_fingerprint
                ),
            )
        if prior:
            if prior.status != "publication uncertain":
                success_count += 1
            results.append(prior)
        else:
            try:
                published = post_to_bluesky(
                    config,
                    bluesky_text,
                    news_url=news_url,
                    news_title=news_item.title if news_item else None,
                    news_summary=news_item.summary if news_item else None,
                )
                if state_store and run_id:
                    try:
                        state_store.record_platform_success(
                            run_id,
                            platform="Bluesky",
                            fingerprint=bluesky_fingerprint,
                            identifier=published.uri,
                            url=published.url,
                        )
                    except Exception as state_exc:
                        results.append(
                            PlatformLogResult(
                                platform="Bluesky",
                                status="publication uncertain",
                                error=f"Published, but durable state update failed: {state_exc}",
                            )
                        )
                    else:
                        success_count += 1
                        results.append(
                            PlatformLogResult(
                                platform="Bluesky",
                                status="published",
                                url=published.url,
                                identifier=published.uri,
                            )
                        )
                else:
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
                current = (
                    state_store.find_platform_state(bluesky_fingerprint, "Bluesky")
                    if state_store
                    else None
                )
                if current and current.get("status") == "publishing":
                    results.append(
                        PlatformLogResult(
                            platform="Bluesky",
                            status="publication uncertain",
                            error=str(exc),
                        )
                    )
                else:
                    results.append(
                        PlatformLogResult(platform="Bluesky", status="failed", error=str(exc))
                    )

    if config.post_to_instagram:
        cloudinary_url: str | None = None
        instagram_fingerprint = content_hash(
            "Instagram",
            instagram_caption,
            tweet,
            config.instagram_image_renderer,
            news_url,
            news_item.title if news_item else None,
            news_item.summary if news_item else None,
            repr(asdict(instagram_infographic_plan))
            if instagram_infographic_plan
            else None,
        )
        prior = existing_publication_result(
            state_store, fingerprint=instagram_fingerprint, platform="Instagram"
        )
        if not prior and state_store and run_id:
            prior = publication_result_from_state(
                "Instagram",
                state_store.claim_platform(
                    run_id, platform="Instagram", fingerprint=instagram_fingerprint
                ),
            )
        if prior:
            if prior.status != "publication uncertain":
                success_count += 1
            results.append(prior)
        else:
            try:
                if instagram_caption is None:
                    raise RuntimeError("Instagram caption was not generated.")
                if config.instagram_image_renderer == "infographic":
                    if instagram_infographic_plan is None:
                        raise RuntimeError("Instagram infographic plan was not generated.")
                    image_path = render_instagram_infographic(
                        instagram_infographic_plan,
                        image_output_path(config, topic),
                    )
                else:
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
                if state_store and run_id:
                    try:
                        state_store.record_platform_success(
                            run_id,
                            platform="Instagram",
                            fingerprint=instagram_fingerprint,
                            identifier=published.media_id,
                            url=published.url,
                        )
                    except Exception as state_exc:
                        results.append(
                            PlatformLogResult(
                                platform="Instagram",
                                status="publication uncertain",
                                error=f"Published, but durable state update failed: {state_exc}",
                            )
                        )
                    else:
                        success_count += 1
                        results.append(
                            PlatformLogResult(
                                platform="Instagram",
                                status="published",
                                url=published.url,
                                identifier=published.media_id,
                            )
                        )
                else:
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
                current = (
                    state_store.find_platform_state(instagram_fingerprint, "Instagram")
                    if state_store
                    else None
                )
                if current and current.get("status") == "publishing":
                    results.append(
                        PlatformLogResult(
                            platform="Instagram",
                            status="publication uncertain",
                            error=error_message,
                        )
                    )
                else:
                    results.append(
                        PlatformLogResult(
                            platform="Instagram",
                            status="failed",
                            error=error_message,
                        )
                    )

    if config.post_to_x:
        x_text = build_post_text(tweet, news_url)
        x_fingerprint = content_hash("X", x_text)
        prior = existing_publication_result(
            state_store, fingerprint=x_fingerprint, platform="X"
        )
        if not prior and state_store and run_id:
            prior = publication_result_from_state(
                "X",
                state_store.claim_platform(
                    run_id, platform="X", fingerprint=x_fingerprint
                ),
            )
        if prior:
            if prior.status != "publication uncertain":
                success_count += 1
            results.append(prior)
        else:
            try:
                published = post_tweet_to_x(config, tweet, news_url=news_url)
                if state_store and run_id:
                    try:
                        state_store.record_platform_success(
                            run_id,
                            platform="X",
                            fingerprint=x_fingerprint,
                            identifier=published.tweet_id,
                            url=published.url,
                        )
                    except Exception as state_exc:
                        results.append(
                            PlatformLogResult(
                                platform="X",
                                status="publication uncertain",
                                error=f"Published, but durable state update failed: {state_exc}",
                            )
                        )
                    else:
                        success_count += 1
                        results.append(
                            PlatformLogResult(
                                platform="X",
                                status="published",
                                url=published.url,
                                identifier=published.tweet_id,
                            )
                        )
                else:
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
                current = (
                    state_store.find_platform_state(x_fingerprint, "X")
                    if state_store
                    else None
                )
                if current and current.get("status") == "publishing":
                    results.append(
                        PlatformLogResult(
                            platform="X",
                            status="publication uncertain",
                            error=str(exc),
                        )
                    )
                else:
                    results.append(
                        PlatformLogResult(platform="X", status="failed", error=str(exc))
                    )

    return PublishOutcome(results=results, success_count=success_count)
