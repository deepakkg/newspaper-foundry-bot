from __future__ import annotations

import hashlib
import json
import random
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from config import AppConfig
from generator import extract_response_text, request_completion
from instagram_hashtags import BOT_HASHTAG
from instagram_image import (
    CONTROL_PATTERN,
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    INK,
    KEYCAP_MARK_PATTERN,
    KEYCAP_SEQUENCE_PATTERN,
    TERMINAL_PUNCTUATION,
    UNSUPPORTED_SYMBOL_PATTERN,
    URL_PATTERN,
    VARIATION_SELECTOR_PATTERN,
    ZERO_WIDTH_PATTERN,
    _load_font,
    _load_monospace_font,
    _text_width,
)
from news_fetcher import NewsItem


INFOGRAPHIC_WIDTH = 1080
INFOGRAPHIC_HEIGHT = 1350
BACKGROUND = (246, 239, 224)
MUTED_BLUE = (81, 107, 128)
LIGHT_BLUE = (216, 226, 232)
PAPER_EDGE = (42, 40, 34)
WARM_GRAY = (225, 218, 204)
ALLOWED_STYLES = {"foundry_editorial", "foundry_schematic", "foundry_briefing"}
TECHNICAL_KEYWORDS = {
    "ai",
    "agent",
    "agents",
    "api",
    "business",
    "cloud",
    "data",
    "defense",
    "engineer",
    "engineering",
    "finance",
    "market",
    "saas",
    "security",
    "software",
    "startup",
    "tech",
    "token",
}
QUOTE_MARKERS = {"direct", "quote", "post", "on-demand"}
INCOMPLETE_END_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "against",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "with",
}


@dataclass(frozen=True)
class InfographicBlock:
    label: str
    text: str


@dataclass(frozen=True)
class InfographicPlan:
    title: str
    blocks: list[InfographicBlock]
    takeaway: str
    style: str
    source_kind: str


@dataclass(frozen=True)
class TextFit:
    font: ImageFont.ImageFont
    lines: list[str]
    line_gap: int


def build_instagram_infographic_plan(
    client: OpenAI,
    config: AppConfig,
    *,
    topic: str,
    tone: str,
    post_text: str,
    news_item: NewsItem | None,
    is_direct_post: bool,
) -> InfographicPlan:
    style = select_infographic_style(
        config.instagram_infographic_style,
        topic=topic,
        tone=tone,
        post_text=post_text,
        news_item=news_item,
        is_direct_post=is_direct_post,
    )
    fallback = fallback_infographic_plan(
        topic=topic,
        tone=tone,
        post_text=post_text,
        news_item=news_item,
        style=style,
        is_direct_post=is_direct_post,
    )
    if is_direct_post:
        return fallback

    try:
        response = request_completion(
            config=config,
            client=client,
            prompt=build_infographic_prompt(topic, tone, post_text, news_item),
        )
        parsed = parse_infographic_response(extract_response_text(response))
        return validate_infographic_plan(
            parsed,
            style=style,
            source_kind="news" if news_item else "topic",
        )
    except Exception:
        return fallback


def select_infographic_style(
    configured_style: str,
    *,
    topic: str,
    tone: str,
    post_text: str,
    news_item: NewsItem | None,
    is_direct_post: bool,
) -> str:
    if configured_style in ALLOWED_STYLES:
        return configured_style
    if is_direct_post or tone.lower() in QUOTE_MARKERS:
        return "foundry_editorial"

    haystack = " ".join(
        part
        for part in (
            topic,
            tone,
            post_text,
            news_item.title if news_item else "",
            news_item.summary if news_item else "",
        )
        if part
    ).lower()
    tokens = set(re.findall(r"[a-z0-9]+", haystack))
    if tokens & TECHNICAL_KEYWORDS:
        return "foundry_schematic"
    return "foundry_briefing"


def build_infographic_prompt(
    topic: str,
    tone: str,
    post_text: str,
    news_item: NewsItem | None,
) -> str:
    news_block = ""
    if news_item:
        news_block = f"""
News context:
- Title: {news_item.title}
- Source: {news_item.source}
- Summary: {news_item.summary or news_item.title}
"""
    return f"""Create structured content for one Instagram infographic image.
Topic: {topic}
Tone: {tone}
Post text:
{post_text}
{news_block}

Return only JSON in this exact shape:
{{
  "title": "short title",
  "blocks": [
    {{"label": "short label", "text": "compact block text"}}
  ],
  "takeaway": "short takeaway"
}}

Rules:
- Use 2 to 4 blocks.
- Title max 72 characters.
- Each label max 24 characters.
- Each block text max 90 characters; shorter is better for schematic layouts.
- Takeaway max 120 characters.
- Every block and takeaway must be a complete standalone sentence.
- Do not end a block or takeaway with a conjunction, preposition, or dangling phrase.
- Do not include URLs, hashtags, emojis, markdown, explanations, or labels outside JSON.
- Do not invent facts beyond the post text and news context.
"""


def parse_infographic_response(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Infographic response must be a JSON object.")
    return payload


def validate_infographic_plan(
    payload: dict[str, object],
    *,
    style: str,
    source_kind: str,
) -> InfographicPlan:
    title = _shorten_heading(clean_infographic_text(str(payload.get("title", ""))), 72)
    raw_blocks = payload.get("blocks")
    if not title or not isinstance(raw_blocks, list):
        raise ValueError("Infographic response is missing title or blocks.")

    blocks: list[InfographicBlock] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            continue
        label = _shorten_heading(clean_infographic_text(str(item.get("label", ""))), 24)
        text = _shorten(clean_infographic_text(str(item.get("text", ""))), 90)
        if label and text:
            blocks.append(InfographicBlock(label=label, text=text))
    blocks = blocks[:4]
    if len(blocks) < 2:
        raise ValueError("Infographic response must include at least two valid blocks.")

    takeaway = _shorten(clean_infographic_text(str(payload.get("takeaway", ""))), 120)
    if not takeaway:
        raise ValueError("Infographic response is missing takeaway.")

    return InfographicPlan(
        title=title,
        blocks=blocks,
        takeaway=takeaway,
        style=style,
        source_kind=source_kind,
    )


def fallback_infographic_plan(
    *,
    topic: str,
    tone: str,
    post_text: str,
    news_item: NewsItem | None,
    style: str,
    is_direct_post: bool,
) -> InfographicPlan:
    cleaned_post = clean_infographic_text(post_text)
    if is_direct_post:
        return InfographicPlan(
            title="Foundry Note",
            blocks=[
                InfographicBlock(label="Post", text=_ensure_terminal(cleaned_post)),
                InfographicBlock(label="Tone", text=_ensure_terminal(tone.title())),
            ],
            takeaway=_ensure_terminal(cleaned_post),
            style=style,
            source_kind="direct_post",
        )

    if news_item:
        summary = clean_infographic_text(news_item.summary or news_item.title)
        return InfographicPlan(
            title=_shorten_heading(clean_infographic_text(news_item.title), 72),
            blocks=[
                InfographicBlock(label="Source", text=_shorten(news_item.source, 110)),
                InfographicBlock(label="Context", text=_shorten(_ensure_terminal(summary), 110)),
                InfographicBlock(label="Take", text=_shorten(_ensure_terminal(cleaned_post), 110)),
            ],
            takeaway=_shorten(_ensure_terminal(cleaned_post), 120),
            style=style,
            source_kind="news",
        )

    return InfographicPlan(
        title=_shorten_heading(topic.title(), 72),
        blocks=[
            InfographicBlock(label="Topic", text=_shorten(_ensure_terminal(topic), 110)),
            InfographicBlock(label="Tone", text=_shorten(_ensure_terminal(tone), 110)),
            InfographicBlock(label="Take", text=_shorten(_ensure_terminal(cleaned_post), 110)),
        ],
        takeaway=_shorten(_ensure_terminal(cleaned_post), 120),
        style=style,
        source_kind="topic",
    )


def clean_infographic_text(text: str) -> str:
    without_keycaps = KEYCAP_SEQUENCE_PATTERN.sub(" ", text)
    without_urls = URL_PATTERN.sub(" ", without_keycaps)
    without_hashtags = HASHTAG_PATTERN.sub(" ", without_urls)
    without_emojis = EMOJI_PATTERN.sub(" ", without_hashtags)
    without_emoji_marks = VARIATION_SELECTOR_PATTERN.sub(" ", without_emojis)
    without_emoji_marks = ZERO_WIDTH_PATTERN.sub(" ", without_emoji_marks)
    without_emoji_marks = KEYCAP_MARK_PATTERN.sub(" ", without_emoji_marks)
    without_symbols = UNSUPPORTED_SYMBOL_PATTERN.sub(" ", without_emoji_marks)
    without_controls = CONTROL_PATTERN.sub(" ", without_symbols)
    printable_chars = [
        char
        for char in without_controls
        if unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    ]
    cleaned = " ".join("".join(printable_chars).split())
    return cleaned.rstrip(" ,;:-")


def _ensure_terminal(text: str) -> str:
    cleaned = text.strip()
    if cleaned and not cleaned.endswith(TERMINAL_PUNCTUATION):
        return f"{cleaned}."
    return cleaned


def _last_word(text: str) -> str:
    match = re.search(r"([A-Za-z0-9']+)[.!?]?$", text.strip())
    return match.group(1).lower() if match else ""


def _ends_with_incomplete_word(text: str) -> bool:
    return _last_word(text) in INCOMPLETE_END_WORDS


def _shorten_heading(text: str, limit: int) -> str:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    shortened = cleaned[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or cleaned[:limit].rstrip(" ,;:-")


def _shorten(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit and not _ends_with_incomplete_word(cleaned):
        return _ensure_terminal(cleaned)

    bounded = cleaned[:limit].rstrip(" ,;:-")
    sentence_matches = list(re.finditer(r"[.!?]", bounded))
    for match in reversed(sentence_matches):
        candidate = bounded[: match.end()].strip()
        if candidate and not _ends_with_incomplete_word(candidate):
            return _ensure_terminal(candidate)

    words = bounded.split()
    while words and _ends_with_incomplete_word(" ".join(words)):
        words.pop()
    shortened = " ".join(words).rstrip(" ,;:-")
    if not shortened:
        shortened = cleaned[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return _ensure_terminal(shortened)


def render_instagram_infographic(
    plan: InfographicPlan,
    output_path: Path,
    *,
    footer_text: str = BOT_HASHTAG,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (INFOGRAPHIC_WIDTH, INFOGRAPHIC_HEIGHT), BACKGROUND)
    _add_texture(image, f"{plan.title} {plan.takeaway}")
    draw = ImageDraw.Draw(image)

    style = plan.style
    if style == "foundry_schematic" and not _schematic_can_fit_full_plan(draw, plan):
        style = "foundry_briefing"

    if style == "foundry_schematic":
        _draw_schematic(draw, plan)
    elif style == "foundry_briefing":
        _draw_briefing(draw, plan)
    else:
        _draw_editorial(draw, plan)

    _draw_portrait_footer(draw, footer_text)
    image.save(output_path, format="PNG")
    return output_path


def _add_texture(image: Image.Image, seed_text: str) -> None:
    pixels = image.load()
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    for _ in range(19000):
        x = rng.randrange(0, INFOGRAPHIC_WIDTH)
        y = rng.randrange(0, INFOGRAPHIC_HEIGHT)
        delta = rng.randrange(-7, 8)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )


def _draw_editorial(draw: ImageDraw.ImageDraw, plan: InfographicPlan) -> None:
    _draw_outer_frame(draw)
    draw.rectangle((92, 104, 252, 114), fill=MUTED_BLUE)
    _draw_title(draw, plan.title, y=152, font_size=58, max_width=880)
    block_top = 370
    block_height = 158
    for index, block in enumerate(plan.blocks):
        y = block_top + index * (block_height + 28)
        _draw_editorial_block(draw, y, block, index + 1)
    _draw_takeaway(draw, plan.takeaway, y=1068)


def _draw_schematic(draw: ImageDraw.ImageDraw, plan: InfographicPlan) -> None:
    _draw_outer_frame(draw)
    draw.rectangle((88, 102, 260, 110), fill=MUTED_BLUE)
    _draw_title(draw, plan.title, y=148, font_size=52, max_width=870)
    axis_x = 138
    block_top, block_gap, block_height, last_block_bottom, takeaway_y = _schematic_layout(
        len(plan.blocks)
    )
    draw.line((axis_x, 374, axis_x, min(1000, last_block_bottom + 34)), fill=MUTED_BLUE, width=4)
    for index, block in enumerate(plan.blocks):
        y = block_top + index * (block_height + block_gap)
        dot_y = y + 28
        draw.ellipse((axis_x - 11, dot_y - 11, axis_x + 11, dot_y + 11), fill=MUTED_BLUE)
        draw.line((axis_x + 18, dot_y, 212, dot_y), fill=MUTED_BLUE, width=2)
        _draw_schematic_block(draw, y, block, height=block_height)
    _draw_takeaway(draw, plan.takeaway, y=takeaway_y)


def _schematic_layout(block_count: int) -> tuple[int, int, int, int, int]:
    block_top = 340
    block_gap = 24
    safe_count = max(1, block_count)
    base_takeaway_y = 1080
    available_height = base_takeaway_y - block_top - 56
    block_height = min(
        180,
        max(150, (available_height - block_gap * (safe_count - 1)) // safe_count),
    )
    last_block_bottom = block_top + safe_count * block_height + (safe_count - 1) * block_gap
    takeaway_y = max(1048, min(1096, last_block_bottom + 46))
    return block_top, block_gap, block_height, last_block_bottom, takeaway_y


def _schematic_can_fit_full_plan(draw: ImageDraw.ImageDraw, plan: InfographicPlan) -> bool:
    _block_top, _block_gap, block_height, _last_block_bottom, _takeaway_y = _schematic_layout(
        len(plan.blocks)
    )
    block_text_height = max(72, block_height - 74)
    return all(
        _text_fits_without_shortening(
            draw,
            block.text,
            width=650,
            height=block_text_height,
            start_font_size=34,
            min_font_size=25,
            max_lines=3,
        )
        for block in plan.blocks
    )


def _draw_briefing(draw: ImageDraw.ImageDraw, plan: InfographicPlan) -> None:
    _draw_outer_frame(draw)
    draw.rectangle((92, 104, 252, 114), fill=MUTED_BLUE)
    _draw_title(draw, plan.title, y=148, font_size=54, max_width=880)
    positions = _briefing_positions(len(plan.blocks))
    for block, rect in zip(plan.blocks, positions):
        _draw_briefing_card(draw, rect, block)
    _draw_takeaway(draw, plan.takeaway, y=1068)


def _draw_outer_frame(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(
        (54, 54, INFOGRAPHIC_WIDTH - 54, INFOGRAPHIC_HEIGHT - 54),
        radius=44,
        outline=INK,
        width=3,
    )
    draw.rounded_rectangle(
        (76, 76, INFOGRAPHIC_WIDTH - 76, INFOGRAPHIC_HEIGHT - 76),
        radius=34,
        outline=INK,
        width=1,
    )


def _draw_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    *,
    y: int,
    font_size: int,
    max_width: int,
) -> None:
    fit = _fit_text_to_box(
        draw,
        title,
        width=max_width,
        height=210,
        start_font_size=font_size,
        min_font_size=38,
        max_lines=3,
        terminal=False,
    )
    for line in fit.lines:
        width = _text_width(draw, line, fit.font)
        draw.text(((INFOGRAPHIC_WIDTH - width) // 2, y), line, font=fit.font, fill=INK)
        y += _line_height(draw, line, fit.font) + fit.line_gap


def _draw_editorial_block(
    draw: ImageDraw.ImageDraw,
    y: int,
    block: InfographicBlock,
    index: int,
) -> None:
    label_font = _load_monospace_font(26)
    draw.text((112, y + 4), f"{index:02}", font=label_font, fill=MUTED_BLUE)
    draw.text((180, y + 4), block.label.upper(), font=label_font, fill=INK)
    draw.line((180, y + 42, 920, y + 42), fill=WARM_GRAY, width=2)
    _draw_wrapped_box(
        draw,
        block.text,
        x=180,
        y=y + 62,
        width=760,
        height=86,
        start_font_size=38,
        min_font_size=28,
        max_lines=3,
    )


def _draw_schematic_block(
    draw: ImageDraw.ImageDraw,
    y: int,
    block: InfographicBlock,
    *,
    height: int,
) -> None:
    label_font = _load_monospace_font(24)
    draw.rounded_rectangle(
        (212, y, 932, y + height),
        radius=18,
        fill=(249, 244, 234),
        outline=MUTED_BLUE,
        width=2,
    )
    draw.text((238, y + 20), block.label.upper(), font=label_font, fill=MUTED_BLUE)
    _draw_wrapped_box(
        draw,
        block.text,
        x=238,
        y=y + 58,
        width=650,
        height=max(72, height - 74),
        start_font_size=34,
        min_font_size=25,
        max_lines=3,
    )


def _briefing_positions(count: int) -> list[tuple[int, int, int, int]]:
    base = [
        (104, 354, 506, 562),
        (532, 354, 956, 562),
        (104, 592, 506, 812),
        (532, 592, 956, 812),
    ]
    if count <= 2:
        return [(104, 378, 956, 578), (104, 612, 956, 812)][:count]
    if count == 3:
        return [base[0], base[1], (104, 592, 956, 812)]
    return base[:count]


def _draw_briefing_card(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    block: InfographicBlock,
) -> None:
    label_font = _load_monospace_font(23)
    draw.rounded_rectangle(rect, radius=22, fill=(249, 244, 234), outline=WARM_GRAY, width=2)
    left, top, right, bottom = rect
    draw.rectangle((left + 24, top + 26, left + 92, top + 34), fill=MUTED_BLUE)
    draw.text((left + 24, top + 54), block.label.upper(), font=label_font, fill=MUTED_BLUE)
    _draw_wrapped_box(
        draw,
        block.text,
        x=left + 24,
        y=top + 94,
        width=right - left - 48,
        height=bottom - top - 116,
        start_font_size=32,
        min_font_size=24,
        max_lines=4,
    )


def _draw_takeaway(draw: ImageDraw.ImageDraw, takeaway: str, *, y: int) -> None:
    label_font = _load_monospace_font(24)
    draw.rounded_rectangle(
        (104, y, 976, y + 138),
        radius=24,
        fill=LIGHT_BLUE,
        outline=None,
    )
    draw.text((132, y + 24), "TAKEAWAY", font=label_font, fill=MUTED_BLUE)
    _draw_wrapped_box(
        draw,
        takeaway,
        x=132,
        y=y + 62,
        width=812,
        height=58,
        start_font_size=34,
        min_font_size=25,
        max_lines=2,
    )


def _draw_portrait_footer(draw: ImageDraw.ImageDraw, footer_text: str) -> None:
    font = _load_monospace_font(24)
    box = draw.textbbox((0, 0), footer_text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    padding_x = 18
    padding_y = 6
    label_width = width + padding_x * 2
    label_height = height + padding_y * 2
    label_left = (INFOGRAPHIC_WIDTH - label_width) // 2
    label_top = INFOGRAPHIC_HEIGHT - 78 - label_height // 2
    draw.rounded_rectangle(
        (label_left, label_top, label_left + label_width, label_top + label_height),
        radius=label_height // 2,
        fill=BACKGROUND,
        outline=None,
    )
    draw.text(
        ((INFOGRAPHIC_WIDTH - width) // 2, label_top + padding_y - box[1]),
        footer_text,
        font=font,
        fill=INK,
    )


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    max_lines: int | None = None,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if _text_width(draw, candidate, font) <= max_width:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
        if _text_width(draw, word, font) > max_width:
            lines.extend(textwrap.wrap(word, width=16))
            current = []
        else:
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [text[:32]]


def _fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    height: int,
    start_font_size: int,
    min_font_size: int,
    max_lines: int | None,
    terminal: bool = True,
) -> TextFit:
    cleaned = " ".join(text.split()).strip()
    if terminal:
        cleaned = _ensure_terminal(cleaned)
    candidates = [cleaned]
    for limit in range(min(len(cleaned) - 1, 120), 34, -8):
        shortened = _shorten(cleaned, limit) if terminal else _shorten_heading(cleaned, limit)
        if shortened and shortened not in candidates:
            candidates.append(shortened)

    for candidate in candidates:
        for font_size in range(start_font_size, min_font_size - 1, -2):
            font = _load_font(font_size)
            lines = _wrap_lines(draw, candidate, font, width, max_lines=max_lines)
            if max_lines is not None and len(lines) > max_lines:
                continue
            line_gap = max(6, font_size // 4)
            if _lines_height(draw, lines, font, line_gap) <= height:
                return TextFit(font=font, lines=lines, line_gap=line_gap)

    fallback = _shorten(cleaned, 48) if terminal else _shorten_heading(cleaned, 48)
    font = _load_font(min_font_size)
    lines = _wrap_lines(draw, fallback, font, width, max_lines=max_lines)
    while max_lines is not None and len(lines) > max_lines and len(fallback) > 20:
        fallback = _shorten(fallback, max(20, len(fallback) - 8)) if terminal else _shorten_heading(
            fallback, max(20, len(fallback) - 8)
        )
        lines = _wrap_lines(draw, fallback, font, width, max_lines=max_lines)
    line_gap = max(5, min_font_size // 5)
    return TextFit(font=font, lines=lines, line_gap=line_gap)


def _text_fits_without_shortening(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    height: int,
    start_font_size: int,
    min_font_size: int,
    max_lines: int | None,
) -> bool:
    cleaned = _ensure_terminal(" ".join(text.split()).strip())
    for font_size in range(start_font_size, min_font_size - 1, -2):
        font = _load_font(font_size)
        lines = _wrap_lines(draw, cleaned, font, width, max_lines=max_lines)
        if max_lines is not None and len(lines) > max_lines:
            continue
        line_gap = max(6, font_size // 4)
        if _lines_height(draw, lines, font, line_gap) <= height:
            return True
    return False


def _draw_wrapped_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    start_font_size: int,
    min_font_size: int,
    max_lines: int | None,
) -> None:
    fit = _fit_text_to_box(
        draw,
        text,
        width=width,
        height=height,
        start_font_size=start_font_size,
        min_font_size=min_font_size,
        max_lines=max_lines,
    )
    for line in fit.lines:
        draw.text((x, y), line, font=fit.font, fill=INK)
        y += _line_height(draw, line, fit.font) + fit.line_gap


def _lines_height(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    line_gap: int,
) -> int:
    if not lines:
        return 0
    return sum(_line_height(draw, line, font) for line in lines) + line_gap * (len(lines) - 1)


def _line_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[3] - box[1]
