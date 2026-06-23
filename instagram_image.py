from __future__ import annotations

import hashlib
import random
import re
import textwrap
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = 1080
OUTER_BORDER_WIDTH = 3
INNER_BORDER_WIDTH = 1
BACKGROUND = (246, 239, 224)
INK = (20, 20, 18)
MUTED_BLUE = (81, 107, 128)
BOT_HASHTAG = "#botWrites"
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
KEYCAP_SEQUENCE_PATTERN = re.compile(r"[0-9#*]\ufe0f?\u20e3")
VARIATION_SELECTOR_PATTERN = re.compile("[\ufe00-\ufe0f]")
ZERO_WIDTH_PATTERN = re.compile("[\u200b-\u200d\u2060]")
KEYCAP_MARK_PATTERN = re.compile("\u20e3")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
UNSUPPORTED_SYMBOL_PATTERN = re.compile("[\ufffd\u25a0-\u25a1\u25ab-\u25ad]")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)
TERMINAL_PUNCTUATION = (".", "!", "?")
ORPHAN_PUNCTUATION = set(".!?;:,")


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in (
        "LibreBaskerville-Regular.ttf",
        "Georgia.ttf",
        "Times New Roman.ttf",
        "DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_emoji_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    for font_name in (
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "NotoColorEmoji.ttf",
        "Apple Color Emoji.ttc",
        "Segoe UI Emoji.ttf",
    ):
        for candidate_size in (size, 64, 72, 96, 109, 128):
            try:
                return ImageFont.truetype(font_name, candidate_size)
            except OSError:
                continue
    return None


def _text_height(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont) -> int:
    if not lines:
        return 0
    heights = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])
    return sum(heights) + (len(lines) - 1) * 22


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _is_orphan_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and all(char in ORPHAN_PUNCTUATION for char in stripped)


def _is_awkward_final_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _is_orphan_line(stripped):
        return True
    return len(stripped) <= 2


def _line_quality_ok(lines: list[str]) -> bool:
    if not lines:
        return False
    if any(not line.strip() or _is_orphan_line(line) for line in lines):
        return False
    return not (len(lines) > 1 and _is_awkward_final_line(lines[-1]))


def _rebalance_final_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if len(lines) < 2 or not _is_awkward_final_line(lines[-1]):
        return lines

    previous_words = lines[-2].split()
    final_words = lines[-1].split()
    if len(previous_words) < 2:
        return lines

    for move_count in range(1, len(previous_words)):
        new_previous = " ".join(previous_words[:-move_count])
        new_final = " ".join([*previous_words[-move_count:], *final_words])
        if not new_previous or not new_final:
            continue
        if _text_width(draw, new_previous, font) > max_width:
            continue
        if _text_width(draw, new_final, font) > max_width:
            continue
        candidate = [*lines[:-2], new_previous, new_final]
        if _line_quality_ok(candidate):
            return candidate
    return lines


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
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
    return _rebalance_final_lines(draw, lines, font, max_width)


def _add_paper_texture(image: Image.Image, seed_text: str) -> None:
    pixels = image.load()
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    for _ in range(14000):
        x = rng.randrange(0, IMAGE_SIZE)
        y = rng.randrange(0, IMAGE_SIZE)
        delta = rng.randrange(-8, 9)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )


def build_instagram_image_body_text(post_text: str) -> str:
    without_keycaps = KEYCAP_SEQUENCE_PATTERN.sub(" ", post_text)
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
    cleaned = cleaned.rstrip(" ,;:-")
    if cleaned and not cleaned.endswith(TERMINAL_PUNCTUATION):
        cleaned = f"{cleaned}."
    return cleaned


def extract_emojis(post_text: str) -> str:
    emojis = EMOJI_PATTERN.findall(post_text)
    return "".join(emojis[:4])


def _draw_text_with_optional_color(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    embedded_color: bool,
) -> None:
    try:
        draw.text(position, text, font=font, fill=fill, embedded_color=embedded_color)
    except TypeError:
        draw.text(position, text, font=font, fill=fill)


def _render_text_sample(text: str, font: ImageFont.ImageFont) -> Image.Image | None:
    image = Image.new("RGBA", (260, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        _draw_text_with_optional_color(
            draw,
            (12, 12),
            text,
            font=font,
            fill=INK,
            embedded_color=True,
        )
    except Exception:
        return None
    return image


def _emoji_text_renders_cleanly(emoji_text: str, font: ImageFont.ImageFont) -> bool:
    rendered = _render_text_sample(emoji_text, font)
    if rendered is None or rendered.getbbox() is None:
        return False

    placeholder = _render_text_sample("□" * len(emoji_text), font)
    if placeholder is not None and rendered.tobytes() == placeholder.tobytes():
        return False
    return True


def _get_renderable_emoji_text(post_text: str) -> tuple[str, ImageFont.ImageFont] | None:
    emoji_text = extract_emojis(post_text)
    if not emoji_text:
        return None
    emoji_font = _load_emoji_font(44)
    if emoji_font is None:
        return None
    if not _emoji_text_renders_cleanly(emoji_text, emoji_font):
        return None
    return emoji_text, emoji_font


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    embedded_color: bool = False,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    x = (IMAGE_SIZE - text_width) // 2
    _draw_text_with_optional_color(
        draw,
        (x, y),
        text,
        font=font,
        fill=fill,
        embedded_color=embedded_color,
    )


def render_instagram_image(
    post_text: str,
    output_path: Path,
    *,
    footer_text: str = BOT_HASHTAG,
) -> Path:
    image_body_text = build_instagram_image_body_text(post_text)
    emoji_render = _get_renderable_emoji_text(post_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    _add_paper_texture(image, image_body_text or post_text)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (58, 58, IMAGE_SIZE - 58, IMAGE_SIZE - 58),
        radius=42,
        outline=INK,
        width=OUTER_BORDER_WIDTH,
    )
    draw.rounded_rectangle(
        (78, 78, IMAGE_SIZE - 78, IMAGE_SIZE - 78),
        radius=34,
        outline=INK,
        width=INNER_BORDER_WIDTH,
    )
    draw.rectangle((120, 124, 250, 132), fill=MUTED_BLUE)

    max_width = 820
    footer_font = _load_font(26)
    footer_box = draw.textbbox((0, 0), footer_text, font=footer_font)
    footer_height = footer_box[3] - footer_box[1]
    footer_y = IMAGE_SIZE - 142
    emoji_gap = 34 if emoji_render else 0
    emoji_height = 58 if emoji_render else 0
    max_height = footer_y - 230 - emoji_gap - emoji_height
    font_size = 58
    lines: list[str] = []
    font: ImageFont.ImageFont = _load_font(font_size)
    while font_size >= 34:
        font = _load_font(font_size)
        lines = _wrap_text(draw, image_body_text, font, max_width)
        if _text_height(draw, lines, font) <= max_height and _line_quality_ok(lines):
            break
        font_size -= 4

    text_height = _text_height(draw, lines, font)
    total_height = text_height + emoji_gap + emoji_height
    y = max(150, (footer_y - total_height) // 2)
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        line_width = box[2] - box[0]
        line_height = box[3] - box[1]
        x = (IMAGE_SIZE - line_width) // 2
        draw.text((x, y), line, font=font, fill=INK)
        y += line_height + 22

    if emoji_render:
        emoji_text, emoji_font = emoji_render
        _draw_centered_text(
            draw,
            emoji_text,
            y=y + emoji_gap,
            font=emoji_font,
            fill=INK,
            embedded_color=True,
        )

    _draw_centered_text(
        draw,
        footer_text,
        y=footer_y - footer_height,
        font=footer_font,
        fill=INK,
    )

    image.save(output_path, format="PNG")
    return output_path
