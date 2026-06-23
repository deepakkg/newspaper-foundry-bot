from __future__ import annotations

import hashlib
import random
import re
import textwrap
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
EMOJI_PATTERN = re.compile(
    "["
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


def _text_height(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont) -> int:
    if not lines:
        return 0
    heights = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])
    return sum(heights) + (len(lines) - 1) * 22


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
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
        if draw.textbbox((0, 0), word, font=font)[2] > max_width:
            lines.extend(textwrap.wrap(word, width=16))
            current = []
        else:
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


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
    without_urls = URL_PATTERN.sub("", post_text)
    without_hashtags = HASHTAG_PATTERN.sub("", without_urls)
    without_emojis = EMOJI_PATTERN.sub("", without_hashtags)
    return " ".join(without_emojis.split())


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    x = (IMAGE_SIZE - text_width) // 2
    draw.text((x, y), text, font=font, fill=fill)


def render_instagram_image(
    post_text: str,
    output_path: Path,
    *,
    footer_text: str = BOT_HASHTAG,
) -> Path:
    image_body_text = build_instagram_image_body_text(post_text)
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
    max_height = footer_y - 230
    font_size = 58
    lines: list[str] = []
    font: ImageFont.ImageFont = _load_font(font_size)
    while font_size >= 34:
        font = _load_font(font_size)
        lines = _wrap_text(draw, image_body_text, font, max_width)
        if _text_height(draw, lines, font) <= max_height:
            break
        font_size -= 4

    total_height = _text_height(draw, lines, font)
    y = max(150, (footer_y - total_height) // 2)
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        line_width = box[2] - box[0]
        line_height = box[3] - box[1]
        x = (IMAGE_SIZE - line_width) // 2
        draw.text((x, y), line, font=font, fill=INK)
        y += line_height + 22

    _draw_centered_text(
        draw,
        footer_text,
        y=footer_y - footer_height,
        font=footer_font,
        fill=INK,
    )

    image.save(output_path, format="PNG")
    return output_path
