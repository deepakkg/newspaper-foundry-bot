from __future__ import annotations

import re
import time
from datetime import timezone
from typing import Any

from openai import OpenAI

from config import AppConfig
from news_fetcher import NewsItem


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
HARD_FAIL_PHRASES = (
    "my brain",
    "feels like",
    "really feels",
    "feels like static",
)
SOFT_FAIL_PHRASES = (
    "just",
    "kind of",
    "sort of",
    "seriously",
    "honestly",
    "static",
)
TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
GENERIC_DRIFT_PHRASES = (
    "coffee",
    "donut",
    "late night",
    "long night",
    "lukewarm",
    "brain",
    "data is",
    "data flows",
    "numbers",
    "economic data",
    "projections",
)
GENERIC_VAGUE_PHRASES = (
    "it is a ritual",
    "it's a ritual",
    "easier to face the day",
    "hitting the spot",
    "pure comfort",
    "focused action",
    "doesn't it?",
    "doesnt it?",
    "often involves",
    "definitely",
)
GENERIC_PATTERNS = (
    re.compile(r"\bit feels\b"),
    re.compile(r"\bstrangely\b"),
    re.compile(r"\bdefinitely\b"),
    re.compile(r"\boften\b"),
)
PSEUDO_PROFOUND_PHRASES = (
    "the real lesson",
    "the bigger lesson",
    "the hidden truth",
    "what it really means",
    "what this teaches us",
    "reminds us that",
    "in a world where",
    "the future belongs to",
    "the true power",
    "true power lies",
    "the real magic",
    "chaos into clarity",
    "complexity into clarity",
    "turn uncertainty into",
    "turns uncertainty into",
)
PSEUDO_PROFOUND_PATTERNS = (
    re.compile(r"\b(?:it'?s|it is|this is)\s+not\s+about\b.{0,100}\babout\b"),
    re.compile(r"\bisn'?t\s+about\b.{0,100}\bit'?s\s+about\b"),
    re.compile(r"\bnot\s+just\s+about\b.{0,100}\bit'?s\s+about\b"),
)


def build_client(config: AppConfig) -> OpenAI:
    return OpenAI(
        api_key=config.llm_api_key or "not-needed",
        base_url=config.llm_base_url,
        timeout=config.timeout_seconds,
    )


def normalize_topic(topic: str) -> tuple[str, list[str]]:
    cleaned = topic.strip()
    lowered = cleaned.lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    meaningful_tokens = [
        token for token in tokens if token not in TOPIC_STOPWORDS and len(token) > 2
    ]
    return cleaned, meaningful_tokens


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_topic_hint(topic: str) -> str:
    tokens = tokenize_text(topic)
    if not tokens:
        return topic.strip()

    meaningful_tokens = [
        token for token in tokens if token not in TOPIC_STOPWORDS and len(token) > 2
    ]
    chosen_tokens = meaningful_tokens[:2] or tokens[:2]
    return " ".join(chosen_tokens)


def format_news_context(news_item: NewsItem | None) -> str:
    if news_item is None:
        return ""

    published_at = news_item.published_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    summary = news_item.summary or news_item.title
    return f"""
Current news context:
- Article title: {news_item.title}
- Source: {news_item.source}
- Published: {published_at}
- Key point: {summary}
"""


def build_prompt(
    topic: str,
    tone: str,
    max_tweet_chars: int,
    attempt_number: int,
    news_item: NewsItem | None = None,
) -> str:
    retry_block = ""
    if attempt_number in (2, 3):
        retry_block = f"""

Retry:
- Attempt {attempt_number - 1} was invalid.
- Be more direct, natural, and clearly on-topic.
- Name the topic directly or make the reference unmistakable.
- Cut filler and keep it within {max_tweet_chars} characters.
"""
    elif attempt_number >= 4:
        retry_block = f"""

Fallback:
- Prior attempts were too generic or off-topic.
- Use plain language and the topic name directly.
- One or two short sentences, max {max_tweet_chars} characters.
"""

    news_context = format_news_context(news_item)
    news_rules = ""
    if news_item is not None:
        news_rules = """
- Use the news item as the trigger for a broader take about the topic.
- Do not summarize the article; react to what it reveals.
- Do not invent facts beyond the provided news context.
- Do not include the article URL.
"""

    return f"""Write one post about: {topic}
Tone: {tone}
{news_context}

Rules:
- Stay clearly about the topic.
- Write like Deepak: direct, practical, concise, and not overly polished.
- Use the topic name directly or make the reference unmistakable.
- Shape: clear opinion or observation, one concrete detail, sharp practical implication or dry punchline.
- Include at least one specific noun from the topic or news context.
- Use short, clean sentences.
- Keep the wording specific, restrained, and human.
- Keep tone in the wording, not as filler.
- Tone guide: witty = dry/sharp/understated; funny = lightly absurd; nostalgic = concrete memory or old-internet feel; analysis = clear implication/tradeoff; rant = controlled frustration, not outrage.
- Do not force first person unless it sounds natural.
- Include 1 or 2 relevant emojis.
- Max {max_tweet_chars} characters.
{news_rules}

Do not use:
- Hashtags, labels, or quotes.
- Generic filler, work-stress drift, or meta commentary.
- "Imagine this", "Picture a world", or "In a world where".
- Pseudo-profound framing like "It's not about X, it's about Y" or "The real lesson".
- Forced inspiration, grand lessons, or performative wisdom.
- More than two emojis or ellipsis.
- Comma-heavy chains.
- Filler like "just", "kind of", "sort of", "my brain", "feels like static", "really feels", "seriously", or "honestly".

Output only the post text.
{retry_block}
"""


def build_compact_prompt(
    topic: str,
    tone: str,
    max_tweet_chars: int,
    attempt_number: int,
    news_item: NewsItem | None = None,
) -> str:
    retry_hint = ""
    if attempt_number > 1:
        retry_hint = " Previous attempt was invalid. Be more direct."

    news_hint = ""
    if news_item is not None:
        news_hint = (
            f" Use this news: {news_item.title} from {news_item.source}. "
            f"Key point: {news_item.summary or news_item.title}."
        )

    return (
        f"Write one post about {topic}. Tone: {tone}. "
        f"{news_hint}"
        "Direct, practical, concise. "
        f"Stay on topic with one concrete detail under {max_tweet_chars} characters."
        " Use 1 or 2 relevant emojis. No hashtags, labels, quotes, no article URL,"
        " filler, meta commentary, or pseudo-profound framing."
        f"{retry_hint} Output only the post text."
    )


def build_minimal_prompt(
    topic: str,
    tone: str,
    max_tweet_chars: int,
    news_item: NewsItem | None = None,
) -> str:
    topic_hint = build_topic_hint(topic)
    news_hint = f" Latest news: {news_item.title}." if news_item else ""
    return (
        f"Post about {topic_hint}.{news_hint} Tone: {tone}. "
        f"Under {max_tweet_chars} chars. Direct, practical. Add 1-2 emojis. No hashtag/link."
    )


def clean_generated_tweet(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    candidate = lines[-1]
    quote_chars = "\"'\u201c\u201d"
    return candidate.strip(quote_chars).strip()


def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(text))


def is_overdecorated(text: str) -> bool:
    ellipsis_count = text.count("...") + text.count("\u2026")
    noisy_punct = any(mark in text for mark in ("??", "!!", "?!?", "—", "––"))
    comma_heavy = text.count(",") >= 4

    if ellipsis_count > 1:
        return True
    if ellipsis_count == 1 and noisy_punct:
        return True
    return noisy_punct or comma_heavy


def get_style_issue(text: str) -> str | None:
    lowered = text.lower()

    emoji_count = count_emojis(text)
    if emoji_count < 1:
        return "missing emoji"
    if emoji_count > 2:
        return "too many emojis"
    if is_overdecorated(text):
        return "too much punctuation clutter"

    if any(phrase in lowered for phrase in HARD_FAIL_PHRASES):
        return "hard-fail filler phrasing"

    soft_hits = sum(1 for phrase in SOFT_FAIL_PHRASES if phrase in lowered)
    if soft_hits >= 2:
        return "too many filler phrases"

    if lowered.startswith(("my brain", "feels like")):
        return "bad filler opening"

    if is_pseudo_profound(text):
        return "pseudo-profound phrasing"

    return None


def is_pseudo_profound(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in PSEUDO_PROFOUND_PHRASES):
        return True
    return any(pattern.search(lowered) for pattern in PSEUDO_PROFOUND_PATTERNS)


def is_on_topic(tweet: str, original_topic: str, topic_tokens: list[str]) -> bool:
    lowered_tweet = tweet.lower()
    tweet_tokens = set(tokenize_text(tweet))
    lowered_topic = original_topic.lower()

    if lowered_topic and lowered_topic in lowered_tweet:
        return True

    if not topic_tokens:
        return lowered_topic in lowered_tweet

    matched_tokens = sum(1 for token in topic_tokens if token in tweet_tokens)
    if len(topic_tokens) == 1:
        return matched_tokens >= 1
    return matched_tokens >= 2


def has_enough_specificity(tweet: str, topic_tokens: list[str]) -> bool:
    lowered = tweet.lower()
    tweet_tokens = tokenize_text(tweet)
    unique_tokens = {
        token
        for token in tweet_tokens
        if token not in TOPIC_STOPWORDS and len(token) > 2 and token not in topic_tokens
    }

    if len(unique_tokens) < 4:
        return False
    if any(phrase in lowered for phrase in GENERIC_VAGUE_PHRASES):
        return False
    if any(pattern.search(lowered) for pattern in GENERIC_PATTERNS):
        return False
    if tweet.rstrip().endswith("?"):
        return False
    return True


def is_generic_drift(tweet: str, original_topic: str, topic_tokens: list[str]) -> bool:
    lowered = tweet.lower()

    if not is_on_topic(tweet, original_topic, topic_tokens):
        return True

    drift_hits = sum(1 for phrase in GENERIC_DRIFT_PHRASES if phrase in lowered)
    if drift_hits >= 2 and not any(token in lowered for token in topic_tokens):
        return True

    short_generic = (
        len(topic_tokens) > 0
        and len(lowered.split()) < 7
        and not any(token in lowered for token in topic_tokens)
    )
    if short_generic:
        return True

    if topic_tokens and not has_enough_specificity(tweet, topic_tokens):
        return True

    return False


def validate_tweet(
    tweet: str,
    original_topic: str,
    topic_tokens: list[str],
    max_tweet_chars: int,
    attempt_number: int,
    max_retries: int,
) -> str | None:
    if len(tweet) > max_tweet_chars:
        return "too long"
    style_issue = get_style_issue(tweet)
    if style_issue and not (
        attempt_number == max_retries
        and style_issue in {"too many filler phrases", "too much punctuation clutter"}
        and count_emojis(tweet) <= 1
    ):
        return style_issue
    if is_generic_drift(tweet, original_topic, topic_tokens):
        if is_on_topic(tweet, original_topic, topic_tokens):
            return "too generic"
        return "off topic"
    return None


def is_context_length_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "prompt too long" in lowered or "context length" in lowered


def extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Server response did not include a valid post.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Server response did not include a valid post.")
    return content


def request_completion(client: OpenAI, config: AppConfig, prompt: str) -> Any:
    return client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )


def request_tweet(
    client: OpenAI,
    config: AppConfig,
    topic: str,
    tone: str,
    attempt_number: int,
    news_item: NewsItem | None = None,
    max_tweet_chars: int | None = None,
) -> str:
    resolved_max_tweet_chars = max_tweet_chars or config.max_tweet_chars
    prompt = build_prompt(
        topic,
        tone,
        resolved_max_tweet_chars,
        attempt_number,
        news_item,
    )
    try:
        response = request_completion(client, config, prompt)
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
        compact_prompt = build_compact_prompt(
            topic, tone, resolved_max_tweet_chars, attempt_number, news_item
        )
        try:
            response = request_completion(client, config, compact_prompt)
        except Exception as compact_exc:
            if not is_context_length_error(compact_exc):
                raise
            response = request_completion(
                client,
                config,
                build_minimal_prompt(
                    topic,
                    tone,
                    resolved_max_tweet_chars,
                    news_item,
                ),
            )
    tweet = extract_response_text(response)
    cleaned_tweet = clean_generated_tweet(tweet)
    if not cleaned_tweet:
        raise RuntimeError("Server response did not include a usable post.")
    return cleaned_tweet


def generate_valid_tweet(
    client: OpenAI,
    config: AppConfig,
    topic: str,
    tone: str,
    news_item: NewsItem | None = None,
    max_tweet_chars: int | None = None,
) -> tuple[str, float, int]:
    original_topic, topic_tokens = normalize_topic(topic)
    resolved_max_tweet_chars = max_tweet_chars or config.max_tweet_chars
    last_reason = "unknown validation failure"
    start = time.perf_counter()

    for attempt in range(1, config.max_retries + 1):
        tweet = request_tweet(
            client,
            config,
            original_topic,
            tone,
            attempt,
            news_item,
            resolved_max_tweet_chars,
        )
        failure_reason = validate_tweet(
            tweet,
            original_topic,
            topic_tokens,
            resolved_max_tweet_chars,
            attempt,
            config.max_retries,
        )
        if failure_reason is None:
            elapsed = time.perf_counter() - start
            return tweet, elapsed, attempt
        last_reason = failure_reason

    elapsed = time.perf_counter() - start
    raise RuntimeError(
        f"Could not generate a valid post after {config.max_retries} attempts: {last_reason}."
    )
