from __future__ import annotations

import re
import time

from ollama import Client, ResponseError

from config import AppConfig


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
    "]+"
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


def build_client(config: AppConfig) -> Client:
    client_kwargs = {"timeout": config.timeout_seconds}
    if config.ollama_api_key:
        client_kwargs["headers"] = {
            "Authorization": f"Bearer {config.ollama_api_key}"
        }
    return Client(host=config.ollama_host, **client_kwargs)


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


def build_prompt(
    topic: str, tone: str, max_tweet_chars: int, attempt_number: int
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

    return f"""Write one tweet about: {topic}
Tone: {tone}

Rules:
- Stay clearly about the topic.
- Sound human, specific, and restrained.
- Prefer clarity over cleverness.
- Use the topic name directly or make the reference unmistakable.
- Include one concrete detail tied to the topic.
- Use short, clean sentences.
- Keep tone in the wording, not as filler.
- Max {max_tweet_chars} characters.

Do not use:
- Hashtags, labels, or quotes.
- Generic filler, work-stress drift, or meta commentary.
- "Imagine this", "Picture a world", or "In a world where".
- More than one emoji or ellipsis.
- Comma-heavy chains.
- Filler like "just", "kind of", "sort of", "my brain", "feels like static", "really feels", "seriously", or "honestly".

Output only the tweet text.
{retry_block}
"""


def build_compact_prompt(
    topic: str, tone: str, max_tweet_chars: int, attempt_number: int
) -> str:
    retry_hint = ""
    if attempt_number > 1:
        retry_hint = " Previous attempt was invalid. Be more direct."

    return (
        f"Write one tweet about {topic}. Tone: {tone}. "
        f"Stay on topic, sound human, use one concrete detail, and keep it under {max_tweet_chars} characters."
        " No hashtags, no labels, no quotes, no filler, no meta commentary."
        f"{retry_hint} Output only the tweet text."
    )


def build_minimal_prompt(topic: str, tone: str, max_tweet_chars: int) -> str:
    topic_hint = build_topic_hint(topic)
    return (
        f"Tweet about {topic_hint}. Tone: {tone}. "
        f"Under {max_tweet_chars} chars. Plain text only."
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

    if count_emojis(text) > 1:
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

    return None


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


def is_context_length_error(exc: ResponseError) -> bool:
    lowered = str(exc).lower()
    return "prompt too long" in lowered or "context length" in lowered


def request_tweet(
    client: Client, config: AppConfig, topic: str, tone: str, attempt_number: int
) -> str:
    prompt = build_prompt(topic, tone, config.max_tweet_chars, attempt_number)
    try:
        response = client.generate(
            model=config.ollama_model,
            prompt=prompt,
        )
    except ResponseError as exc:
        if not is_context_length_error(exc):
            raise
        compact_prompt = build_compact_prompt(
            topic, tone, config.max_tweet_chars, attempt_number
        )
        try:
            response = client.generate(
                model=config.ollama_model,
                prompt=compact_prompt,
            )
        except ResponseError as compact_exc:
            if not is_context_length_error(compact_exc):
                raise
            response = client.generate(
                model=config.ollama_model,
                prompt=build_minimal_prompt(topic, tone, config.max_tweet_chars),
            )
    tweet = response.get("response")
    if not isinstance(tweet, str) or not tweet.strip():
        raise RuntimeError("Server response did not include a valid tweet.")

    cleaned_tweet = clean_generated_tweet(tweet)
    if not cleaned_tweet:
        raise RuntimeError("Server response did not include a usable tweet.")
    return cleaned_tweet


def generate_valid_tweet(
    client: Client, config: AppConfig, topic: str, tone: str
) -> tuple[str, float, int]:
    original_topic, topic_tokens = normalize_topic(topic)
    last_reason = "unknown validation failure"
    start = time.perf_counter()

    for attempt in range(1, config.max_retries + 1):
        tweet = request_tweet(client, config, original_topic, tone, attempt)
        failure_reason = validate_tweet(
            tweet,
            original_topic,
            topic_tokens,
            config.max_tweet_chars,
            attempt,
            config.max_retries,
        )
        if failure_reason is None:
            elapsed = time.perf_counter() - start
            return tweet, elapsed, attempt
        last_reason = failure_reason

    elapsed = time.perf_counter() - start
    raise RuntimeError(
        f"Could not generate a valid tweet after {config.max_retries} attempts: {last_reason}."
    )
