# Gemma Tweet Bot

This bot generates short tweets with Ollama, posts them to X, records each successful post in GitHub, and sends a short Telegram summary.

## GitHub Actions schedule

The workflow wakes at these fixed India-time slots:

- 02:00 IST
- 06:00 IST
- 10:00 IST
- 14:00 IST
- 18:00 IST
- 22:00 IST

Only slots listed in `ENABLED_RUN_SLOTS` will post. The default is:

```text
06:00,10:00,14:00,18:00
```

## GitHub Secrets

Store these in repository Settings -> Secrets and variables -> Actions -> Secrets:

- `OLLAMA_API_KEY`
- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## GitHub Variables

Store these in repository Settings -> Secrets and variables -> Actions -> Variables:

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `TOPICS`
- `TONES`
- `MAX_TWEET_CHARS`
- `MAX_RETRIES`
- `OLLAMA_TIMEOUT_SECONDS`
- `POST_TO_X`
- `X_USERNAME`
- `RUN_TIMEZONE`
- `ENABLED_RUN_SLOTS`
- `LOG_FILE_PATH`

Recommended defaults:

```text
OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=gemma4:31b-cloud
MAX_TWEET_CHARS=230
MAX_RETRIES=5
OLLAMA_TIMEOUT_SECONDS=120
POST_TO_X=true
RUN_TIMEZONE=Asia/Kolkata
ENABLED_RUN_SLOTS=06:00,10:00,14:00,18:00
LOG_FILE_PATH=logs/tweet-history.md
```

## Local setup

Create a local `.env` from `.env.example`, then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run once locally:

```bash
.venv/bin/python tweet_generator.py
```

Run with the GitHub schedule guard:

```bash
.venv/bin/python tweet_generator.py --respect-schedule
```

## Logs and Telegram

Successful posts are appended to `logs/tweet-history.md`.

Telegram receives only:

- Topic
- Tone
- Total time taken
- Number of generation attempts
- Full tweet text
