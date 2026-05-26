# Gemma Tweet Bot

This bot generates short tweets with Ollama, posts them to X, records each successful post in GitHub, and sends a short Telegram summary.

## GitHub Actions schedule

The workflow makes several attempts shortly after these fixed India-time slots and posts only once per slot:

- 06:00 IST
- 12:00 IST
- 18:00 IST
- 22:00 IST

The Python schedule guard controls the actual posting slots. `ENABLED_RUN_SLOTS` should use the same list:

```text
06:00,12:00,18:00,22:00
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

Recommended defaults:

```text
OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=gemma4:31b-cloud
MAX_TWEET_CHARS=230
MAX_RETRIES=5
OLLAMA_TIMEOUT_SECONDS=120
POST_TO_X=true
RUN_TIMEZONE=Asia/Kolkata
ENABLED_RUN_SLOTS=06:00,12:00,18:00,22:00
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

GitHub Actions writes successful posts to `tweet-history.md` on a separate branch named `tweet-history`. This keeps the `main` branch stable, so routine bot runs do not create conflicts when code or workflow changes are pushed.

To view the log in GitHub, switch the branch selector from `main` to `tweet-history` and open `tweet-history.md`.

For local runs, the default log path is `logs/tweet-history.md` unless `LOG_FILE_PATH` is set in `.env`.

Telegram receives only:

- Topic
- Tone
- Total time taken
- Number of generation attempts
- Full tweet text
