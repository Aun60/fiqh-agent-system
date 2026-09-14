"""
llm_client.py — single shared wrapper around the LLM so every agent calls
the model the same way. Swap models/providers here in one place.

Uses GOOGLE GEMINI — free tier, no credit card required, and meaningfully
higher free rate limits than Groq's free tier for this workload.

Setup (one-time):
  1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
  2. Click "Create API key" (no billing/card required for the free tier).
  3. Create a file named `.env` in the project root containing:
         GEMINI_API_KEY=your_key_here
  4. python-dotenv (already in requirements.txt) loads it automatically.

Free tier for gemini-2.0-flash-lite (as of writing): 30 requests/minute,
1,500 requests/day — this pipeline makes 2-3 calls per user question, so
that's roughly 10-15 questions/minute of headroom, which should feel
smooth for interactive testing. If you ever need higher quality over
speed, swap MODEL_NAME to "gemini-2.5-flash" (lower free RPM, better
reasoning) — check current numbers at https://ai.google.dev/gemini-api/docs/rate-limits
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5


def call_llm(system: str, user: str, max_tokens: int = 800) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Create a .env file in the project root "
            "with a line: GEMINI_API_KEY=your_key_here "
            "(get a free key at aistudio.google.com/apikey)"
        )

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_tokens,
        },
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=60,
        )

        if response.status_code == 429:
            wait = BASE_BACKOFF_SECONDS * attempt
            print(f"[RATE LIMIT] Gemini 429, retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            last_error = response
            continue

        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # e.g. response was blocked by safety filters and has no content
            finish_reason = data.get("candidates", [{}])[0].get("finishReason", "unknown")
            raise RuntimeError(f"Gemini returned no usable content (finishReason={finish_reason}): {data}")

    last_error.raise_for_status()
