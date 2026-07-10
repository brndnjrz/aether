"""
AI client — routes between Claude (Anthropic API) and Ollama (local LLM).

Provider selection via AI_PROVIDER env var:
  auto   → Claude if ANTHROPIC_API_KEY is set, else Ollama (default)
  claude → Claude only
  ollama → Ollama only

Model defaults: CLAUDE_MODEL and OLLAMA_MODEL in config/settings.py.
"""
import logging
import requests
from typing import Optional
from config.settings import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    AI_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ask_ollama(prompt: str, system: str = "", max_tokens: int = 1500, model: Optional[str] = None) -> Optional[str]:
    model = model or OLLAMA_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Reasoning models (deepseek-r1 distills, deepscaler, qwq, etc.) spend
    # hundreds of tokens on a hidden <think> pass — which Ollama reports back
    # as message["thinking"] — before writing anything into message["content"].
    # A caller-requested budget of 350-600 tokens starves them before they ever
    # reach the answer, so give Ollama generous headroom regardless of the
    # nominal ask; non-reasoning models just stop early (done_reason "stop").
    num_predict = max(max_tokens, 1500)

    for attempt in range(2):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": num_predict},
                },
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            message = data.get("message", {})
            content = message.get("content", "")
            if content:
                return content
            if data.get("done_reason") == "length" and message.get("thinking") and attempt == 0:
                logger.warning(
                    "Ollama model %s exhausted %d-token budget mid-reasoning with no "
                    "answer yet; retrying once with num_predict=%d",
                    model, num_predict, num_predict * 2,
                )
                num_predict *= 2
                continue
            logger.error("Ollama model %s returned no content (done_reason=%s)", model, data.get("done_reason"))
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable at %s", OLLAMA_BASE_URL)
            return None
        except Exception as e:
            logger.error("Ollama error: %s", e)
            return None
    return None


# ── Claude ────────────────────────────────────────────────────────────────────

def _ask_claude(prompt: str, system: str = "", max_tokens: int = 1500) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed")
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        kwargs = {
            "model": CLAUDE_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except Exception as e:
        logger.error("Claude API error: %s", e)
        return None


# ── Router ────────────────────────────────────────────────────────────────────

def ask_ai(prompt: str, system: str = "", max_tokens: int = 1500, ollama_model: Optional[str] = None) -> Optional[str]:
    """
    Route the request to the configured AI provider.
    `ollama_model` overrides OLLAMA_MODEL for this call only (no effect on Claude).
    Returns the response text, or None if no provider is available.
    """
    provider = (AI_PROVIDER or "auto").lower()

    if provider == "claude":
        return _ask_claude(prompt, system, max_tokens)

    if provider == "ollama":
        return _ask_ollama(prompt, system, max_tokens, model=ollama_model)

    # auto: prefer Claude when key is present, fall back to Ollama
    if ANTHROPIC_API_KEY:
        result = _ask_claude(prompt, system, max_tokens)
        if result is not None:
            return result
        logger.info("Claude unavailable, falling back to Ollama")

    return _ask_ollama(prompt, system, max_tokens, model=ollama_model)


def ai_available() -> bool:
    """Return True if a usable AI backend (Claude or Ollama) is currently configured/reachable."""
    provider = (AI_PROVIDER or "auto").lower()
    if provider == "claude":
        return bool(ANTHROPIC_API_KEY)
    if provider == "ollama":
        return _ollama_available()
    return bool(ANTHROPIC_API_KEY) or _ollama_available()


def active_provider() -> str:
    """Return a human-readable label for the currently active AI provider."""
    provider = (AI_PROVIDER or "auto").lower()
    if provider == "claude":
        return "Claude" if ANTHROPIC_API_KEY else "Claude (no key — set ANTHROPIC_API_KEY)"
    if provider == "ollama":
        return f"Ollama ({OLLAMA_MODEL})"
    # auto
    if ANTHROPIC_API_KEY:
        return "Claude (auto)"
    if _ollama_available():
        return f"Ollama ({OLLAMA_MODEL}) (auto)"
    return "None — set ANTHROPIC_API_KEY or start Ollama"


# Backward-compatible alias — all existing code calling ask_claude() keeps working
ask_claude = ask_ai
