"""Ollama client, reasoning splitter and text chunking."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from config import Config

TG_LIMIT = 4000
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class OllamaError(Exception):
    pass


# One generation at a time: this box is CPU-only, parallel calls just thrash.
MODEL_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# reasoning / text helpers
# --------------------------------------------------------------------------- #
def split_reasoning(raw: str) -> tuple[str, str]:
    """Return (thinking, answer) from raw model output containing <think> blocks."""
    think_parts: list[str] = []
    answer_parts: list[str] = []
    index = 0
    in_think = False
    while index < len(raw):
        if not in_think:
            start = raw.find(THINK_OPEN, index)
            if start < 0:
                answer_parts.append(raw[index:])
                break
            answer_parts.append(raw[index:start])
            index = start + len(THINK_OPEN)
            in_think = True
        else:
            end = raw.find(THINK_CLOSE, index)
            if end < 0:
                think_parts.append(raw[index:])
                break
            think_parts.append(raw[index:end])
            index = end + len(THINK_CLOSE)
            in_think = False

    answer = _hold_partial_tag("".join(answer_parts))
    thinking = _hold_partial_tag("".join(think_parts))
    return thinking.strip(), answer


def _hold_partial_tag(text: str) -> str:
    cut = text.rfind("<")
    if cut < 0:
        return text
    tail = text[cut:]
    if THINK_OPEN.startswith(tail) or THINK_CLOSE.startswith(tail):
        return text[:cut]
    return text


def chunk_text(text: str, limit: int = TG_LIMIT) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + limit, length)
        if end < length:
            newline = text.rfind("\n", start, end)
            if newline > start + int(limit * 0.6):
                end = newline + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def clamp(text: str, limit: int = TG_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head - 20
    return f"{text[:head]}\n… ({len(text)} chars total) …\n{text[-tail:]}"


# --------------------------------------------------------------------------- #
# ollama API
# --------------------------------------------------------------------------- #
def ollama_tags(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise OllamaError(f"cannot reach Ollama at {url}: {exc}") from exc
    return [m.get("name", "") for m in body.get("models", [])]


def _options(cfg: Config, num_predict: int | None = None) -> dict:
    options = {
        "num_ctx": cfg.num_ctx,
        "num_predict": num_predict if num_predict is not None else cfg.num_predict,
    }
    if cfg.temperature:
        try:
            options["temperature"] = float(cfg.temperature)
        except ValueError:
            pass
    return options


def _keep_alive(cfg: Config):
    value: object = cfg.keep_alive
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def ollama_chat(
    url: str,
    model: str,
    messages: list[dict],
    cfg: Config,
    on_piece,
    cancel: threading.Event,
    num_predict: int | None = None,
) -> dict:
    """Stream a chat completion, one generation at a time across the process."""
    with MODEL_LOCK:
        return _chat_stream(url, model, messages, cfg, on_piece, cancel, num_predict)


def _chat_stream(
    url: str,
    model: str,
    messages: list[dict],
    cfg: Config,
    on_piece,
    cancel: threading.Event,
    num_predict: int | None = None,
) -> dict:
    """Stream a chat completion. Calls on_piece(text) for every content token."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": _options(cfg, num_predict),
        "keep_alive": _keep_alive(cfg),
    }
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=cfg.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"cannot reach Ollama at {url}: {exc}") from exc

    stats: dict = {"cancelled": False, "thinking": ""}
    thinking_parts: list[str] = []
    try:
        with resp:
            for raw_line in resp:
                if cancel.is_set():
                    stats["cancelled"] = True
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                message = chunk.get("message") or {}
                piece = message.get("content") or ""
                reasoning = message.get("thinking") or ""
                # Ollama reports reasoning in its own field for thinking models.
                # Keep it out of the content stream: injecting tags here would
                # swallow the answer when the field never emits a closing tag.
                if reasoning:
                    thinking_parts.append(reasoning)
                if piece:
                    on_piece(piece)
                if chunk.get("done"):
                    stats.update(
                        {
                            "eval_count": chunk.get("eval_count", 0),
                            "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                            "total_duration": chunk.get("total_duration", 0),
                            "done_reason": chunk.get("done_reason", ""),
                        }
                    )
                    break
    except (OSError, TimeoutError) as exc:
        if not stats.get("eval_count"):
            raise OllamaError(f"Ollama stream failed: {exc}") from exc
    stats["thinking"] = "".join(thinking_parts)
    return stats


def ollama_once(
    url: str,
    model: str,
    messages: list[dict],
    cfg: Config,
    num_predict: int | None = None,
    think: bool = True,
    stop: list[str] | None = None,
) -> tuple[str, str, dict]:
    """Non-streaming call. Returns (answer, thinking, stats).

    With think=True Ollama returns reasoning in a separate field, so the answer
    is clean JSON instead of a <think> blob.
    """
    with MODEL_LOCK:
        return _once_call(url, model, messages, cfg, num_predict, think, stop)


def _once_call(
    url: str,
    model: str,
    messages: list[dict],
    cfg: Config,
    num_predict: int | None,
    think: bool,
    stop: list[str] | None,
) -> tuple[str, str, dict]:
    options = _options(cfg, num_predict)
    if stop:
        options["stop"] = stop
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": options,
        "keep_alive": _keep_alive(cfg),
    }
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"cannot reach Ollama at {url}: {exc}") from exc

    message = body.get("message") or {}
    answer = message.get("content") or ""
    thinking = message.get("thinking") or ""
    if THINK_OPEN in answer:
        tag_think, answer = split_reasoning(answer)
        thinking = f"{thinking}\n{tag_think}".strip()
    stats = {
        "eval_count": body.get("eval_count", 0),
        "prompt_eval_count": body.get("prompt_eval_count", 0),
        "total_duration": body.get("total_duration", 0),
        "done_reason": body.get("done_reason", ""),
    }
    return answer.strip(), thinking.strip(), stats
