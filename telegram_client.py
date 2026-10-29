"""Minimal Telegram Bot API client (long polling, text, inline keyboards)."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

from llm import chunk_text

log = logging.getLogger("minamo.tg")


class TelegramError(Exception):
    pass


class RateLimited(TelegramError):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


def keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row] for row in rows
        ]
    }


def remove_keyboard() -> dict:
    return {"inline_keyboard": []}


class Telegram:
    def __init__(self, token: str) -> None:
        self.base = f"https://api.telegram.org/bot{token}/"

    def call(
        self,
        method: str,
        payload: dict | None = None,
        timeout: int = 45,
        retries: int = 2,
    ):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error = "telegram unreachable"
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.base + method,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if body.get("ok"):
                    return body["result"]
                if body.get("error_code") == 429:
                    retry_after = (body.get("parameters") or {}).get("retry_after", 3)
                    if attempt < retries:
                        time.sleep(min(float(retry_after), 10))
                        continue
                    raise RateLimited(float(retry_after))
                raise TelegramError(f"{method}: {body.get('description', 'error')}")
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", "replace")
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                if exc.code == 429:
                    retry_after = (body.get("parameters") or {}).get("retry_after", 3)
                    if attempt < retries:
                        time.sleep(min(float(retry_after), 10))
                        continue
                    raise RateLimited(float(retry_after))
                raise TelegramError(
                    f"{method}: HTTP {exc.code} {body.get('description', raw[:200])}"
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{method}: network error: {exc}"
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise TelegramError(last_error)

    # -- updates ----------------------------------------------------------- #
    def get_updates(self, offset: int, timeout: int = 25) -> list[dict]:
        return self.call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
            timeout=timeout + 20,
        )

    # -- sending ----------------------------------------------------------- #
    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict | None = None,
        reply_to: int | None = None,
    ) -> dict:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        return self.call("sendMessage", payload)

    def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict | None = None,
    ) -> None:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self.call("editMessageText", payload)

    def send_typing(self, chat_id: int) -> None:
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, retries=0)
        except TelegramError:
            pass

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        try:
            self.call(
                "answerCallbackQuery",
                {"callback_query_id": callback_id, "text": text[:200]},
                retries=0,
            )
        except TelegramError as exc:
            log.debug("answerCallbackQuery failed: %s", exc)

    def get_me(self) -> dict:
        return self.call("getMe")


class StreamRenderer:
    """Keeps one growing Telegram message per chunk, editing it at a sane rate."""

    def __init__(self, tg: Telegram, chat_id: int, message_id: int, interval: float) -> None:
        self.tg = tg
        self.chat_id = chat_id
        self.message_ids = [message_id]
        self.sent: list[str | None] = [None]
        self.interval = interval
        self.last_flush = 0.0
        self.lock = threading.Lock()

    def render(self, text: str, force: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_flush < self.interval:
                return
            for index, chunk in enumerate(chunk_text(text)):
                if index >= len(self.message_ids):
                    try:
                        message = self.tg.send_message(self.chat_id, chunk)
                    except TelegramError as exc:
                        log.warning("send failed: %s", exc)
                        return
                    self.message_ids.append(message["message_id"])
                    self.sent.append(chunk)
                    continue
                if self.sent[index] == chunk:
                    continue
                try:
                    self.tg.edit_message(self.chat_id, self.message_ids[index], chunk)
                    self.sent[index] = chunk
                except RateLimited as exc:
                    time.sleep(min(exc.retry_after, 5))
                except TelegramError as exc:
                    log.debug("edit failed (ignored): %s", exc)
            self.last_flush = now
