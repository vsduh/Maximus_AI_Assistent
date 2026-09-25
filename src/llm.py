"""
LLM-клиент для DeepSeek API (OpenAI-совместимый).

Стриминг ответов + история диалога.
"""
import json
import logging
from typing import Generator, List, Dict

import requests

from .config import Config

logger = logging.getLogger(__name__)


class DeepSeekLLM:
    """Клиент DeepSeek API."""

    SYSTEM_PROMPT = (
        "Ты — голосовой ассистент по имени Максимус. "
        "Отвечай кратко, дружелюбно, без воды. "
        "Если вопрос сложный — отвечай по существу, "
        "но не более 3-4 предложений. "
        "Говори естественно, как человек в разговоре. "
        "Не используй списки и markdown — только текст."
    )

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        system_prompt: str = None,
        max_history: int = 10,
    ):
        """
        api_key: DeepSeek API ключ
        base_url: https://api.deepseek.com
        model: deepseek-chat
        system_prompt: системный промпт (по умолчанию — Максимус)
        max_history: максимум сообщений в истории (user+assistant)
        """
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = (
            base_url or Config.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or Config.DEEPSEEK_MODEL
        self.system_prompt = (
            system_prompt or self.SYSTEM_PROMPT)
        self.max_history = max_history

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY не задан в .env")

        self.history: List[Dict[str, str]] = []
        logger.info(
            "DeepSeek LLM: %s (%s)",
            self.model, self.base_url)

    def clear_history(self) -> None:
        """Очистить историю диалога."""
        self.history = []
        logger.info("История очищена")

    def _build_messages(self, user_text: str) -> List[Dict]:
        """Собрать сообщения для API (system + history + новый вопрос)."""
        messages = [{
            "role": "system",
            "content": self.system_prompt,
        }]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def chat_stream(
        self, user_text: str,
    ) -> Generator[str, None, None]:
        """
        Стриминговый ответ. Yield-ит куски текста по мере генерации.

        Использование:
            for chunk in llm.chat_stream("Привет"):
                print(chunk, end="", flush=True)
        """
        messages = self._build_messages(user_text)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 500,
        }

        full_answer = ""

        try:
            with requests.post(
                url, headers=headers, json=payload,
                stream=True, timeout=60,
            ) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get(
                            "delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_answer += content
                            yield content
                    except (json.JSONDecodeError, KeyError,
                            IndexError):
                        continue

        except requests.exceptions.HTTPError as e:
            logger.error("HTTP ошибка DeepSeek: %s", e)
            if e.response is not None:
                logger.error("Ответ: %s",
                             e.response.text[:500])
            yield f"[Ошибка API: {e}]"
            return
        except Exception as e:
            logger.error("Ошибка LLM: %s", e)
            yield f"[Ошибка: {e}]"
            return

        # Обновляем историю
        self.history.append(
            {"role": "user", "content": user_text})
        self.history.append(
            {"role": "assistant", "content": full_answer})

        # Обрезаем историю
        if len(self.history) > self.max_history * 2:
            self.history = self.history[
                -self.max_history * 2:]

    def chat(self, user_text: str) -> str:
        """
        Обычный (нестриминговый) ответ.

        Возвращает полный текст ответа.
        """
        return "".join(self.chat_stream(user_text))