"""
Конфиг Maximus AI Assistant.

Читает переменные окружения из .env (python-dotenv).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Загрузить .env из корня проекта
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    """Безопасное чтение env-переменной."""
    return os.getenv(key, default)


class Config:
    """Все настройки ассистента."""

    # === DeepSeek API ===
    DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = _env(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")

    # === Wake Word ===
    WAKE_WORD_MODEL = _env(
        "WAKE_WORD_MODEL", "models/maksimus.onnx")
    WAKE_WORD_THRESHOLD = float(
        _env("WAKE_WORD_THRESHOLD", "0.5"))

    # === Whisper (STT) ===
    WHISPER_MODEL = _env("WHISPER_MODEL", "base")
    WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "ru")
    WHISPER_DEVICE = _env("WHISPER_DEVICE", "cuda")

    # === Edge TTS ===
    TTS_VOICE = _env("TTS_VOICE", "ru-RU-DmitryNeural")

    # === Микрофон ===
    MIC_SAMPLE_RATE = int(_env("MIC_SAMPLE_RATE", "16000"))
    MIC_CHANNELS = int(_env("MIC_CHANNELS", "1"))

    @classmethod
    def validate(cls) -> list:
        """Проверить, что все необходимые настройки на месте.

        Возвращает список ошибок (пустой = всё ок).
        """
        errors = []
        if not cls.DEEPSEEK_API_KEY or \
                cls.DEEPSEEK_API_KEY.startswith("sk-xxxx"):
            errors.append(
                "DEEPSEEK_API_KEY не задан в .env")
        return errors

    @classmethod
    def wake_word_path(cls) -> Path:
        """Абсолютный путь к модели wake word."""
        p = Path(cls.WAKE_WORD_MODEL)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p


if __name__ == "__main__":
    # Быстрая проверка конфига
    print("=== Maximus Config ===")
    print("DeepSeek URL:    ", Config.DEEPSEEK_BASE_URL)
    print("DeepSeek model:  ", Config.DEEPSEEK_MODEL)
    print("DeepSeek key:    ",
          ("*" * 8 + Config.DEEPSEEK_API_KEY[-6:])
          if Config.DEEPSEEK_API_KEY else "(пусто)")
    print("Wake word model: ", Config.wake_word_path())
    print("Wake threshold:  ", Config.WAKE_WORD_THRESHOLD)
    print("Whisper:         ",
          f"{Config.WHISPER_MODEL} ({Config.WHISPER_LANGUAGE}, "
          f"{Config.WHISPER_DEVICE})")
    print("TTS voice:       ", Config.TTS_VOICE)
    print("Микрофон:        ",
          f"{Config.MIC_SAMPLE_RATE} Hz, "
          f"{Config.MIC_CHANNELS} ch")

    errors = Config.validate()
    if errors:
        print()
        print("ОШИБКИ:")
        for e in errors:
            print("  -", e)
    else:
        print()
        print("OK: конфиг валиден")