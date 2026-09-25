# Maximus AI Assistant

Локальный голосовой ассистент с wake word «Максимус».

## Что умеет

- Слушает микрофон постоянно в фоне.
- Реагирует на слово «Максимус» (wake word через openWakeWord).
- Распознаёт речь через Whisper (русский, английский, китайский).
- Отвечает через DeepSeek API (стриминг).
- Озвучивает ответ через Edge TTS (человеческий голос).

## Архитектура

    Микрофон
       ↓
    Wake Word «Максимус» (openWakeWord)
       ↓
    Whisper STT
       ↓
    DeepSeek API
       ↓
    Edge TTS
       ↓
    Колонки

## Установка

См. docs/setup.md.

## Структура

- src/ — модули ассистента
- scripts/ — тренировка wake word
- models/ — ONNX-модели (в .gitignore)
- docs/ — документация
- requirements.txt — зависимости
- .env.example — шаблон конфига

## Лицензия

MIT