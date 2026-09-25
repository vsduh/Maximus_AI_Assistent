# Установка Maximus AI Assistant

## Требования

- Windows 10/11 + WSL2 (Ubuntu 22.04) или Linux
- Python 3.10
- NVIDIA GPU (опционально, для быстрой тренировки wake word)
- Микрофон + колонки

## Установка (WSL2)

Шаг 1. Зависимости системы:

    sudo apt update
    sudo apt install -y python3.10-venv python3.10-dev build-essential \
        git libexpat1-dev wget portaudio19-dev

Шаг 2. Клонировать репозиторий:

    git clone https://github.com/vsduh/Maximus_AI_Assistent.git
    cd Maximus_AI_Assistent

Шаг 3. Создать venv:

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

Шаг 4. Установить зависимости:

    pip install -r requirements.txt
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

Шаг 5. Настроить конфиг:

    cp .env.example .env

Отредактировать `.env` — вставить DeepSeek API-ключ.

Шаг 6. Тренировка wake word:

    python scripts/train_wake_word.py

Шаг 7. Запуск:

    python src/assistant.py

## Тренировка wake word

Через `scripts/train_wake_word.py` (openWakeWord + piper-sample-generator).

Параметры:

- target_word: `Maksimus` (фонетически)
- number_of_examples: 1000
- number_of_training_steps: 10000
- false_activation_penalty: 1500

Результат: `models/maksimus.onnx`.

## Микрофон в WSL2

⚠️ WSL2 не имеет прямого доступа к микрофону Windows.

Решения:

- A. Запускать ассистента на Windows (не в WSL) — проще.
- B. Настроить PulseAudio bridge (сложно).
- C. Запускать в Docker с пробросом audio (сложно).

Рекомендация: A — Python на Windows.

## Troubleshooting

- Микрофон не работает → проверь `sounddevice` (`python -m sounddevice`).
- Wake word не срабатывает → перетренируй модель, проверь `WAKE_WORD_THRESHOLD`.
- Whisper долго → поставь модель `tiny` или `base`.
- CUDA не видна → проверь `nvidia-smi` в WSL.