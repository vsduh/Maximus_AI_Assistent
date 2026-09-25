"""
Тренировка wake word «Максимус» через openWakeWord.

Использование:
    python scripts/train_wake_word.py

Результат:
    models/maksimus.onnx

⚠️ Требует:
- Python 3.10
- openWakeWord + piper-sample-generator
- Скачанные датасеты (~10 ГБ)
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_wake_word")

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_WORD = "Maksimus"
OUTPUT_MODEL = MODELS_DIR / "maksimus.onnx"


def check_dependencies() -> bool:
    """Проверить, что все зависимости установлены."""
    logger.info("Проверка зависимостей...")

    try:
        import openwakeword
        logger.info("  openwakeword: %s",
                    openwakeword.__version__)
    except ImportError:
        logger.error("  openwakeword НЕ установлен")
        return False

    try:
        import piper_phonemize
        logger.info("  piper_phonemize: OK")
    except ImportError:
        logger.error("  piper_phonemize НЕ установлен")
        return False

    try:
        import torch
        logger.info("  torch: %s (CUDA: %s)",
                    torch.__version__,
                    torch.cuda.is_available())
    except ImportError:
        logger.error("  torch НЕ установлен")
        return False

    return True


def generate_training_data() -> bool:
    """
    Генерирует синтетические примеры произношения «Maksimus».

    Использует piper-sample-generator.
    """
    logger.info("=" * 60)
    logger.info("Шаг 1: генерация тренировочных примеров")
    logger.info("=" * 60)

    piper_dir = PROJECT_ROOT.parent / "piper-sample-generator"
    if not piper_dir.exists():
        logger.error(
            "piper-sample-generator не найден: %s", piper_dir)
        logger.error(
            "Клонируй: git clone "
            "https://github.com/rhasspy/piper-sample-generator.git")
        return False

    output_dir = PROJECT_ROOT / "training_data" / TARGET_WORD
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(piper_dir / "generate_samples.py"),
        TARGET_WORD,
        "--max-samples", "1000",
        "--batch-size", "1",
        "--output-dir", str(output_dir),
    ]

    logger.info("Команда: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, check=True, cwd=str(piper_dir))
        logger.info("Сгенерировано примеров: %s",
                    len(list(output_dir.glob("*.wav"))))
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Ошибка генерации: %s", e)
        return False
    except Exception as e:
        logger.error("Ошибка: %s", e)
        return False


def train_model() -> bool:
    """
    Обучает модель wake word.

    Использует openWakeWord training pipeline.
    """
    logger.info("=" * 60)
    logger.info("Шаг 2: обучение модели")
    logger.info("=" * 60)

    openww_dir = PROJECT_ROOT.parent / "openWakeWord"
    if not openww_dir.exists():
        logger.error(
            "openWakeWord не найден: %s", openww_dir)
        return False

    # Скачиваем базовые модели (mel + embedding)
    logger.info("Скачивание базовых моделей...")
    try:
        from openwakeword.utils import download_models
        download_models()
    except Exception as e:
        logger.warning("download_models: %s", e)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Запускаем обучение
    train_config = PROJECT_ROOT / "training_config.yml"
    if not train_config.exists():
        logger.error(
            "training_config.yml не найден: %s",
            train_config)
        logger.error(
            "Создай его на основе "
            "openWakeWord/examples/custom_model.yml")
        return False

    cmd = [
        sys.executable,
        str(openww_dir / "openwakeword" / "train.py"),
        "--training_config", str(train_config),
        "--output_dir", str(MODELS_DIR),
    ]

    logger.info("Команда: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, cwd=str(openww_dir))
        logger.info("Модель обучена: %s", OUTPUT_MODEL)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Ошибка обучения: %s", e)
        return False


def main():
    logger.info("=" * 60)
    logger.info("Тренировка wake word «%s»", TARGET_WORD)
    logger.info("=" * 60)

    if not check_dependencies():
        logger.error("Установи зависимости: "
                     "pip install -r requirements.txt")
        sys.exit(1)

    if not generate_training_data():
        logger.error("Не удалось сгенерировать данные")
        sys.exit(1)

    if not train_model():
        logger.error("Не удалось обучить модель")
        sys.exit(1)

    if not OUTPUT_MODEL.exists():
        logger.error(
            "Модель не создана: %s", OUTPUT_MODEL)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("✅ Готово! Модель: %s", OUTPUT_MODEL)
    logger.info("Размер: %.1f КБ",
                OUTPUT_MODEL.stat().st_size / 1024)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()