"""
Wake Word «Максимус» на базе openWakeWord.

Слушает микрофон непрерывно, детектирует слово «Максимус»,
возвращает управление в основной цикл ассистента.
"""
import logging
import time

import numpy as np
from openwakeword.model import Model

from .config import Config

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Детектор wake word «Максимус» через openWakeWord."""

    def __init__(
        self,
        model_path: str = None,
        threshold: float = None,
        sample_rate: int = 16000,
        frame_length: int = 1280,  # 80 мс при 16 кГц
    ):
        """
        model_path: путь к maksimus.onnx
        threshold: порог уверенности (0.5 по умолчанию)
        sample_rate: частота дискретизации (16000 для openWakeWord)
        frame_length: длина чанка в сэмплах (1280 = 80 мс)
        """
        self.model_path = model_path or str(
            Config.wake_word_path())
        self.threshold = (
            threshold if threshold is not None
            else Config.WAKE_WORD_THRESHOLD
        )
        self.sample_rate = sample_rate
        self.frame_length = frame_length

        # Загружаем модель openWakeWord
        logger.info("Загрузка модели: %s", self.model_path)
        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
        )

        # Название модели (для доступа к вероятности)
        self.model_name = list(
            self.model.models.keys())[0]
        logger.info(
            "Модель загружена: %s (порог %.2f)",
            self.model_name, self.threshold)

    def predict(self, audio_frame: np.ndarray) -> float:
        """
        Один вызов для одного кадра аудио (int16, 1280 сэмплов).

        Возвращает вероятность wake word (0.0 - 1.0).
        """
        # openWakeWord ожидает int16
        if audio_frame.dtype != np.int16:
            audio_frame = audio_frame.astype(np.int16)

        prediction = self.model.predict(audio_frame)
        return float(prediction.get(self.model_name, 0.0))

    def wait_for_wake_word(self, audio_stream) -> bool:
        """
        Блокирующее ожидание wake word.

        audio_stream: sounddevice.InputStream (или совместимый)
                      с samplerate=16000, channels=1, dtype=int16.

        Возвращает True, когда wake word услышан.
        """
        logger.info("Слушаю... (скажи «Максимус»)")

        while True:
            try:
                # Читаем 1280 сэмплов (80 мс)
                frame, overflowed = audio_stream.read(
                    self.frame_length)

                if overflowed:
                    logger.warning(
                        "Audio buffer overflow (пропущены сэмплы)")

                if frame.ndim > 1:
                    frame = frame[:, 0]

                score = self.predict(frame)

                if score >= self.threshold:
                    logger.info(
                        "✅ Wake word «Максимус» (score=%.3f)",
                        score)
                    return True

            except KeyboardInterrupt:
                logger.info("Прервано пользователем")
                return False
            except Exception as e:
                logger.error("Ошибка wake word: %s", e)
                time.sleep(0.5)

    def reset(self):
        """Сбросить внутреннее состояние модели (например, после длинной паузы)."""
        try:
            self.model.reset()
        except Exception as e:
            logger.debug("reset error: %s", e)